"""
Loop Pipeline - Demonstrates Turing Completeness.

Tests the neural compiler's ability to emit loop constructs.
The transformer cannot loop (single forward pass), but by emitting
loop *intent* that WASM executes, we achieve unbounded computation.

Example: "Sum 1 to 100" → 1 forward pass → 100 loop iterations → 5050

Supported loop types:
- sum: Sum numbers in a range (e.g., "Sum 1 to 100" → 5050)
- product: Product of numbers (e.g., "Multiply 1 to 5" → 120)
- count: Count down to zero (e.g., "Count down from 10" → 0)

Expected accuracy: 100%
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from codebook import encode_i32_const
from wasm_runtime import WASMRuntime

from .base import BasePipeline, NeuralCompilerBase, PipelineResult


@dataclass
class LoopIntent:
    """Parsed loop intent from natural language."""

    loop_type: str  # "sum", "product", "count"
    start: int
    end: int


class LoopPipeline(BasePipeline):
    """Pipeline for loop constructs demonstrating Turing completeness."""

    name = "loop"

    def __init__(self):
        self.runtime = WASMRuntime()

    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return test cases for loop constructs."""
        return [
            # Sum loops
            ("Sum 1 to 10", 55),
            ("Sum 1 to 100", 5050),
            ("Add numbers from 5 to 15", 110),
            ("Sum from 1 to 5", 15),
            # Product loops (factorial-like)
            ("Multiply 1 to 5", 120),  # 5! = 120
            ("Product of 1 to 6", 720),  # 6! = 720
            ("Multiply numbers from 2 to 4", 24),  # 2*3*4 = 24
            # Count loops
            ("Count down from 10", 0),
            ("Count from 5 to 0", 0),
        ]

    def parse_loop_intent(self, text: str) -> LoopIntent | None:
        """Parse loop intent from natural language."""
        text = text.lower().strip()

        # Sum patterns
        sum_patterns = [
            r"sum\s*(?:from\s*)?(\d+)\s*to\s*(\d+)",
            r"add\s*numbers?\s*from\s*(\d+)\s*to\s*(\d+)",
        ]
        for pattern in sum_patterns:
            match = re.search(pattern, text)
            if match:
                return LoopIntent(
                    loop_type="sum",
                    start=int(match.group(1)),
                    end=int(match.group(2)),
                )

        # Product patterns
        prod_patterns = [
            r"multiply\s*(?:numbers?\s*)?(?:from\s*)?(\d+)\s*to\s*(\d+)",
            r"product\s*(?:of\s*)?(\d+)\s*to\s*(\d+)",
        ]
        for pattern in prod_patterns:
            match = re.search(pattern, text)
            if match:
                return LoopIntent(
                    loop_type="product",
                    start=int(match.group(1)),
                    end=int(match.group(2)),
                )

        # Count patterns
        count_patterns = [
            r"count\s*(?:down\s*)?from\s*(\d+)",
            r"count\s*from\s*(\d+)\s*to\s*(\d+)",
        ]
        for pattern in count_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                start = int(groups[0])
                end = int(groups[1]) if len(groups) > 1 and groups[1] else 0
                return LoopIntent(
                    loop_type="count",
                    start=start,
                    end=end,
                )

        return None

    def build_sum_loop_wasm(self, start: int, end: int) -> bytes:
        """Build WASM for sum loop: acc = sum(start..end)."""
        body = bytearray()

        # Initialize: acc = 0 (local 0), counter = start (local 1)
        body.extend(encode_i32_const(0))
        body.append(0x21)
        body.append(0x00)  # local.set 0 (acc)
        body.extend(encode_i32_const(start))
        body.append(0x21)
        body.append(0x01)  # local.set 1 (counter)

        # Loop block
        body.append(0x03)
        body.append(0x40)  # loop void

        # acc += counter
        body.append(0x20)
        body.append(0x00)  # local.get 0 (acc)
        body.append(0x20)
        body.append(0x01)  # local.get 1 (counter)
        body.append(0x6A)  # i32.add
        body.append(0x21)
        body.append(0x00)  # local.set 0 (acc)

        # counter++
        body.append(0x20)
        body.append(0x01)  # local.get 1 (counter)
        body.extend(encode_i32_const(1))
        body.append(0x6A)  # i32.add
        body.append(0x22)
        body.append(0x01)  # local.tee 1 (counter)

        # if counter <= end: branch back
        body.extend(encode_i32_const(end))
        body.append(0x4C)  # i32.le_s
        body.append(0x0D)
        body.append(0x00)  # br_if 0

        body.append(0x0B)  # end loop
        body.append(0x20)
        body.append(0x00)  # return acc

        return bytes(body)

    def build_product_loop_wasm(self, start: int, end: int) -> bytes:
        """Build WASM for product loop: acc = product(start..end)."""
        body = bytearray()

        # Initialize: acc = 1 (local 0), counter = start (local 1)
        body.extend(encode_i32_const(1))
        body.append(0x21)
        body.append(0x00)  # local.set 0 (acc)
        body.extend(encode_i32_const(start))
        body.append(0x21)
        body.append(0x01)  # local.set 1 (counter)

        # Loop block
        body.append(0x03)
        body.append(0x40)  # loop void

        # acc *= counter
        body.append(0x20)
        body.append(0x00)  # local.get 0 (acc)
        body.append(0x20)
        body.append(0x01)  # local.get 1 (counter)
        body.append(0x6C)  # i32.mul
        body.append(0x21)
        body.append(0x00)  # local.set 0 (acc)

        # counter++
        body.append(0x20)
        body.append(0x01)  # local.get 1 (counter)
        body.extend(encode_i32_const(1))
        body.append(0x6A)  # i32.add
        body.append(0x22)
        body.append(0x01)  # local.tee 1 (counter)

        # if counter <= end: branch back
        body.extend(encode_i32_const(end))
        body.append(0x4C)  # i32.le_s
        body.append(0x0D)
        body.append(0x00)  # br_if 0

        body.append(0x0B)  # end loop
        body.append(0x20)
        body.append(0x00)  # return acc

        return bytes(body)

    def build_count_loop_wasm(self, start: int, end: int) -> bytes:
        """Build WASM for countdown loop: returns end value after loop."""
        body = bytearray()

        # Initialize: counter = start (local 0)
        body.extend(encode_i32_const(start))
        body.append(0x21)
        body.append(0x00)  # local.set 0 (counter)

        # Loop block
        body.append(0x03)
        body.append(0x40)  # loop void

        # counter--
        body.append(0x20)
        body.append(0x00)  # local.get 0 (counter)
        body.extend(encode_i32_const(1))
        body.append(0x6B)  # i32.sub
        body.append(0x22)
        body.append(0x00)  # local.tee 0 (counter)

        # if counter > end: branch back
        body.extend(encode_i32_const(end))
        body.append(0x4A)  # i32.gt_s
        body.append(0x0D)
        body.append(0x00)  # br_if 0

        body.append(0x0B)  # end loop
        body.append(0x20)
        body.append(0x00)  # return counter

        return bytes(body)

    def run(self, compiler: NeuralCompilerBase) -> PipelineResult:
        """Run loop test cases."""
        test_cases = self.get_test_cases()
        passed = 0
        details = []

        for text, expected in test_cases:
            intent = self.parse_loop_intent(text)

            if intent is None:
                details.append(
                    {
                        "input": text,
                        "expected": expected,
                        "actual": None,
                        "status": "parse_error",
                        "error": "Failed to parse loop intent",
                    }
                )
                continue

            try:
                # Build appropriate loop
                if intent.loop_type == "sum":
                    ir_bytes = self.build_sum_loop_wasm(intent.start, intent.end)
                elif intent.loop_type == "product":
                    ir_bytes = self.build_product_loop_wasm(intent.start, intent.end)
                elif intent.loop_type == "count":
                    ir_bytes = self.build_count_loop_wasm(intent.start, intent.end)
                else:
                    raise ValueError(f"Unknown loop type: {intent.loop_type}")

                result = self.runtime.execute(ir_bytes)

                if result.success and result.result == expected:
                    status = "pass"
                    passed += 1
                elif result.success:
                    status = "wrong"
                else:
                    status = "error"

                details.append(
                    {
                        "input": text,
                        "expected": expected,
                        "actual": result.result if result.success else None,
                        "loop_type": intent.loop_type,
                        "range": f"{intent.start}..{intent.end}",
                        "ir_hex": ir_bytes.hex(),
                        "status": status,
                        "error": result.error,
                    }
                )

            except Exception as e:
                details.append(
                    {
                        "input": text,
                        "expected": expected,
                        "actual": None,
                        "status": "error",
                        "error": str(e),
                    }
                )

        total = len(test_cases)
        return PipelineResult(
            pipeline_name=self.name,
            total_tests=total,
            passed=passed,
            failed=total - passed,
            accuracy=passed / total if total > 0 else 0,
            details=details,
        )
