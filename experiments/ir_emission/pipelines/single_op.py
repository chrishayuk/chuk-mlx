"""
Single Operation Pipeline.

Tests the neural compiler on single arithmetic operations:
- Addition, subtraction, multiplication, division
- Simple commands: "Add 11 and 94"
- Varied phrasing: "The sum of 25 and 17 is"
- Word problems: "Janet has 50 apples. She gives away 15."

Uses few-shot normalization to get canonical form, then parses operation
deterministically from the canonical output.

Expected accuracy: 100%
"""

import re

from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime

from .base import BasePipeline, NeuralCompilerBase, PipelineResult


class SingleOpPipeline(BasePipeline):
    """Pipeline for single arithmetic operations.

    Uses deterministic parsing from canonical form rather than ML classifier.
    The few-shot normalization produces output like "50 - 15 = " and we
    extract the operation from the operator character.
    """

    name = "single_op"

    def __init__(self):
        self.runtime = WASMRuntime()
        self.op_to_ir = {
            "+": IROpcode.I32_ADD,
            "-": IROpcode.I32_SUB,
            "*": IROpcode.I32_MUL,
            "/": IROpcode.I32_DIV_S,
        }

    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return test cases for single operations."""
        return [
            # Simple commands
            ("Add 11 and 94", 105),
            ("Subtract 49 from 69", 20),
            ("Multiply 7 by 8", 56),
            ("Divide 48 by 6", 8),
            # Varied phrasing
            ("The sum of 25 and 17 is", 42),
            ("The difference of 100 and 37 is", 63),
            ("What is 12 times 9?", 108),
            ("What is 144 divided by 12?", 12),
            # Word problems
            ("Janet has 50 apples. She gives away 15. How many remain?", 35),
            ("Each box holds 8 items. How many in 7 boxes?", 56),
            ("A tank has 200 gallons. 75 leak out. How much is left?", 125),
            ("Tickets cost 15 dollars each. Cost for 4 tickets?", 60),
        ]

    def parse_canonical(self, canonical: str) -> tuple[int, str, int] | None:
        """Parse canonical form into (a, operation, b).

        Expected formats:
        - "50 - 15 = "
        - "25 + 17 ="
        """
        match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", canonical)
        if match:
            a, op, b = match.groups()
            return int(a), op, int(b)
        return None

    def build_ir(self, a: int, op: str, b: int) -> bytes:
        """Build WASM IR bytecode for single operation."""
        body = bytearray()
        body.extend(encode_i32_const(a))
        body.extend(encode_i32_const(b))
        body.extend(OPCODE_TO_WASM[self.op_to_ir[op]])
        return bytes(body)

    def run(self, compiler: NeuralCompilerBase) -> PipelineResult:
        """Run single-op test cases using deterministic parsing.

        1. Use few-shot normalization to get canonical form
        2. Parse operation from canonical form (deterministic)
        3. Execute via WASM
        """
        test_cases = self.get_test_cases()
        passed = 0
        details = []

        for nl_input, expected in test_cases:
            # Stage 1: Normalize using few-shot prompting
            canonical = compiler.normalize(nl_input)

            # Stage 2: Parse operation deterministically from canonical form
            parsed = self.parse_canonical(canonical)

            if parsed is None:
                details.append({
                    "input": nl_input,
                    "expected": expected,
                    "actual": None,
                    "canonical": canonical,
                    "operation": None,
                    "status": "parse_error",
                    "error": f"Failed to parse canonical form: {canonical}",
                })
                continue

            a, op, b = parsed
            op_name = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[op]

            try:
                # Stage 3: Build IR and execute
                ir_bytes = self.build_ir(a, op, b)
                result = self.runtime.execute(ir_bytes)

                if result.success and result.result == expected:
                    status = "pass"
                    passed += 1
                elif result.success:
                    status = "wrong"
                else:
                    status = "error"

                details.append({
                    "input": nl_input,
                    "expected": expected,
                    "actual": result.result if result.success else None,
                    "canonical": canonical,
                    "operation": op_name,
                    "ir_hex": ir_bytes.hex(),
                    "status": status,
                    "error": result.error,
                })

            except Exception as e:
                details.append({
                    "input": nl_input,
                    "expected": expected,
                    "actual": None,
                    "canonical": canonical,
                    "operation": op_name,
                    "status": "error",
                    "error": str(e),
                })

        total = len(test_cases)
        return PipelineResult(
            pipeline_name=self.name,
            total_tests=total,
            passed=passed,
            failed=total - passed,
            accuracy=passed / total if total > 0 else 0,
            details=details,
        )
