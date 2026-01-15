"""
Multi-Operation Pipeline.

Tests the neural compiler on sequential operations where
the result of one operation feeds into the next.

Example: "16 - 3, then multiply by 5" → (16-3)*5 = 65

Uses stack-based WASM execution - the result stays on stack
and the next operation just pushes the second operand.

Expected accuracy: ~75% (parenthesized expressions need improved parsing)
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from wasm_runtime import WASMRuntime

from .base import BasePipeline, NeuralCompilerBase, PipelineResult


@dataclass
class MultiOpStep:
    """A single step in a multi-op chain."""

    a: int | None  # None if using result from previous step
    b: int
    operation: str


class MultiOpPipeline(BasePipeline):
    """Pipeline for multi-operation chains."""

    name = "multi_op"

    def __init__(self):
        self.runtime = WASMRuntime()
        self.op_to_ir = {
            "+": IROpcode.I32_ADD,
            "-": IROpcode.I32_SUB,
            "*": IROpcode.I32_MUL,
            "/": IROpcode.I32_DIV_S,
        }

    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return test cases for multi-op chains."""
        return [
            ("16 - 3, then multiply by 5", 65),  # (16-3)*5
            ("Add 10 and 20, then subtract 5", 25),  # (10+20)-5
            ("Multiply 4 by 7, then add 8", 36),  # (4*7)+8
            ("Start with 50, subtract 20, divide by 3", 10),  # (50-20)/3
            ("(8 + 4) * 3", 36),  # Parenthesized
            ("(20 - 5) * 2", 30),  # Parenthesized
            ("6 * 7, then add 10", 52),  # (6*7)+10
            ("100 - 40, then divide by 2", 30),  # (100-40)/2
        ]

    def parse_multi_op(self, text: str) -> list[MultiOpStep] | None:
        """Parse multi-op expression into steps."""
        text = text.lower().strip()

        # Try parenthesized format: (a op b) op c
        paren_match = re.match(r"\((\d+)\s*([+\-*/])\s*(\d+)\)\s*([+\-*/])\s*(\d+)", text)
        if paren_match:
            a, op1, b, op2, c = paren_match.groups()
            return [
                MultiOpStep(a=int(a), b=int(b), operation=op1),
                MultiOpStep(a=None, b=int(c), operation=op2),
            ]

        # Try "X op Y, then op Z" format
        chain_match = re.match(
            r"(\d+)\s*([+\-*/]|minus|plus|times|divided by)\s*(\d+)[,.]?\s*then\s*(\w+)\s*(?:by\s*)?(\d+)",
            text,
        )
        if chain_match:
            a, op1, b, op2_word, c = chain_match.groups()
            op1 = self._word_to_op(op1)
            op2 = self._word_to_op(op2_word)
            if op1 and op2:
                return [
                    MultiOpStep(a=int(a), b=int(b), operation=op1),
                    MultiOpStep(a=None, b=int(c), operation=op2),
                ]

        # Try "operation X and Y, then op Z"
        word_match = re.match(
            r"(\w+)\s*(\d+)\s*(?:and|by)\s*(\d+)[,.]?\s*then\s*(\w+)\s*(\d+)", text
        )
        if word_match:
            op1_word, a, b, op2_word, c = word_match.groups()
            op1 = self._word_to_op(op1_word)
            op2 = self._word_to_op(op2_word)
            if op1 and op2:
                return [
                    MultiOpStep(a=int(a), b=int(b), operation=op1),
                    MultiOpStep(a=None, b=int(c), operation=op2),
                ]

        # Try "start with X, op Y, op Z"
        start_match = re.match(
            r"start with\s*(\d+)[,.]?\s*(\w+)\s*(\d+)[,.]?\s*(\w+)\s*(?:by\s*)?(\d+)", text
        )
        if start_match:
            a, op1_word, b, op2_word, c = start_match.groups()
            op1 = self._word_to_op(op1_word)
            op2 = self._word_to_op(op2_word)
            if op1 and op2:
                return [
                    MultiOpStep(a=int(a), b=int(b), operation=op1),
                    MultiOpStep(a=None, b=int(c), operation=op2),
                ]

        return None

    def _word_to_op(self, word: str) -> str | None:
        """Convert operation word to symbol."""
        word = word.lower().strip()
        mapping = {
            "add": "+",
            "plus": "+",
            "+": "+",
            "subtract": "-",
            "minus": "-",
            "-": "-",
            "multiply": "*",
            "times": "*",
            "*": "*",
            "divide": "/",
            "divided": "/",
            "/": "/",
        }
        return mapping.get(word)

    def build_chain_ir(self, steps: list[MultiOpStep]) -> bytes:
        """Build WASM IR for a chain of operations."""
        body = bytearray()

        for i, step in enumerate(steps):
            if i == 0:
                # First step: push both operands
                body.extend(encode_i32_const(step.a))
                body.extend(encode_i32_const(step.b))
            else:
                # Later steps: result already on stack, just push second operand
                body.extend(encode_i32_const(step.b))

            ir_op = self.op_to_ir[step.operation]
            body.extend(OPCODE_TO_WASM[ir_op])

        return bytes(body)

    def run(self, compiler: NeuralCompilerBase) -> PipelineResult:
        """Run multi-op test cases."""
        test_cases = self.get_test_cases()
        passed = 0
        details = []

        for text, expected in test_cases:
            steps = self.parse_multi_op(text)

            if steps is None:
                details.append(
                    {
                        "input": text,
                        "expected": expected,
                        "actual": None,
                        "status": "parse_error",
                        "error": "Failed to parse multi-op expression",
                    }
                )
                continue

            try:
                ir_bytes = self.build_chain_ir(steps)
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
                        "steps": [{"a": s.a, "b": s.b, "op": s.operation} for s in steps],
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
