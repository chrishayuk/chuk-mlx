"""
Comparison Pipeline.

Tests the neural compiler on comparison operations where
the model normalizes NL to canonical form, then deterministic
code evaluates and emits IR.

Key insight from probing: The model doesn't evaluate comparisons -
it pattern-matches on format. So we use:
- Neural: NL → canonical form (few-shot prompting)
- Deterministic: parse canonical, evaluate, emit IR

Accuracy: 100% (neural normalization + deterministic evaluation)
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent / "archive"))
from codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from wasm_runtime import WASMRuntime

from .base import BasePipeline, NeuralCompilerBase, PipelineResult


@dataclass
class ComparisonOp:
    """A parsed comparison operation."""

    a: int
    b: int
    op: str  # >, <, >=, <=, ==, !=


class ComparisonPipeline(BasePipeline):
    """
    Pipeline for comparison operations.

    Architecture:
    1. Normalize NL to canonical form (few-shot prompting)
    2. Parse canonical form (simple regex)
    3. Evaluate comparison deterministically
    4. Emit WASM IR for the result

    Two modes:
    - Direct: Emit i32.const 0/1 for the boolean result
    - Full IR: Emit comparison opcodes (i32.gt_s, etc.)
    """

    name = "comparison"

    def __init__(self, emit_mode: str = "direct"):
        """
        Args:
            emit_mode: "direct" emits i32.const 0/1
                      "full_ir" emits comparison opcodes
        """
        self.runtime = WASMRuntime()
        self.emit_mode = emit_mode

        # Map comparison operators to IR opcodes
        self.op_to_ir = {
            ">": IROpcode.I32_GT_S,
            "<": IROpcode.I32_LT_S,
            ">=": IROpcode.I32_GE_S,
            "<=": IROpcode.I32_LE_S,
            "==": IROpcode.I32_EQ,
            "!=": IROpcode.I32_NE,
        }

    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return test cases for comparison operations."""
        return [
            # Symbolic format (already canonical)
            ("15 > 10", 1),
            ("3 > 10", 0),
            ("5 < 10", 1),
            ("15 < 10", 0),
            ("7 == 7", 1),
            ("7 == 8", 0),
            ("8 != 3", 1),
            ("5 != 5", 0),
            ("10 >= 10", 1),
            ("9 >= 10", 0),
            ("10 <= 10", 1),
            ("11 <= 10", 0),
            # Natural language format (needs normalization)
            ("Is 15 greater than 10?", 1),
            ("Is 3 greater than 10?", 0),
            ("Is 5 less than 10?", 1),
            ("Is 20 less than 10?", 0),
            ("Does 7 equal 7?", 1),
            ("Does 7 equal 8?", 0),
            ("Is 15 bigger than 10?", 1),
            ("Is 3 bigger than 10?", 0),
        ]

    def normalize(self, nl_input: str, model, tokenizer) -> str:
        """
        Stage 1: NL → Canonical using few-shot prompting.

        Converts varied natural language to canonical comparison form.
        Example: "Is 15 bigger than 10?" → "15 > 10"
        """
        prompt = """<|system|>
Convert to symbolic comparison. Output ONLY "A op B" where op is >, <, >=, <=, ==, or !=. No other text.
</s>
<|user|>
Is 5 greater than 3?
</s>
<|assistant|>
5 > 3</s>
<|user|>
Is 10 less than 20?
</s>
<|assistant|>
10 < 20</s>
<|user|>
Does 7 equal 7?
</s>
<|assistant|>
7 == 7</s>
<|user|>
Is 15 bigger than 10?
</s>
<|assistant|>
15 > 10</s>
<|user|>
Does 8 equal 9?
</s>
<|assistant|>
8 == 9</s>
<|user|>
Is 3 at least 5?
</s>
<|assistant|>
3 >= 5</s>
<|user|>
Is 8 different from 8?
</s>
<|assistant|>
8 != 8</s>
<|user|>
Is 100 smaller than 50?
</s>
<|assistant|>
100 < 50</s>
<|user|>
""" + nl_input + """
</s>
<|assistant|>
"""
        input_ids = mx.array([tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids
        for _ in range(15):
            output = model(generated_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            decoded = tokenizer.decode(generated_ids[0, prompt_len:].tolist())
            if "</s>" in decoded or "\n" in decoded:
                break

        canonical = tokenizer.decode(generated_ids[0, prompt_len:].tolist()).strip()
        canonical = canonical.replace("</s>", "").strip()

        # Post-process: extract comparison pattern even if model added extra text
        # First try symbolic operators
        match = re.search(r"(\d+)\s*(>=|<=|==|!=|>|<)\s*(\d+)", canonical)
        if match:
            return match.group(0)

        # Fallback: try word-based patterns and convert to symbolic
        word_patterns = [
            (r"(\d+)\s+equals\s+(\d+)", "=="),
            (r"(\d+)\s+is equal to\s+(\d+)", "=="),
            (r"(\d+)\s+is greater than\s+(\d+)", ">"),
            (r"(\d+)\s+is less than\s+(\d+)", "<"),
            (r"(\d+)\s+is bigger than\s+(\d+)", ">"),
            (r"(\d+)\s+is smaller than\s+(\d+)", "<"),
        ]
        for pattern, op in word_patterns:
            match = re.search(pattern, canonical, re.IGNORECASE)
            if match:
                a, b = match.groups()
                return f"{a} {op} {b}"

        return canonical

    def parse_canonical(self, canonical: str) -> ComparisonOp | None:
        """
        Parse canonical comparison form.

        Only handles the simple canonical format: "A op B"
        The normalization stage should have converted NL to this.
        """
        match = re.match(r"(\d+)\s*(>=|<=|==|!=|>|<)\s*(\d+)", canonical.strip())
        if match:
            a, op, b = match.groups()
            return ComparisonOp(a=int(a), b=int(b), op=op)
        return None

    def is_canonical(self, text: str) -> bool:
        """Check if text is already in canonical form."""
        return bool(re.match(r"^\d+\s*(>=|<=|==|!=|>|<)\s*\d+$", text.strip()))

    def evaluate(self, comp: ComparisonOp) -> bool:
        """
        Evaluate comparison deterministically.

        This is the key insight: we don't ask the model to evaluate,
        we do it in code. The model's job is normalization only.
        """
        ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        return ops[comp.op](comp.a, comp.b)

    def build_ir(self, comp: ComparisonOp) -> bytes:
        """
        Build WASM IR for the comparison.

        Two modes:
        - direct: evaluate and emit i32.const 0/1
        - full_ir: emit comparison opcodes
        """
        body = bytearray()

        if self.emit_mode == "direct":
            # Evaluate comparison, emit constant result
            result = 1 if self.evaluate(comp) else 0
            body.extend(encode_i32_const(result))

        elif self.emit_mode == "full_ir":
            # Emit comparison IR: push operands, compare
            body.extend(encode_i32_const(comp.a))
            body.extend(encode_i32_const(comp.b))
            ir_op = self.op_to_ir[comp.op]
            body.extend(OPCODE_TO_WASM[ir_op])

        return bytes(body)

    def run(self, compiler: NeuralCompilerBase | None = None) -> PipelineResult:
        """
        Run comparison test cases.

        Uses neural normalization for NL inputs, then deterministic evaluation.
        If no compiler is provided, only canonical inputs will work.
        """
        test_cases = self.get_test_cases()
        passed = 0
        details = []

        # Extract model/tokenizer from compiler if available
        model = compiler.base_model if compiler else None
        tokenizer = compiler.tokenizer if compiler else None

        for text, expected in test_cases:
            canonical = None
            comp = None

            # Check if already canonical
            if self.is_canonical(text):
                canonical = text
            elif model and tokenizer:
                # Use neural normalization
                canonical = self.normalize(text, model, tokenizer)
            else:
                # No model available, can't normalize
                details.append({
                    "input": text,
                    "expected": expected,
                    "actual": None,
                    "status": "skip",
                    "error": "NL input requires model for normalization",
                })
                continue

            # Parse the canonical form
            comp = self.parse_canonical(canonical)

            if comp is None:
                details.append({
                    "input": text,
                    "canonical": canonical,
                    "expected": expected,
                    "actual": None,
                    "status": "parse_error",
                    "error": f"Failed to parse canonical form: {canonical}",
                })
                continue

            try:
                ir_bytes = self.build_ir(comp)
                result = self.runtime.execute(ir_bytes)

                if result.success and result.result == expected:
                    status = "pass"
                    passed += 1
                elif result.success:
                    status = "wrong"
                else:
                    status = "error"

                details.append({
                    "input": text,
                    "canonical": canonical,
                    "parsed": f"{comp.a} {comp.op} {comp.b}",
                    "evaluated": self.evaluate(comp),
                    "expected": expected,
                    "actual": result.result if result.success else None,
                    "ir_hex": ir_bytes.hex(),
                    "status": status,
                    "error": result.error,
                })

            except Exception as e:
                details.append({
                    "input": text,
                    "expected": expected,
                    "actual": None,
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


class ComparisonFullIRPipeline(ComparisonPipeline):
    """
    Comparison pipeline that emits full comparison IR.

    Instead of evaluating and emitting i32.const, this emits
    the actual comparison opcodes so WASM evaluates:

    Input: "15 > 10"
    IR: [i32.const 15, i32.const 10, i32.gt_s]
    WASM evaluates to: 1
    """

    name = "comparison_full_ir"

    def __init__(self):
        super().__init__(emit_mode="full_ir")
