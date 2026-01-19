"""
Base class for neural compiler pipelines.

Extracts shared logic from full_pipeline_v2.py for reuse across
single_op, multi_op, and loop pipelines.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from running a pipeline."""

    pipeline_name: str
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "details": self.details,
        }


class NeuralCompilerBase:
    """
    Base neural compiler with few-shot normalization.

    Shared logic for:
    1. Few-shot prompting for NL → canonical
    2. Logit lens classification for canonical → IR operation
    3. WASM execution for IR → result
    """

    def __init__(
        self,
        base_model,
        cls_model,
        tokenizer,
        config,
        classifier_tokens: dict[str, int],
        decision_layer: int,
    ):
        self.base_model = base_model
        self.cls_model = cls_model
        self.tokenizer = tokenizer
        self.config = config
        self.classifier_tokens = classifier_tokens
        self.decision_layer = decision_layer

        self.class_to_ir = {
            "add": IROpcode.I32_ADD,
            "subtract": IROpcode.I32_SUB,
            "multiply": IROpcode.I32_MUL,
            "divide": IROpcode.I32_DIV_S,
        }

        self.runtime = WASMRuntime()

    def normalize(self, nl_input: str) -> str:
        """Stage 1: NL → Canonical using few-shot prompting."""
        prompt = f"""<|system|>
You convert word problems to math equations. Output ONLY the equation in format "number operator number = " with no other text.
</s>
<|user|>
What is 5 times 3?
</s>
<|assistant|>
5 * 3 = </s>
<|user|>
Janet has 20 apples. She gives away 7.
</s>
<|assistant|>
20 - 7 = </s>
<|user|>
Subtract 10 from 50
</s>
<|assistant|>
50 - 10 = </s>
<|user|>
The difference of 100 and 30 is
</s>
<|assistant|>
100 - 30 = </s>
<|user|>
Each box has 6 items. How many in 8 boxes?
</s>
<|assistant|>
6 * 8 = </s>
<|user|>
A tank has 150 gallons. 40 leak out.
</s>
<|assistant|>
150 - 40 = </s>
<|user|>
Tickets cost 20 dollars. Cost for 3?
</s>
<|assistant|>
20 * 3 = </s>
<|user|>
{nl_input}
</s>
<|assistant|>
"""
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids
        for _ in range(15):
            output = self.base_model(generated_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            decoded = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
            if "</s>" in decoded or "\n" in decoded:
                break
            if "=" in decoded and len(decoded.strip()) > 3:
                output = self.base_model(generated_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
                break

        canonical = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist()).strip()
        canonical = canonical.replace("</s>", "").strip()

        if "=" in canonical:
            match = re.search(r"(\d+)\s*([+\-*/×÷x])\s*(\d+)\s*=", canonical)
            if match:
                a, op, b = match.groups()
                op = op.replace("×", "*").replace("÷", "/").replace("x", "*")
                canonical = f"{a} {op} {b} = "
            else:
                eq_pos = canonical.find("=")
                canonical = canonical[: eq_pos + 1].strip() + " "

        return canonical

    def classify(self, canonical: str) -> str:
        """Stage 2: Canonical → Operation using logit lens at decision layer."""
        backbone = self.cls_model.model
        tokens = self.tokenizer.encode(canonical)
        input_ids = mx.array([tokens])

        h = backbone.embed_tokens(input_ids)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(backbone.layers):
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output
            if i == self.decision_layer:
                break

        h_normed = backbone.norm(h)
        head_output = self.cls_model.lm_head(h_normed)
        logits = head_output.logits if hasattr(head_output, "logits") else head_output

        probs = mx.softmax(logits[0, -1, :])

        best_class = None
        best_prob = 0
        for class_name, token_id in self.classifier_tokens.items():
            prob = float(probs[token_id].item())
            if prob > best_prob:
                best_prob = prob
                best_class = class_name

        return best_class

    def build_ir(self, operation: str, operands: list[int]) -> bytes:
        """Stage 3: Build WASM IR bytecode."""
        ir_op = self.class_to_ir[operation]
        body = bytearray()
        body.extend(encode_i32_const(operands[0]))
        body.extend(encode_i32_const(operands[1]))
        body.extend(OPCODE_TO_WASM[ir_op])
        return bytes(body)

    def execute(self, ir_bytes: bytes) -> int:
        """Stage 4: Execute WASM and return result."""
        result = self.runtime.execute(ir_bytes)
        if result.success:
            return result.result
        raise RuntimeError(f"Execution failed: {result.error}")

    def compile_and_run(self, nl_input: str) -> dict:
        """Full pipeline: NL → canonical → classify → IR → execute."""
        canonical = self.normalize(nl_input)

        match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", canonical)
        if not match:
            return {
                "input": nl_input,
                "canonical": canonical,
                "success": False,
                "error": "Failed to parse canonical form",
            }

        a, op_char, b = match.groups()
        operands = [int(a), int(b)]

        operation = self.classify(canonical)

        try:
            ir_bytes = self.build_ir(operation, operands)
            result = self.execute(ir_bytes)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        return {
            "input": nl_input,
            "canonical": canonical,
            "operands": operands,
            "operation": operation,
            "ir_hex": ir_bytes.hex() if success else None,
            "result": result,
            "success": success,
            "error": error,
        }


class BasePipeline(ABC):
    """Abstract base for pipeline implementations."""

    name: str = "base"

    @abstractmethod
    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return list of (input, expected_output) test cases."""
        pass

    def run(self, compiler: NeuralCompilerBase) -> PipelineResult:
        """Run all test cases and return results."""
        test_cases = self.get_test_cases()
        passed = 0
        details = []

        for nl_input, expected in test_cases:
            result = compiler.compile_and_run(nl_input)

            if result["success"] and result["result"] == expected:
                status = "pass"
                passed += 1
            elif result["success"]:
                status = "wrong"
            else:
                status = "error"

            details.append(
                {
                    "input": nl_input,
                    "expected": expected,
                    "actual": result.get("result"),
                    "canonical": result.get("canonical"),
                    "operation": result.get("operation"),
                    "status": status,
                    "error": result.get("error"),
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
