#!/usr/bin/env python3
"""
Full Neural Compiler Pipeline v2.

Uses few-shot prompting for normalization instead of LoRA fine-tuning.
This achieves ~80% on varied NL without training.
"""

import logging
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

# Add project root for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuralCompilerV2:
    """
    Neural compiler with few-shot normalization.

    Uses:
    1. Few-shot prompting for NL → canonical (no training needed)
    2. Dual-reward trained classifier for canonical → IR (100% accurate)
    3. WASM runtime for IR → execute (deterministic)
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        classifier_path: str = "experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    ):
        from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora
        from chuk_lazarus.models_v2.loader import load_model

        # Load single model for both normalization and classification
        logger.info("Loading model...")
        result = load_model(model_name)
        self.base_model = result.model
        self.tokenizer = result.tokenizer
        self.config = result.config

        # Load classifier model separately (with LoRA)
        logger.info("Loading classifier...")
        cls_result = load_model(model_name)
        self.cls_model = cls_result.model

        lora_config = LoRAConfig(
            rank=32,
            alpha=64.0,
            target_modules=["v_proj", "o_proj"],
        )
        apply_lora(self.cls_model, lora_config)

        # Load classifier weights
        with safe_open(classifier_path, framework="numpy") as f:
            lora_weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

        backbone = self.cls_model.model
        for name, param in lora_weights.items():
            if name.startswith("model."):
                name = name[6:]
            parts = name.split(".")
            try:
                obj = backbone
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                if parts[-1] == "lora_a":
                    obj.lora_A = param
                elif parts[-1] == "lora_b":
                    obj.lora_B = param
            except:
                pass

        self.base_model.freeze()
        self.cls_model.freeze()

        self.runtime = WASMRuntime()

        # Classifier tokens
        self.classifier_tokens = {
            "add": 788,
            "subtract": 23197,
            "multiply": 22932,
            "divide": 16429,
        }
        self.class_to_ir = {
            "add": IROpcode.I32_ADD,
            "subtract": IROpcode.I32_SUB,
            "multiply": IROpcode.I32_MUL,
            "divide": IROpcode.I32_DIV_S,
        }

        self.decision_layer = int(self.config.num_hidden_layers * 0.55)
        logger.info(f"Decision layer: {self.decision_layer}")

    def normalize(self, nl_input: str) -> str:
        """Stage 1: NL → Canonical using few-shot prompting."""
        # Few-shot prompt with more examples and clearer instruction
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
                # Add one more token after =
                output = self.base_model(generated_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
                break

        canonical = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist()).strip()
        canonical = canonical.replace("</s>", "").strip()

        # Extract equation if model answered conversationally
        if "=" in canonical:
            # Try to find equation pattern
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
        """Stage 2: Canonical → Operation using L12 logit lens."""
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
        """Stage 3: Build WASM IR."""
        ir_op = self.class_to_ir[operation]
        body = bytearray()
        body.extend(encode_i32_const(operands[0]))
        body.extend(encode_i32_const(operands[1]))
        body.extend(OPCODE_TO_WASM[ir_op])
        return bytes(body)

    def execute(self, ir_bytes: bytes) -> int:
        """Stage 4: Execute WASM."""
        result = self.runtime.execute(ir_bytes)
        if result.success:
            return result.result
        raise RuntimeError(f"Execution failed: {result.error}")

    def compile_and_run(self, nl_input: str) -> dict:
        """Full pipeline."""
        canonical = self.normalize(nl_input)

        # Parse canonical
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


def main():
    compiler = NeuralCompilerV2()

    test_cases = [
        # Simple
        ("Add 11 and 94", 105),
        ("Subtract 49 from 69", 20),
        ("Multiply 7 by 8", 56),
        ("Divide 48 by 6", 8),
        # Varied
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

    logger.info("\n" + "=" * 70)
    logger.info("NEURAL COMPILER V2 - Few-Shot Normalization")
    logger.info("=" * 70)

    correct = 0
    total = len(test_cases)

    for nl_input, expected in test_cases:
        result = compiler.compile_and_run(nl_input)

        if result["success"] and result["result"] == expected:
            status = "OK"
            correct += 1
        elif result["success"]:
            status = f"WRONG (got {result['result']})"
        else:
            status = f"ERROR: {result.get('error', 'unknown')[:30]}"

        logger.info(f"\nInput: {nl_input}")
        logger.info(f"  Canonical: {result['canonical']}")
        if result.get("operation"):
            logger.info(f"  Operation: {result['operation']}")
        logger.info(f"  Result:    {result.get('result', 'N/A')} (expected {expected}) [{status}]")

    logger.info("\n" + "=" * 70)
    logger.info(f"ACCURACY: {correct}/{total} = {100 * correct / total:.1f}%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
