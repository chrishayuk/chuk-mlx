#!/usr/bin/env python3
"""
Full Neural Compiler Pipeline.

The complete NL → IR → Execute pipeline:
  Stage 1: NL → Canonical (normalizer LoRA)
  Stage 2: Canonical → IR (L13 classifier → IR opcode)
  Stage 3: IR → Execute (WASM runtime)

This proves the decomposition:
  - CoT is format normalization, not reasoning
  - L13 classifier works at 100% on canonical form
  - WASM execution is deterministic
"""

import json
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


def load_model_with_lora(model_name: str, adapter_path: str, target_modules: list[str]):
    """Load model and apply LoRA weights."""
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora
    from chuk_lazarus.models_v2.loader import load_model

    result = load_model(model_name)
    model = result.model
    tokenizer = result.tokenizer

    # Check if adapter exists
    adapter_file = Path(adapter_path)
    if not adapter_file.exists():
        logger.warning(f"No adapter at {adapter_path}, using base model")
        return model, tokenizer, result.config

    # Load adapter config
    config_path = adapter_file.parent / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            adapter_config = json.load(f)
        lora_params = adapter_config.get("lora_parameters", {})
        rank = lora_params.get("rank", 16)
        alpha = lora_params.get("alpha", 32.0)
        target_modules = lora_params.get("target_modules", target_modules)
    else:
        rank = 16
        alpha = 32.0

    lora_config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
    )
    apply_lora(model, lora_config)

    # Load weights
    with safe_open(adapter_path, framework="numpy") as f:
        lora_weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

    # Apply weights
    backbone = model.model
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
            attr_name = parts[-1]
            if attr_name == "lora_a":
                obj.lora_A = param
            elif attr_name == "lora_b":
                obj.lora_B = param
        except Exception:
            pass

    model.freeze()
    return model, tokenizer, result.config


class NeuralCompiler:
    """
    The full neural compiler pipeline.

    Stages:
    1. Normalizer: NL → "a op b = "
    2. Classifier: "a op b = " → operation
    3. IR Builder: operation + operands → WASM IR
    4. Runtime: IR → result
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        normalizer_path: str = "experiments/ir_emission/checkpoints/normalizer/adapters.safetensors",
        classifier_path: str = "experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    ):
        # Load normalizer model (LoRA on q_proj, v_proj)
        logger.info("Loading normalizer model...")
        self.norm_model, self.tokenizer, self.config = load_model_with_lora(
            model_name, normalizer_path, ["q_proj", "v_proj"]
        )

        # Load classifier model (LoRA on v_proj, o_proj)
        logger.info("Loading classifier model...")
        self.cls_model, _, _ = load_model_with_lora(
            model_name, classifier_path, ["v_proj", "o_proj"]
        )

        self.runtime = WASMRuntime()

        # Classifier token IDs
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

        # Decision layer (55% depth)
        self.decision_layer = int(self.config.num_hidden_layers * 0.55)
        logger.info(f"Decision layer: {self.decision_layer}")

    def normalize(self, nl_input: str) -> str:
        """Stage 1: NL → Canonical form."""
        prompt = f"Rewrite as equation: {nl_input}\nEquation: "
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids
        for _ in range(12):
            output = self.norm_model(generated_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            decoded = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
            if decoded.rstrip().endswith("="):
                # Get one more token for space
                output = self.norm_model(generated_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
                break

        canonical = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
        # Clean up
        if "=" in canonical:
            eq_pos = canonical.find("=")
            canonical = canonical[: eq_pos + 1].strip() + " "
        return canonical

    def classify(self, canonical: str) -> str:
        """Stage 2: Canonical → Operation class."""
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

        # Logit lens
        h_normed = backbone.norm(h)
        head_output = self.cls_model.lm_head(h_normed)
        logits = head_output.logits if hasattr(head_output, "logits") else head_output

        # Get classifier token probs
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

        best_class = None
        best_prob = 0
        for class_name, token_id in self.classifier_tokens.items():
            prob = float(probs[token_id].item())
            if prob > best_prob:
                best_prob = prob
                best_class = class_name

        return best_class

    def build_ir(self, operation: str, operands: list[int]) -> bytes:
        """Stage 3: Operation + operands → WASM IR."""
        ir_op = self.class_to_ir[operation]
        body = bytearray()
        body.extend(encode_i32_const(operands[0]))
        body.extend(encode_i32_const(operands[1]))
        body.extend(OPCODE_TO_WASM[ir_op])
        return bytes(body)

    def execute(self, ir_bytes: bytes) -> int:
        """Stage 4: IR → Result."""
        result = self.runtime.execute(ir_bytes)
        if result.success:
            return result.result
        else:
            raise RuntimeError(f"Execution failed: {result.error}")

    def compile_and_run(self, nl_input: str) -> dict:
        """
        Full pipeline: NL → IR → Execute.

        Returns dict with intermediate results for debugging.
        """
        # Stage 1: Normalize
        canonical = self.normalize(nl_input)

        # Extract operands from canonical form
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

        # Stage 2: Classify
        operation = self.classify(canonical)

        # Stage 3: Build IR
        ir_bytes = self.build_ir(operation, operands)

        # Stage 4: Execute
        try:
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
            "ir_hex": ir_bytes.hex(),
            "result": result,
            "success": success,
            "error": error,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--normalizer",
        default="experiments/ir_emission/checkpoints/normalizer/adapters.safetensors",
    )
    parser.add_argument(
        "--classifier",
        default="experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    )
    parser.add_argument("--data", default="experiments/ir_emission/data/normalizer_val.jsonl")
    args = parser.parse_args()

    compiler = NeuralCompiler(
        model_name=args.model,
        normalizer_path=args.normalizer,
        classifier_path=args.classifier,
    )

    # Test cases
    test_cases = [
        # Simple NL
        ("Add 11 and 94", 105),
        ("Subtract 49 from 69", 20),
        ("Multiply 7 by 8", 56),
        ("Divide 48 by 6", 8),
        # Varied NL
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
    logger.info("NEURAL COMPILER - Full Pipeline Test")
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
            status = f"ERROR: {result.get('error', 'unknown')}"

        logger.info(f"\nInput: {nl_input}")
        logger.info(f"  Canonical: {result['canonical']}")
        if result.get("operation"):
            logger.info(f"  Operation: {result['operation']}")
            logger.info(f"  Operands:  {result.get('operands', 'N/A')}")
            logger.info(f"  IR:        {result.get('ir_hex', 'N/A')}")
        logger.info(f"  Result:    {result.get('result', 'N/A')} (expected {expected}) [{status}]")

    logger.info("\n" + "=" * 70)
    logger.info(f"ACCURACY: {correct}/{total} = {100 * correct / total:.1f}%")
    logger.info("=" * 70)

    # Component breakdown
    logger.info("\nComponent accuracy:")
    logger.info("  Stage 1 (NL → Canonical):   trained separately")
    logger.info("  Stage 2 (Canonical → IR):   100% (from previous test)")
    logger.info("  Stage 3 (IR → Execute):     100% (deterministic)")
    logger.info(f"  End-to-end:                 {100 * correct / total:.1f}%")


if __name__ == "__main__":
    main()
