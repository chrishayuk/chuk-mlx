#!/usr/bin/env python3
"""
Simple IR emission: just predict the operation from hidden state.

Uses a simple MLP instead of the full autoregressive decoder.
This tests if the hidden states have enough info for IR prediction.
"""

import json
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from codebook import IROpcode
from wasm_runtime import WASMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleIRPredictor(nn.Module):
    """Simple MLP to predict operation type from classifier token logits."""

    def __init__(self, input_dim: int = 4):
        super().__init__()
        # Input: 4 classifier token logits
        # Output: 4 operation types
        self.fc1 = nn.Linear(input_dim, 32)
        self.out = nn.Linear(32, 4)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        return self.out(x)


def load_model_with_lora(model_name: str, adapter_path: str):
    """Load model and apply LoRA weights."""
    from chuk_lazarus.models_v2.loader import load_model
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora

    result = load_model(model_name)
    model = result.model
    tokenizer = result.tokenizer

    lora_config = LoRAConfig(
        rank=32,
        alpha=64.0,
        target_modules=["v_proj", "o_proj"],
    )
    apply_lora(model, lora_config)

    with safe_open(adapter_path, framework="numpy") as f:
        lora_weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

    backbone = model.model
    for name, param in lora_weights.items():
        if name.startswith("model."):
            name = name[6:]
        parts = name.split(".")
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

    model.freeze()
    return model, tokenizer, result.config


def get_hidden_state(model, tokenizer, prompt: str, layer_idx: int) -> mx.array:
    """Get hidden state at intermediate layer."""
    backbone = model.model
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    h = backbone.embed_tokens(input_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(backbone.layers):
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, "hidden_states") else output
        if i == layer_idx:
            break

    return h[0, -1, :]


def get_classifier_features(model, tokenizer, prompt: str, layer_idx: int) -> mx.array:
    """Get classifier-relevant features using logit lens at specific tokens."""
    backbone = model.model
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    h = backbone.embed_tokens(input_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(backbone.layers):
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, "hidden_states") else output
        if i == layer_idx:
            break

    # Apply norm and LM head to get logits
    h_normed = backbone.norm(h)
    head_output = model.lm_head(h_normed)
    logits = head_output.logits if hasattr(head_output, "logits") else head_output

    # Return logits at last token
    return logits[0, -1, :]


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--adapter",
        default="experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    )
    parser.add_argument("--data", default="experiments/ir_emission/data/phase1_train.jsonl")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load model
    model, tokenizer, model_config = load_model_with_lora(args.model, args.adapter)
    hidden_dim = model_config.hidden_size
    decision_layer = int(model_config.num_hidden_layers * 0.55)
    logger.info(f"Decision layer: {decision_layer}, hidden_dim: {hidden_dim}")

    # Load data
    samples = []
    with open(args.data) as f:
        for line in f:
            s = json.loads(line)
            if s.get("operation") in ["add", "sub", "mul", "div"]:
                samples.append(s)
    logger.info(f"Loaded {len(samples)} samples")

    # Operation mapping
    op_to_idx = {"add": 0, "sub": 1, "mul": 2, "div": 3}
    idx_to_ir = {
        0: IROpcode.I32_ADD,
        1: IROpcode.I32_SUB,
        2: IROpcode.I32_MUL,
        3: IROpcode.I32_DIV_S,
    }

    # Classifier token IDs (from dual-reward training)
    classifier_tokens = {
        "multiply": 22932,  # maps to idx 2 (mul)
        "add": 788,         # maps to idx 0
        "subtract": 23197,  # maps to idx 1
        "divide": 16429,    # maps to idx 3
    }
    # Order matters: we want logits in [add, sub, mul, div] order
    classifier_token_ids = [
        classifier_tokens["add"],
        classifier_tokens["subtract"],
        classifier_tokens["multiply"],
        classifier_tokens["divide"],
    ]

    # Create predictor - input is 4 classifier token logits
    predictor = SimpleIRPredictor(input_dim=4)
    optimizer = optim.Adam(learning_rate=args.lr)
    runtime = WASMRuntime()

    random.shuffle(samples)

    logger.info(f"\nTraining simple IR predictor for {args.steps} steps...")

    for step in range(args.steps):
        batch_idx = [(step * args.batch_size + i) % len(samples) for i in range(args.batch_size)]
        batch = [samples[i] for i in batch_idx]

        features = []
        labels = []
        for sample in batch:
            logits = get_classifier_features(model, tokenizer, sample["prompt"], decision_layer)
            # Extract just the 4 classifier token logits
            cls_logits = mx.array([logits[tid] for tid in classifier_token_ids])
            features.append(cls_logits)
            labels.append(op_to_idx[sample["operation"]])

        X = mx.stack(features)
        y = mx.array(labels)
        mx.eval(X)

        def loss_fn(params):
            predictor.update(params)
            logits = predictor(X)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss, grads = nn.value_and_grad(predictor, loss_fn)(predictor.parameters())
        optimizer.update(predictor, grads)
        mx.eval(predictor.parameters())

        # Compute accuracy
        logits = predictor(X)
        preds = mx.argmax(logits, axis=-1)
        acc = float(mx.mean(preds == y).item())

        if (step + 1) % 50 == 0:
            logger.info(f"Step {step + 1}: loss={float(loss.item()):.4f}, acc={acc:.1%}")

    # Final evaluation with execution
    logger.info("\nFinal evaluation with WASM execution...")
    correct = 0
    valid = 0
    total = min(100, len(samples))

    # IR generation helper
    def indices_to_wasm(op_idx, operands):
        """Build WASM for: operand[0] op operand[1]"""
        from codebook import encode_i32_const, OPCODE_TO_WASM

        ir_op = idx_to_ir[op_idx]
        body = bytearray()
        body.extend(encode_i32_const(operands[0]))
        body.extend(encode_i32_const(operands[1]))
        body.extend(OPCODE_TO_WASM[ir_op])
        return bytes(body)

    for sample in samples[:total]:
        logits = get_classifier_features(model, tokenizer, sample["prompt"], decision_layer)
        cls_logits = mx.array([[logits[tid] for tid in classifier_token_ids]])
        mx.eval(cls_logits)

        pred_logits = predictor(cls_logits)
        pred_idx = int(mx.argmax(pred_logits, axis=-1).item())
        true_idx = op_to_idx[sample["operation"]]

        operands = sample["operands"]
        expected = sample["expected_result"]

        try:
            body = indices_to_wasm(pred_idx, operands)
            result = runtime.execute(body)

            if result.success:
                valid += 1
                if result.result == expected:
                    correct += 1
        except Exception:
            pass

    logger.info(f"Operation prediction accuracy: {correct}/{total}")
    logger.info(f"Execution accuracy: {correct}/{total} = {correct/total:.1%}")
    logger.info(f"Valid IR: {valid}/{total} = {valid/total:.1%}")

    # Show confusion matrix
    logger.info("\nPer-operation breakdown:")
    for op, idx in op_to_idx.items():
        op_samples = [s for s in samples[:total] if s["operation"] == op]
        op_correct = 0
        for s in op_samples:
            logits = get_classifier_features(model, tokenizer, s["prompt"], decision_layer)
            cls_logits = mx.array([[logits[tid] for tid in classifier_token_ids]])
            mx.eval(cls_logits)
            pred = int(mx.argmax(predictor(cls_logits), axis=-1).item())
            if pred == idx:
                op_correct += 1
        logger.info(f"  {op}: {op_correct}/{len(op_samples)} = {op_correct/len(op_samples):.1%}")


if __name__ == "__main__":
    main()
