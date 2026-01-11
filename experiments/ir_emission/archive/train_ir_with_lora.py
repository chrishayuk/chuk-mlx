#!/usr/bin/env python3
"""
Train IR emission using dual-reward trained LoRA weights.

Uses the logit lens approach - the classifier is based on
specific token probabilities at L12, not hidden state clustering.
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
from codebook import CodebookConfig, IROpcode, IRSequenceDecoder
from wasm_runtime import WASMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_with_lora(model_name: str, adapter_path: str):
    """Load model and apply LoRA weights."""
    from chuk_lazarus.models_v2.loader import load_model
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora

    result = load_model(model_name)
    model = result.model
    tokenizer = result.tokenizer

    # First apply LoRA structure (creates LoRALinear wrappers)
    lora_config = LoRAConfig(
        rank=32,  # Must match training config
        alpha=64.0,  # alpha = 2 * rank per trainer
        target_modules=["v_proj", "o_proj"],
    )
    lora_layers = apply_lora(model, lora_config)
    logger.info(f"Applied LoRA structure to {len(lora_layers)} layers")

    # Now load the trained weights
    logger.info(f"Loading LoRA weights from {adapter_path}")
    with safe_open(adapter_path, framework="numpy") as f:
        lora_weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

    # Map saved weights to LoRALinear layers
    # Keys are like: model.layers.0.self_attn.v_proj.lora_a
    backbone = model.model
    applied = 0
    for name, param in lora_weights.items():
        if name.startswith("model."):
            name = name[6:]  # Strip 'model.'

        parts = name.split(".")
        # Navigate to the LoRALinear layer
        try:
            obj = backbone
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)

            # Set the lora_A or lora_B attribute
            attr_name = parts[-1]
            if attr_name == "lora_a":
                obj.lora_A = param
                applied += 1
            elif attr_name == "lora_b":
                obj.lora_B = param
                applied += 1
        except Exception as e:
            logger.warning(f"Failed to apply {name}: {e}")

    logger.info(f"Loaded {applied} LoRA weight matrices")
    model.freeze()
    return model, tokenizer, result.config


def get_intermediate_logits(model, tokenizer, prompt: str, layer_idx: int) -> mx.array:
    """Get logits from intermediate layer using logit lens."""
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

    # Apply norm and LM head
    h_normed = backbone.norm(h)
    head_output = model.lm_head(h_normed)
    logits = head_output.logits if hasattr(head_output, "logits") else head_output

    return logits[0, -1, :]  # Last token


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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--adapter",
        default="experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    )
    parser.add_argument("--data", default="experiments/ir_emission/data/phase1_train.jsonl")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load model with LoRA
    model, tokenizer, model_config = load_model_with_lora(args.model, args.adapter)
    model.freeze()

    num_layers = model_config.num_hidden_layers
    hidden_dim = model_config.hidden_size
    decision_layer = int(num_layers * 0.55)
    logger.info(f"Decision layer: {decision_layer}, hidden_dim: {hidden_dim}")

    # Classifier token IDs (from dual-reward training)
    classifier_tokens = {
        "multiply": 22932,
        "add": 788,
        "subtract": 23197,
        "divide": 16429,
    }

    # Test classifier with LoRA
    logger.info("\nTesting logit-lens classifier...")
    test_prompts = [
        ("7 * 8 = ", "multiply"),
        ("23 + 45 = ", "add"),
        ("50 - 23 = ", "subtract"),
        ("48 / 6 = ", "divide"),
    ]

    for prompt, expected in test_prompts:
        logits = get_intermediate_logits(model, tokenizer, prompt, decision_layer)
        probs = mx.softmax(logits)

        best_class = None
        best_prob = 0
        for class_name, token_id in classifier_tokens.items():
            prob = float(probs[token_id].item())
            if prob > best_prob:
                best_prob = prob
                best_class = class_name

        status = "OK" if best_class == expected else "XX"
        logger.info(f"  {prompt:12} -> {best_class:10} ({best_prob:.1%}) [{status}]")

    # Load training data
    samples = []
    with open(args.data) as f:
        for line in f:
            s = json.loads(line)
            if s.get("operation") in ["add", "sub", "mul", "div"]:
                samples.append(s)
    logger.info(f"\nLoaded {len(samples)} samples")

    # Map our operation names to classifier names
    op_to_class = {"add": "add", "sub": "subtract", "mul": "multiply", "div": "divide"}
    class_to_ir_op = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    # Create IR decoder
    codebook_config = CodebookConfig(
        codebook_size=64,
        hidden_dim=hidden_dim,
        embedding_dim=128,
        max_ir_length=8,
    )
    decoder = IRSequenceDecoder(codebook_config)
    runtime = WASMRuntime()
    optimizer = optim.Adam(learning_rate=args.lr)

    # Training loop
    import random
    random.shuffle(samples)

    logger.info(f"\nTraining IR decoder for {args.steps} steps...")

    for step in range(args.steps):
        batch_idx = [(step * args.batch_size + i) % len(samples) for i in range(args.batch_size)]
        batch = [samples[i] for i in batch_idx]

        hidden_states = []
        target_irs = []
        operands_list = []
        expected_results = []

        for sample in batch:
            h = get_hidden_state(model, tokenizer, sample["prompt"], decision_layer)
            hidden_states.append(h)

            # Build target IR
            ir_op = class_to_ir_op[op_to_class[sample["operation"]]]
            target_ir = [IROpcode.START, IROpcode.SLOT_0, IROpcode.SLOT_1, ir_op, IROpcode.END]
            target_irs.append(target_ir)

            operands_list.append(sample["operands"])
            expected_results.append(sample["expected_result"])

        hidden_states = mx.stack(hidden_states)
        mx.eval(hidden_states)

        # Pad target IRs
        max_len = max(len(ir) for ir in target_irs)
        target_ir_padded = [ir + [IROpcode.PAD] * (max_len - len(ir)) for ir in target_irs]
        target_ir = mx.array(target_ir_padded)

        # Loss function
        def loss_fn(decoder_params):
            decoder.update(decoder_params)
            logits, commitment_loss = decoder(hidden_states, target_ir)
            ce_loss = nn.losses.cross_entropy(
                logits.reshape(-1, codebook_config.codebook_size),
                target_ir.reshape(-1),
                reduction="mean",
            )
            return ce_loss + 0.25 * commitment_loss

        loss, grads = nn.value_and_grad(decoder, loss_fn)(decoder.parameters())
        optimizer.update(decoder, grads)
        mx.eval(decoder.parameters())

        # Compute metrics
        logits, _ = decoder(hidden_states, target_ir)
        predicted_ir = mx.argmax(logits, axis=-1)

        correct = 0
        valid_ir = 0
        batch_size = len(batch)

        for i in range(batch_size):
            ir_indices = predicted_ir[i].tolist()
            operands = operands_list[i]
            expected = expected_results[i]

            try:
                body = decoder.codebook.indices_to_wasm(ir_indices, operands)
                result = runtime.execute(body)

                if result.success:
                    valid_ir += 1
                    if result.result == expected:
                        correct += 1
            except Exception:
                pass

        if (step + 1) % 50 == 0:
            logger.info(
                f"Step {step + 1}: loss={float(loss.item()):.4f}, "
                f"acc={correct/batch_size:.1%}, valid_ir={valid_ir/batch_size:.1%}"
            )

    # Final evaluation
    logger.info("\nFinal evaluation...")
    correct = 0
    valid = 0
    total = min(100, len(samples))

    for sample in samples[:total]:
        h = get_hidden_state(model, tokenizer, sample["prompt"], decision_layer)
        h = h[None, :]
        mx.eval(h)

        ir_indices = decoder.generate(h, temperature=0)
        operands = sample["operands"]
        expected = sample["expected_result"]

        try:
            body = decoder.codebook.indices_to_wasm(ir_indices, operands)
            result = runtime.execute(body)

            if result.success:
                valid += 1
                if result.result == expected:
                    correct += 1
        except Exception:
            pass

    logger.info(f"Accuracy: {correct}/{total} = {correct/total:.1%}")
    logger.info(f"Valid IR: {valid}/{total} = {valid/total:.1%}")


if __name__ == "__main__":
    main()
