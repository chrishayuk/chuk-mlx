#!/usr/bin/env python3
"""
Train dual-reward classifier on VARIED NL prompts.

Instead of normalizing NL→canonical→classifier, train the classifier
to work directly on varied NL.
"""

import json
import logging
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_samples(path: str) -> list[dict]:
    """Load training samples."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", default="experiments/ir_emission/data/normalizer_train_v2.jsonl")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--checkpoint-dir", default="experiments/ir_emission/checkpoints/classifier_varied")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    from chuk_lazarus.models_v2.loader import load_model
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora

    result = load_model(args.model)
    model = result.model
    tokenizer = result.tokenizer
    config = result.config

    num_layers = config.num_hidden_layers
    decision_layer = int(num_layers * 0.55)
    logger.info(f"Decision layer: {decision_layer}")

    # Apply LoRA
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=float(args.lora_rank * 2),
        target_modules=["v_proj", "o_proj"],
    )
    lora_layers = apply_lora(model, lora_config)
    logger.info(f"Applied LoRA to {len(lora_layers)} layers")

    # Freeze base, train LoRA
    model.freeze()
    for name, layer in lora_layers.items():
        layer.unfreeze()

    # Classifier token IDs - the tokens we want to emerge
    classifier_tokens = {
        "add": 788,         # "add" token
        "sub": 23197,       # "subtract" token
        "mul": 22932,       # "multiply" token
        "div": 16429,       # "divide" token
    }

    # Load data - use nl_input not canonical_output
    samples = load_samples(args.data)
    # Filter to valid operations
    valid_ops = {"add", "sub", "mul", "div"}
    samples = [s for s in samples if s.get("operation") in valid_ops]
    logger.info(f"Loaded {len(samples)} samples")

    optimizer = optim.Adam(learning_rate=args.lr)
    backbone = model.model

    def get_logits_at_layer(prompt: str, layer_idx: int):
        """Get logits at specific layer using logit lens."""
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

        h_normed = backbone.norm(h)
        head_output = model.lm_head(h_normed)
        logits = head_output.logits if hasattr(head_output, "logits") else head_output

        return logits[0, -1, :]  # Last token

    random.shuffle(samples)
    logger.info(f"\nTraining classifier for {args.steps} steps...")

    for step in range(args.steps):
        batch_idx = [(step * args.batch_size + i) % len(samples) for i in range(args.batch_size)]
        batch = [samples[i] for i in batch_idx]

        def loss_fn(model_params):
            model.update(model_params)
            total_loss = mx.array(0.0)

            for sample in batch:
                # Use nl_input (varied NL) not canonical
                prompt = sample["nl_input"]
                op = sample["operation"]
                target_token = classifier_tokens[op]

                logits = get_logits_at_layer(prompt, decision_layer)

                # Cross-entropy: maximize probability of correct classifier token
                target = mx.array([target_token])
                ce = nn.losses.cross_entropy(logits[None, :], target, reduction="mean")
                total_loss = total_loss + ce

            return total_loss / len(batch)

        loss, grads = nn.value_and_grad(model, loss_fn)(model.parameters())
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        if (step + 1) % 50 == 0:
            # Compute accuracy
            correct = 0
            for sample in batch:
                logits = get_logits_at_layer(sample["nl_input"], decision_layer)
                probs = mx.softmax(logits)

                best_class = None
                best_prob = 0
                for op, token_id in classifier_tokens.items():
                    prob = float(probs[token_id].item())
                    if prob > best_prob:
                        best_prob = prob
                        best_class = op

                if best_class == sample["operation"]:
                    correct += 1

            acc = correct / len(batch)
            logger.info(f"Step {step + 1}: loss={float(loss.item()):.4f}, acc={acc:.1%}")

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import save_file
    import numpy as np

    lora_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        arr = np.array(param.astype(mx.float32))
        lora_weights[name] = arr

    save_file(lora_weights, str(checkpoint_dir / "adapters.safetensors"))

    config_dict = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "alpha": float(args.lora_rank * 2),
            "target_modules": ["v_proj", "o_proj"],
        }
    }
    with open(checkpoint_dir / "adapter_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"\nSaved checkpoint to {checkpoint_dir}")

    # Test on varied prompts
    logger.info("\nTesting on varied NL prompts...")
    test_cases = [
        ("What is 12 times 9?", "mul"),
        ("Janet has 50 apples. She gives away 15. How many remain?", "sub"),
        ("What is 144 divided by 12?", "div"),
        ("The sum of 25 and 17 is", "add"),
        ("Each box holds 8 items. How many in 7 boxes?", "mul"),
        ("A tank has 200 gallons. 75 leak out. How much is left?", "sub"),
        ("Tickets cost 15 dollars each. Cost for 4 tickets?", "mul"),
        ("Add 11 and 94", "add"),
        ("Multiply 7 by 8", "mul"),
        ("Divide 48 by 6", "div"),
    ]

    correct = 0
    for prompt, expected in test_cases:
        logits = get_logits_at_layer(prompt, decision_layer)
        probs = mx.softmax(logits)

        best_class = None
        best_prob = 0
        for op, token_id in classifier_tokens.items():
            prob = float(probs[token_id].item())
            if prob > best_prob:
                best_prob = prob
                best_class = op

        status = "OK" if best_class == expected else "XX"
        if best_class == expected:
            correct += 1
        logger.info(f"  {prompt[:50]:50} → {best_class:10} ({best_prob:.1%}) [{status}]")

    logger.info(f"\nAccuracy: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")


if __name__ == "__main__":
    main()
