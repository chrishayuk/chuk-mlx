#!/usr/bin/env python3
"""
Train NL → Canonical normalizer.

The frontend of the neural compiler. Transforms varied NL expressions
into canonical "a op b = " form that the L13 classifier handles at 100%.

This is what CoT actually does - format normalization.
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
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_prompt(nl_input: str) -> str:
    """Format input for the normalizer."""
    return f"Rewrite as equation: {nl_input}\nEquation: "


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--train-data", default="experiments/ir_emission/data/normalizer_train.jsonl"
    )
    parser.add_argument("--val-data", default="experiments/ir_emission/data/normalizer_val.jsonl")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--checkpoint-dir", default="experiments/ir_emission/checkpoints/normalizer"
    )
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora
    from chuk_lazarus.models_v2.loader import load_model

    result = load_model(args.model)
    model = result.model
    tokenizer = result.tokenizer

    # Apply LoRA for fine-tuning
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=float(args.lora_rank * 2),
        target_modules=["q_proj", "v_proj"],  # Different targets for generation
    )
    lora_layers = apply_lora(model, lora_config)
    logger.info(f"Applied LoRA to {len(lora_layers)} layers")

    # Freeze base model, train only LoRA
    model.freeze()
    for name, lora_layer in lora_layers.items():
        lora_layer.unfreeze()

    # Count trainable params
    trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load data
    train_samples = load_samples(args.train_data)
    val_samples = load_samples(args.val_data)
    logger.info(f"Training samples: {len(train_samples)}, Validation: {len(val_samples)}")

    optimizer = optim.Adam(learning_rate=args.lr)

    def get_batch(samples: list[dict], batch_size: int, step: int):
        """Get a batch of training examples."""
        batch_indices = [(step * batch_size + i) % len(samples) for i in range(batch_size)]
        batch = [samples[i] for i in batch_indices]

        # Tokenize inputs and targets
        input_ids_list = []
        target_ids_list = []
        max_len = 0

        for sample in batch:
            prompt = format_prompt(sample["nl_input"])
            target = sample["canonical_output"]
            full_text = prompt + target

            input_ids = tokenizer.encode(prompt)
            full_ids = tokenizer.encode(full_text)

            # Target is only the canonical output part
            target_ids = [-100] * len(input_ids) + full_ids[len(input_ids) :]

            input_ids_list.append(full_ids)
            target_ids_list.append(target_ids)
            max_len = max(max_len, len(full_ids))

        # Pad sequences
        pad_id = (
            tokenizer.pad_token_id
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id
            else 0
        )
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad_len
            target_ids_list[i] = target_ids_list[i] + [-100] * pad_len

        return mx.array(input_ids_list), mx.array(target_ids_list)

    def loss_fn(model, input_ids, target_ids):
        """Compute cross-entropy loss on target tokens only."""
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_targets = target_ids[:, 1:]

        # Mask out padding and prompt tokens
        mask = shift_targets != -100

        # Compute loss only on valid targets
        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.reshape(-1, vocab_size)
        flat_targets = mx.where(shift_targets == -100, 0, shift_targets).reshape(-1)

        ce = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")
        ce = ce.reshape(shift_targets.shape)

        # Apply mask
        masked_loss = mx.where(mask, ce, 0.0)
        loss = mx.sum(masked_loss) / (mx.sum(mask) + 1e-8)

        return loss

    # Training loop
    random.shuffle(train_samples)
    logger.info(f"\nTraining normalizer for {args.steps} steps...")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(args.steps):
        input_ids, target_ids = get_batch(train_samples, args.batch_size, step)
        mx.eval(input_ids, target_ids)

        loss, grads = loss_and_grad(model, input_ids, target_ids)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        if (step + 1) % 50 == 0:
            # Validation
            val_batch_input, val_batch_target = get_batch(val_samples, 8, step)
            mx.eval(val_batch_input, val_batch_target)
            val_loss = loss_fn(model, val_batch_input, val_batch_target)

            logger.info(
                f"Step {step + 1}: train_loss={float(loss.item()):.4f}, val_loss={float(val_loss.item()):.4f}"
            )

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    import numpy as np
    from safetensors.numpy import save_file

    lora_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        # Convert to float32 for safetensors compatibility
        arr = np.array(param.astype(mx.float32))
        lora_weights[name] = arr

    save_file(lora_weights, str(checkpoint_dir / "adapters.safetensors"))

    # Save config
    config = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "alpha": float(args.lora_rank * 2),
            "target_modules": ["q_proj", "v_proj"],
        }
    }
    with open(checkpoint_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nSaved checkpoint to {checkpoint_dir}")

    # Test generation
    logger.info("\nTesting normalization...")
    test_cases = [
        "Add 11 and 94",
        "The difference of 69 and 49 is",
        "Multiply 7 by 8",
        "What is 48 divided by 6?",
        "Janet has 25 apples. She gives away 10. How many remain?",
        "Each box holds 5 items. How many in 12 boxes?",
    ]

    # Get token IDs for stopping
    newline_id = tokenizer.encode("\n")[0] if tokenizer.encode("\n") else None

    for nl_input in test_cases:
        prompt = format_prompt(nl_input)
        input_ids = mx.array([tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        # Generate - we only want "a op b = " which is ~8 tokens max
        generated_ids = input_ids
        for _ in range(12):  # Max 12 tokens for "123 + 456 = "
            output = model(generated_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            # Stop at newline
            tok_id = int(next_token.item())
            if newline_id and tok_id == newline_id:
                break

            # Also stop after "= " pattern (space after equals)
            decoded_so_far = tokenizer.decode(generated_ids[0, prompt_len:].tolist())
            if decoded_so_far.rstrip().endswith("="):
                # Add one more token to get the space, then stop
                output = model(generated_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
                break

        # Decode only the generated part
        generated_part = tokenizer.decode(generated_ids[0, prompt_len:].tolist())
        # Clean up - just get first equation pattern
        equation = generated_part.strip()
        # Stop at first complete equation pattern
        if "=" in equation:
            # Take up to first "= " or "=\n"
            eq_pos = equation.find("=")
            equation = equation[: eq_pos + 1].strip() + " "

        logger.info(f"  {nl_input[:40]:40} → {equation}")


if __name__ == "__main__":
    main()
