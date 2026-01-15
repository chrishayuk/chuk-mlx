#!/usr/bin/env python3
"""
Train NL → Canonical normalizer v2.

Uses chat format and explicit instruction to improve translation.
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


def format_prompt(nl_input: str) -> str:
    """Format with explicit instruction for translation."""
    return f"""<|system|>
You translate math problems to equations. Output only the equation, nothing else.
</s>
<|user|>
{nl_input}
</s>
<|assistant|>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--train-data", default="experiments/ir_emission/data/normalizer_train_v2.jsonl"
    )
    parser.add_argument(
        "--val-data", default="experiments/ir_emission/data/normalizer_val_v2.jsonl"
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--checkpoint-dir", default="experiments/ir_emission/checkpoints/normalizer_v2"
    )
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora
    from chuk_lazarus.models_v2.loader import load_model

    result = load_model(args.model)
    model = result.model
    tokenizer = result.tokenizer

    # Apply LoRA - more modules for better translation
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=float(args.lora_rank * 2),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    lora_layers = apply_lora(model, lora_config)
    logger.info(f"Applied LoRA to {len(lora_layers)} layers")

    model.freeze()
    for name, layer in lora_layers.items():
        layer.unfreeze()

    trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load data
    def load_samples(path):
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    train_samples = load_samples(args.train_data)
    val_samples = load_samples(args.val_data)
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    optimizer = optim.Adam(learning_rate=args.lr)

    def get_batch(samples, batch_size, step):
        batch_indices = [(step * batch_size + i) % len(samples) for i in range(batch_size)]
        batch = [samples[i] for i in batch_indices]

        input_ids_list = []
        target_ids_list = []
        max_len = 0

        for sample in batch:
            prompt = format_prompt(sample["nl_input"])
            target = sample["canonical_output"]
            full_text = prompt + target

            input_ids = tokenizer.encode(prompt)
            full_ids = tokenizer.encode(full_text)

            target_ids = [-100] * len(input_ids) + full_ids[len(input_ids) :]

            input_ids_list.append(full_ids)
            target_ids_list.append(target_ids)
            max_len = max(max_len, len(full_ids))

        pad_id = 0
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            input_ids_list[i] = input_ids_list[i] + [pad_id] * pad_len
            target_ids_list[i] = target_ids_list[i] + [-100] * pad_len

        return mx.array(input_ids_list), mx.array(target_ids_list)

    def loss_fn(model, input_ids, target_ids):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output

        shift_logits = logits[:, :-1, :]
        shift_targets = target_ids[:, 1:]

        mask = shift_targets != -100

        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.reshape(-1, vocab_size)
        flat_targets = mx.where(shift_targets == -100, 0, shift_targets).reshape(-1)

        ce = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")
        ce = ce.reshape(shift_targets.shape)

        masked_loss = mx.where(mask, ce, 0.0)
        loss = mx.sum(masked_loss) / (mx.sum(mask) + 1e-8)

        return loss

    random.shuffle(train_samples)
    logger.info(f"\nTraining for {args.steps} steps...")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(args.steps):
        input_ids, target_ids = get_batch(train_samples, args.batch_size, step)
        mx.eval(input_ids, target_ids)

        loss, grads = loss_and_grad(model, input_ids, target_ids)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        if (step + 1) % 100 == 0:
            val_input, val_target = get_batch(val_samples, 8, step)
            mx.eval(val_input, val_target)
            val_loss = loss_fn(model, val_input, val_target)
            logger.info(
                f"Step {step + 1}: train={float(loss.item()):.4f}, val={float(val_loss.item()):.4f}"
            )

    # Save
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from safetensors.numpy import save_file

    lora_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        arr = np.array(param.astype(mx.float32))
        lora_weights[name] = arr

    save_file(lora_weights, str(checkpoint_dir / "adapters.safetensors"))

    config = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "alpha": float(args.lora_rank * 2),
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
    }
    with open(checkpoint_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nSaved to {checkpoint_dir}")

    # Test
    logger.info("\nTesting...")
    test_cases = [
        "What is 12 times 9?",
        "Janet has 50 apples. She gives away 15. How many remain?",
        "What is 144 divided by 12?",
        "The sum of 25 and 17 is",
        "Each box holds 8 items. How many in 7 boxes?",
        "A tank has 200 gallons. 75 leak out. How much is left?",
        "Tickets cost 15 dollars each. Cost for 4 tickets?",
        "Add 11 and 94",
    ]

    for nl_input in test_cases:
        prompt = format_prompt(nl_input)
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
            if "=" in decoded and decoded.rstrip().endswith(" "):
                break
            if "</s>" in decoded or "\n" in decoded:
                break

        canonical = tokenizer.decode(generated_ids[0, prompt_len:].tolist()).strip()
        canonical = canonical.replace("</s>", "").strip()
        if "=" in canonical:
            eq_pos = canonical.find("=")
            canonical = canonical[: eq_pos + 1].strip() + " "

        logger.info(f"  {nl_input[:45]:45} → {canonical}")


if __name__ == "__main__":
    main()
