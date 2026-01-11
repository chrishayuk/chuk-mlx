#!/usr/bin/env python3
"""
Analyze hidden states to understand what L13 encodes.

Check if operation types are separable in hidden state space.
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_samples(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_hidden_states_all_layers(model, tokenizer, prompt: str) -> list[mx.array]:
    """Extract hidden states at all layers."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    backbone = model.model if hasattr(model, 'model') else model
    h = backbone.embed_tokens(input_ids)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    hidden_states = [h[0, -1, :]]  # After embedding

    for layer in backbone.layers:
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, 'hidden_states') else output
        hidden_states.append(h[0, -1, :])

    return hidden_states


def main():
    # Load model
    logger.info("Loading model...")
    from chuk_lazarus.models_v2.loader import load_model
    load_result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = load_result.model
    tokenizer = load_result.tokenizer
    model.freeze()

    num_layers = load_result.config.num_hidden_layers
    logger.info(f"Model has {num_layers} layers")

    # Load samples
    samples = load_samples("experiments/ir_emission/data/phase1_train.jsonl")
    op_to_idx = {"add": 0, "sub": 1, "mul": 2, "div": 3}
    samples = [s for s in samples if s.get("operation") in op_to_idx]

    # Collect hidden states by operation
    hidden_by_op = {op: [] for op in op_to_idx}
    n_per_op = 20  # Samples per operation

    for sample in samples:
        op = sample["operation"]
        if len(hidden_by_op[op]) >= n_per_op:
            continue

        states = get_hidden_states_all_layers(model, tokenizer, sample["prompt"])
        mx.eval(states)

        # Store as numpy for analysis (convert via tolist to handle bfloat16)
        hidden_by_op[op].append([np.array(s.astype(mx.float32).tolist()) for s in states])

        if all(len(v) >= n_per_op for v in hidden_by_op.values()):
            break

    logger.info(f"Collected {sum(len(v) for v in hidden_by_op.values())} samples")

    # Analyze separability at each layer
    logger.info("\nAnalyzing separability by layer...")
    logger.info("Layer | Between-class var | Within-class var | Ratio (higher = more separable)")
    logger.info("-" * 70)

    for layer_idx in range(num_layers + 1):
        # Collect all hidden states at this layer
        all_states = []
        all_labels = []

        for op, states_list in hidden_by_op.items():
            for sample_states in states_list:
                all_states.append(sample_states[layer_idx])
                all_labels.append(op_to_idx[op])

        X = np.stack(all_states)  # (n_samples, hidden_dim)
        y = np.array(all_labels)

        # Compute class means
        class_means = {}
        for op_idx in range(4):
            mask = y == op_idx
            if np.sum(mask) > 0:
                class_means[op_idx] = np.mean(X[mask], axis=0)

        global_mean = np.mean(X, axis=0)

        # Between-class variance
        between_var = 0
        for op_idx, mean in class_means.items():
            n = np.sum(y == op_idx)
            between_var += n * np.sum((mean - global_mean) ** 2)
        between_var /= len(y)

        # Within-class variance
        within_var = 0
        for op_idx, mean in class_means.items():
            mask = y == op_idx
            within_var += np.sum((X[mask] - mean) ** 2)
        within_var /= len(y)

        ratio = between_var / (within_var + 1e-10)

        layer_name = f"L{layer_idx:02d}" if layer_idx > 0 else "Emb"
        logger.info(f"{layer_name:>5} | {between_var:>17.2f} | {within_var:>16.2f} | {ratio:>8.4f}")

    # Show some example prompts
    logger.info("\nExample prompts by operation:")
    for op in ["add", "sub", "mul", "div"]:
        example = next(s for s in samples if s["operation"] == op)
        logger.info(f"  {op}: {example['prompt'][:50]}...")

    # Check cosine similarity between operation means at L12 (decision layer)
    logger.info("\nCosine similarity between class means at L12:")
    layer_idx = 12

    class_means = {}
    for op, states_list in hidden_by_op.items():
        states = [s[layer_idx] for s in states_list]
        class_means[op] = np.mean(states, axis=0)

    ops = list(op_to_idx.keys())
    logger.info(f"       {' '.join(f'{op:>6}' for op in ops)}")
    for i, op1 in enumerate(ops):
        row = []
        for op2 in ops:
            m1 = class_means[op1]
            m2 = class_means[op2]
            cos = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
            row.append(f"{cos:>6.3f}")
        logger.info(f"{op1:>6} {' '.join(row)}")


if __name__ == "__main__":
    main()
