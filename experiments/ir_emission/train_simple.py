#!/usr/bin/env python3
"""
Simplified IR Emission Training

A minimal version to debug the learning signal.
Just predicts the operation type from L13 hidden state.
"""

import argparse
import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Simple operation classifier
class OperationClassifier(nn.Module):
    """Predict operation type from hidden state."""

    def __init__(self, hidden_dim: int, num_ops: int = 4):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 256)
        self.out = nn.Linear(256, num_ops)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.proj(x))
        return self.out(x)


def load_samples(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_hidden_state(model, tokenizer, prompt: str, decision_layer: int) -> mx.array:
    """Extract hidden state at decision layer."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Access backbone
    backbone = model.model if hasattr(model, 'model') else model

    # Forward through embedding
    h = backbone.embed_tokens(input_ids)

    # Forward through layers
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(backbone.layers):
        if i > decision_layer:
            break
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, 'hidden_states') else output

    return h[0, -1, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", "-d", default="experiments/ir_emission/data/phase1_train.jsonl")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    from chuk_lazarus.models_v2.loader import load_model
    load_result = load_model(args.model)
    model = load_result.model
    tokenizer = load_result.tokenizer
    model.freeze()

    # Decision layer
    num_layers = load_result.config.num_hidden_layers
    decision_layer = int(num_layers * 0.55)
    hidden_dim = load_result.config.hidden_size
    logger.info(f"Decision layer: {decision_layer}, hidden_dim: {hidden_dim}")

    # Load data
    samples = load_samples(args.data)
    logger.info(f"Loaded {len(samples)} samples")

    # Map operations to indices
    op_to_idx = {"add": 0, "sub": 1, "mul": 2, "div": 3}

    # Filter to samples with known operations
    samples = [s for s in samples if s.get("operation") in op_to_idx]
    logger.info(f"Filtered to {len(samples)} samples with operations")

    # Create classifier
    classifier = OperationClassifier(hidden_dim, len(op_to_idx))
    optimizer = optim.Adam(learning_rate=args.lr)

    # Training loop
    import random
    random.shuffle(samples)

    for step in range(args.steps):
        # Sample batch
        batch_idx = [(step * args.batch_size + i) % len(samples) for i in range(args.batch_size)]
        batch = [samples[i] for i in batch_idx]

        # Get hidden states
        hidden_states = []
        labels = []
        for sample in batch:
            h = get_hidden_state(model, tokenizer, sample["prompt"], decision_layer)
            hidden_states.append(h)
            labels.append(op_to_idx[sample["operation"]])

        hidden_states = mx.stack(hidden_states)
        labels = mx.array(labels)
        mx.eval(hidden_states)  # Detach from base model

        # Forward + loss
        def loss_fn(classifier_params):
            classifier.update(classifier_params)
            logits = classifier(hidden_states)
            return nn.losses.cross_entropy(logits, labels, reduction="mean")

        loss, grads = nn.value_and_grad(classifier, loss_fn)(classifier.parameters())
        optimizer.update(classifier, grads)
        mx.eval(classifier.parameters())

        # Compute accuracy
        logits = classifier(hidden_states)
        preds = mx.argmax(logits, axis=-1)
        acc = float(mx.mean(preds == labels).item())

        if (step + 1) % 10 == 0:
            logger.info(f"Step {step + 1}: loss={float(loss.item()):.4f}, acc={acc:.2%}")

    # Final evaluation on all data
    logger.info("\nFinal evaluation...")
    correct = 0
    total = 0
    confusion = [[0] * 4 for _ in range(4)]

    for sample in samples[:200]:
        h = get_hidden_state(model, tokenizer, sample["prompt"], decision_layer)
        h = h[None, :]
        mx.eval(h)

        logits = classifier(h)
        pred = int(mx.argmax(logits, axis=-1).item())
        true = op_to_idx[sample["operation"]]

        confusion[true][pred] += 1
        if pred == true:
            correct += 1
        total += 1

    logger.info(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

    # Print confusion matrix
    ops = ["add", "sub", "mul", "div"]
    logger.info("\nConfusion matrix:")
    logger.info(f"        {' '.join(f'{op:>6}' for op in ops)}")
    for i, op in enumerate(ops):
        row = ' '.join(f'{confusion[i][j]:>6}' for j in range(4))
        logger.info(f"{op:>6}: {row}")


if __name__ == "__main__":
    main()
