#!/usr/bin/env python3
"""
Train on operator token hidden states.

Instead of using the last token, find the operator token (+, -, *, /)
and use its hidden state as the classification target.
"""

import json
import logging
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class OperationClassifier(nn.Module):
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


def find_operator_position(tokens: list[int], tokenizer) -> int:
    """Find the position of the operator token."""
    text = tokenizer.decode(tokens)

    # Decode each token and find operator
    for i, tok in enumerate(tokens):
        tok_text = tokenizer.decode([tok])
        if tok_text.strip() in ['+', '-', '*', 'x', '/', '×', '÷']:
            return i

    # Fallback: look for token containing operator
    for i, tok in enumerate(tokens):
        tok_text = tokenizer.decode([tok])
        if any(op in tok_text for op in ['+', '-', '*', '/', 'x']):
            return i

    # Last fallback: return second-to-last token
    return len(tokens) - 2


def get_hidden_state_at_position(
    model, tokenizer, prompt: str, decision_layer: int, position: int
) -> mx.array:
    """Extract hidden state at specific position and layer."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    backbone = model.model if hasattr(model, 'model') else model
    h = backbone.embed_tokens(input_ids)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(backbone.layers):
        if i > decision_layer:
            break
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, 'hidden_states') else output

    # Clamp position to valid range
    pos = min(position, h.shape[1] - 1)
    return h[0, pos, :]


def main():
    logger.info("Loading model...")
    from chuk_lazarus.models_v2.loader import load_model
    load_result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = load_result.model
    tokenizer = load_result.tokenizer
    model.freeze()

    num_layers = load_result.config.num_hidden_layers
    decision_layer = int(num_layers * 0.55)
    hidden_dim = load_result.config.hidden_size
    logger.info(f"Decision layer: {decision_layer}")

    # Load samples
    samples = load_samples("experiments/ir_emission/data/phase1_train.jsonl")
    op_to_idx = {"add": 0, "sub": 1, "mul": 2, "div": 3}
    samples = [s for s in samples if s.get("operation") in op_to_idx]
    logger.info(f"Loaded {len(samples)} samples")

    # Show some tokenizations
    logger.info("\nTokenization examples:")
    for op in ["add", "sub", "mul", "div"]:
        sample = next(s for s in samples if s["operation"] == op)
        tokens = tokenizer.encode(sample["prompt"])
        decoded = [tokenizer.decode([t]) for t in tokens]
        op_pos = find_operator_position(tokens, tokenizer)
        logger.info(f"  {op}: {sample['prompt'][:30]} -> op at pos {op_pos}")
        logger.info(f"       tokens: {decoded[:8]}...")

    classifier = OperationClassifier(hidden_dim, len(op_to_idx))
    optimizer = optim.Adam(learning_rate=1e-3)

    import random
    random.shuffle(samples)

    batch_size = 16
    for step in range(300):
        batch_idx = [(step * batch_size + i) % len(samples) for i in range(batch_size)]
        batch = [samples[i] for i in batch_idx]

        hidden_states = []
        labels = []
        for sample in batch:
            tokens = tokenizer.encode(sample["prompt"])
            op_pos = find_operator_position(tokens, tokenizer)
            h = get_hidden_state_at_position(
                model, tokenizer, sample["prompt"], decision_layer, op_pos
            )
            hidden_states.append(h)
            labels.append(op_to_idx[sample["operation"]])

        hidden_states = mx.stack(hidden_states)
        labels = mx.array(labels)
        mx.eval(hidden_states)

        def loss_fn(classifier_params):
            classifier.update(classifier_params)
            logits = classifier(hidden_states)
            return nn.losses.cross_entropy(logits, labels, reduction="mean")

        loss, grads = nn.value_and_grad(classifier, loss_fn)(classifier.parameters())
        optimizer.update(classifier, grads)
        mx.eval(classifier.parameters())

        logits = classifier(hidden_states)
        preds = mx.argmax(logits, axis=-1)
        acc = float(mx.mean(preds == labels).item())

        if (step + 1) % 20 == 0:
            logger.info(f"Step {step + 1}: loss={float(loss.item()):.4f}, acc={acc:.2%}")

    # Final evaluation
    logger.info("\nFinal evaluation...")
    correct = 0
    total = 0
    confusion = [[0] * 4 for _ in range(4)]

    for sample in samples[:200]:
        tokens = tokenizer.encode(sample["prompt"])
        op_pos = find_operator_position(tokens, tokenizer)
        h = get_hidden_state_at_position(
            model, tokenizer, sample["prompt"], decision_layer, op_pos
        )
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

    ops = ["add", "sub", "mul", "div"]
    logger.info("\nConfusion matrix:")
    logger.info(f"        {' '.join(f'{op:>6}' for op in ops)}")
    for i, op in enumerate(ops):
        row = ' '.join(f'{confusion[i][j]:>6}' for j in range(4))
        logger.info(f"{op:>6}: {row}")


if __name__ == "__main__":
    main()
