"""
Quick sanity check for IR head concept.

Tests in stages:
1. Operation classification only (should work - we proved this)
2. Single operand extraction
3. Full IR extraction
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


def get_hidden_states(model, input_ids, layer_idx=12):
    """Extract normalized hidden states from layer."""
    backbone = model.model
    h = backbone.embed_tokens(input_ids)

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(backbone.layers):
        if i >= layer_idx:
            break
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, "hidden_states") else output

    # Normalize!
    h = backbone.norm(h)
    return h


def test_operation_classification():
    """Test 1: Can we classify operations from L12?"""
    print("\n" + "=" * 50)
    print("TEST 1: Operation Classification")
    print("=" * 50)

    print("Loading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    # Simple training data
    train_data = [
        ("5 + 3", 0),  # add
        ("10 + 20", 0),
        ("7 - 2", 1),  # subtract
        ("100 - 50", 1),
        ("4 * 6", 2),  # multiply
        ("8 * 9", 2),
        ("20 / 4", 3),  # divide
        ("100 / 10", 3),
    ]

    # Simple classifier
    hidden_dim = 2048
    classifier = nn.Linear(hidden_dim, 4)
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(classifier, h, target):
        logits = classifier(h)
        return nn.losses.cross_entropy(logits, target, reduction="mean")

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    print("Training operation classifier (100 steps)...")
    for step in range(100):
        for text, label in train_data:
            tokens = mx.array([tokenizer.encode(text)])
            h = get_hidden_states(model, tokens)[:, -1, :]  # Last token
            mx.eval(h)

            target = mx.array([label])
            loss, grads = loss_and_grad(classifier, h, target)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters())

    # Test
    print("\nTesting:")
    test_data = [
        ("15 + 25", 0, "add"),
        ("30 - 12", 1, "subtract"),
        ("7 * 8", 2, "multiply"),
        ("48 / 6", 3, "divide"),
    ]

    correct = 0
    for text, label, name in test_data:
        tokens = mx.array([tokenizer.encode(text)])
        h = get_hidden_states(model, tokens)[:, -1, :]
        logits = classifier(h)
        pred = int(mx.argmax(logits).item())
        status = "✓" if pred == label else "✗"
        if pred == label:
            correct += 1
        print(f"  {status} '{text}' → pred={pred}, expected={label} ({name})")

    print(f"\nOperation accuracy: {correct}/{len(test_data)} = {correct/len(test_data):.0%}")
    return correct == len(test_data)


def test_number_extraction():
    """Test 2: Can we extract numbers from hidden states?"""
    print("\n" + "=" * 50)
    print("TEST 2: Number Extraction (first operand only)")
    print("=" * 50)

    print("Loading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    # Training data: simple "X + Y" format, predict X
    train_data = [
        ("5 + 3", 5),
        ("10 + 7", 10),
        ("25 + 12", 25),
        ("3 + 9", 3),
        ("50 + 20", 50),
        ("8 + 4", 8),
        ("15 + 6", 15),
        ("42 + 11", 42),
    ]

    # Regression head for first number
    hidden_dim = 2048
    head = nn.Sequential(
        nn.Linear(hidden_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    optimizer = optim.Adam(learning_rate=0.001)

    def loss_fn(head, h, target):
        pred = head(h).squeeze()
        return mx.mean((pred - target) ** 2)

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    print("Training number extractor (200 steps)...")
    for step in range(200):
        total_loss = 0
        for text, num in train_data:
            tokens = mx.array([tokenizer.encode(text)])
            h = get_hidden_states(model, tokens)[:, -1, :]
            mx.eval(h)

            target = mx.array([float(num)])
            loss, grads = loss_and_grad(head, h, target)
            optimizer.update(head, grads)
            mx.eval(head.parameters())
            total_loss += float(loss.item())

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: loss={total_loss/len(train_data):.4f}")

    # Test
    print("\nTesting:")
    test_data = [
        ("20 + 5", 20),
        ("7 + 3", 7),
        ("35 + 10", 35),
        ("12 + 8", 12),
    ]

    total_error = 0
    for text, expected in test_data:
        tokens = mx.array([tokenizer.encode(text)])
        h = get_hidden_states(model, tokens)[:, -1, :]
        pred = float(head(h).squeeze().item())
        error = abs(pred - expected)
        total_error += error
        print(f"  '{text}' → pred={pred:.1f}, expected={expected}, error={error:.1f}")

    avg_error = total_error / len(test_data)
    print(f"\nAverage error: {avg_error:.1f}")
    return avg_error < 10  # Success if average error < 10


if __name__ == "__main__":
    print("=" * 50)
    print("  QUICK IR HEAD SANITY CHECK")
    print("=" * 50)

    op_ok = test_operation_classification()
    num_ok = test_number_extraction()

    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Operation classification: {'✓ PASS' if op_ok else '✗ FAIL'}")
    print(f"  Number extraction: {'✓ PASS' if num_ok else '✗ FAIL'}")

    if op_ok and num_ok:
        print("\n  → Both tests pass! Full IR head should work.")
    elif op_ok:
        print("\n  → Operations work, but numbers need different approach.")
    else:
        print("\n  → Fundamental issues with hidden state extraction.")
