"""
Quick test v2: More training data, verify concept works.
"""

import sys
from pathlib import Path
import random

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

    h = backbone.norm(h)
    return h


def main():
    print("Loading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()
    hidden_dim = 2048

    # Generate more training data
    random.seed(42)
    train_data = []
    ops = [("+", 0), ("-", 1), ("*", 2), ("/", 3)]

    for _ in range(100):  # 100 examples
        op_sym, op_idx = random.choice(ops)
        a = random.randint(1, 50)
        b = random.randint(1, 20)
        if op_sym == "/":
            a = b * random.randint(1, 10)  # Clean division
        train_data.append((f"{a} {op_sym} {b}", op_idx, a, b))

    # Simple IR head: operation + first operand
    class SimpleIRHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
            )
            self.op_head = nn.Linear(512, 4)
            self.num_head = nn.Linear(512, 1)

        def __call__(self, h):
            shared = self.shared(h)
            return self.op_head(shared), self.num_head(shared).squeeze()

    head = SimpleIRHead()
    optimizer = optim.Adam(learning_rate=0.005)

    def loss_fn(head, h, op_target, num_target):
        op_logits, num_pred = head(h)
        op_loss = nn.losses.cross_entropy(op_logits, op_target, reduction="mean")
        num_loss = mx.mean((num_pred - num_target) ** 2) / 1000  # Scale down
        return op_loss + num_loss

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    print(f"Training on {len(train_data)} examples...")
    print("-" * 40)

    for epoch in range(10):
        random.shuffle(train_data)
        total_loss = 0

        for text, op_idx, a, b in train_data:
            tokens = mx.array([tokenizer.encode(text)])
            h = get_hidden_states(model, tokens)[:, -1, :]
            mx.eval(h)

            loss, grads = loss_and_grad(
                head, h,
                mx.array([op_idx]),
                mx.array([float(a)])
            )
            optimizer.update(head, grads)
            mx.eval(head.parameters())
            total_loss += float(loss.item())

        print(f"Epoch {epoch + 1}: loss={total_loss/len(train_data):.4f}")

    # Test on held-out examples
    print("\n" + "=" * 40)
    print("Testing on new examples:")
    print("=" * 40)

    test_data = [
        ("25 + 13", 0, 25, "add"),
        ("47 - 19", 1, 47, "subtract"),
        ("6 * 9", 2, 6, "multiply"),
        ("72 / 8", 3, 72, "divide"),
        ("33 + 7", 0, 33, "add"),
        ("18 - 5", 1, 18, "subtract"),
    ]

    op_correct = 0
    num_errors = []

    for text, expected_op, expected_num, op_name in test_data:
        tokens = mx.array([tokenizer.encode(text)])
        h = get_hidden_states(model, tokens)[:, -1, :]
        op_logits, num_pred = head(h)

        pred_op = int(mx.argmax(op_logits).item())
        pred_num = float(num_pred.item())

        op_ok = pred_op == expected_op
        if op_ok:
            op_correct += 1
        num_err = abs(pred_num - expected_num)
        num_errors.append(num_err)

        status = "✓" if op_ok else "✗"
        print(f"{status} '{text}' → op={pred_op}({op_name}), num={pred_num:.1f} (expected {expected_num})")

    print(f"\nOperation accuracy: {op_correct}/{len(test_data)} = {op_correct/len(test_data):.0%}")
    print(f"Avg number error: {sum(num_errors)/len(num_errors):.1f}")

    if op_correct >= 5 and sum(num_errors)/len(num_errors) < 10:
        print("\n✓ CONCEPT VALIDATED: L12 encodes both operation and operands!")
    else:
        print("\n⚠ Needs more training or different approach")


if __name__ == "__main__":
    main()
