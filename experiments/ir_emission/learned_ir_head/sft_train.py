"""
SFT Training for Word Problem → Expression Model.

Trains TinyLlama to output canonical math expressions from word problems.
Uses the verified training data from generate_sft_data.py.
"""

import sys
from pathlib import Path
import json
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from archive.wasm_runtime import WASMRuntime
from generate_sft_data import parse_expression, execute_ir


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_example(question: str, expression: str = None) -> str:
    """Format for training/inference."""
    if expression:
        return f"Q: {question}\nA: {expression}"
    return f"Q: {question}\nA:"


def train_sft(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_dir: str = None,
    num_epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 5e-6,
    max_seq_len: int = 128,
):
    """
    SFT training on verified (question, expression) pairs.
    """
    print("=" * 70)
    print("  SFT TRAINING: Word Problems → Expressions")
    print("=" * 70)

    if data_dir is None:
        data_dir = Path(__file__).parent / "sft_data"

    # Load data
    print("\nLoading datasets...")
    train_data = load_dataset(data_dir / "train.jsonl")
    val_data = load_dataset(data_dir / "val.jsonl")
    test_data = load_dataset(data_dir / "test.jsonl")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Load model
    print(f"\nLoading {model_name}...")
    result = load_model(model_name)
    model, tokenizer = result.model, result.tokenizer

    # Freeze all except last layer and lm_head
    print("\nFreezing model, training last layer + lm_head...")
    model.freeze()
    model.model.layers[-1].unfreeze()  # Last transformer layer
    model.lm_head.unfreeze()

    optimizer = optim.Adam(learning_rate=learning_rate)

    def compute_loss(model, batch_tokens, batch_masks):
        """Compute loss on answer tokens only."""
        logits = model(batch_tokens)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        targets = batch_tokens[:, 1:]
        masks = batch_masks[:, 1:]

        # Cross entropy
        vocab_size = logits.shape[-1]
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        )
        ce = ce.reshape(targets.shape)

        # Masked loss (only on answer tokens)
        loss = (ce * masks).sum() / (masks.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    print(f"\nTraining for {num_epochs} epochs (batch_size={batch_size})...")
    print("-" * 70)

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Tokenize batch
            all_tokens = []
            all_masks = []

            for item in batch:
                full_text = format_example(item["question"], item["expression"])
                q_text = format_example(item["question"])

                full_tokens = tokenizer.encode(full_text)[:max_seq_len]
                q_tokens = tokenizer.encode(q_text)

                # Mask: 0 for question, 1 for answer
                q_len = len(q_tokens)
                mask = [0] * q_len + [1] * (len(full_tokens) - q_len)

                all_tokens.append(full_tokens)
                all_masks.append(mask)

            # Pad to max length in batch
            max_len = max(len(t) for t in all_tokens)
            padded_tokens = []
            padded_masks = []

            for tokens, mask in zip(all_tokens, all_masks):
                pad_len = max_len - len(tokens)
                padded_tokens.append(tokens + [0] * pad_len)
                padded_masks.append(mask + [0] * pad_len)

            batch_tokens = mx.array(padded_tokens)
            batch_masks = mx.array(padded_masks, dtype=mx.float32)

            # Train step
            loss, grads = loss_and_grad(model, batch_tokens, batch_masks)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {num_batches}: loss={total_loss/num_batches:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # Quick validation
        val_acc = evaluate(model, tokenizer, val_data[:100])
        print(f"  Val accuracy (100 samples): {val_acc:.1%}")

    # Final test evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)

    test_acc = evaluate(model, tokenizer, test_data, verbose=True)
    print(f"\nTest accuracy: {test_acc:.1%}")

    return model, tokenizer


def generate_expression(model, tokenizer, question: str, max_new_tokens: int = 30) -> str:
    """Generate expression for a question."""
    prompt = format_example(question)
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_new_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        # Stop at newline or EOS
        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        # Stop at =
        decoded = tokenizer.decode(generated)
        if "=" in decoded:
            break

    result = tokenizer.decode(generated).strip()
    if "=" in result:
        result = result[:result.index("=") + 1].strip()
    return result


def evaluate(model, tokenizer, data: list[dict], verbose: bool = False) -> float:
    """Evaluate model on dataset."""
    runtime = WASMRuntime()
    correct = 0
    parse_success = 0

    for item in data:
        pred_expr = generate_expression(model, tokenizer, item["question"])

        # Check if parses
        ir = parse_expression(pred_expr)
        if ir is not None:
            parse_success += 1
            result = execute_ir(ir, runtime)
            if result == item["answer"]:
                correct += 1
                if verbose and correct <= 5:
                    print(f"✓ {item['question'][:50]}...")
                    print(f"    Pred: {pred_expr} → {result}")
            elif verbose and (len(data) - correct) <= 3:
                print(f"✗ {item['question'][:50]}...")
                print(f"    Pred: {pred_expr} → {result}, Expected: {item['answer']}")
        elif verbose and parse_success < 3:
            print(f"✗ PARSE FAIL: {item['question'][:50]}...")
            print(f"    Got: {pred_expr}")

    if verbose:
        print(f"\nParse success: {parse_success}/{len(data)} = {parse_success/len(data):.1%}")

    return correct / len(data)


def main():
    model, tokenizer = train_sft(
        num_epochs=2,
        batch_size=2,
        learning_rate=5e-6,
    )

    # Interactive test
    print("\n" + "=" * 70)
    print("  INTERACTIVE TEST")
    print("=" * 70)

    test_questions = [
        "Jenny has 5 apples. She gives 2 to Bob. How many left?",
        "A book costs $15. Tom buys 3 books. Total?",
        "Tom has 3 boxes with 12 pencils each. Gives 2 pencils to each of 5 friends. Left?",
        "144 students in teams of 6. Each team needs 4 balls. Total balls?",
    ]

    runtime = WASMRuntime()
    for q in test_questions:
        expr = generate_expression(model, tokenizer, q)
        ir = parse_expression(expr)
        result = execute_ir(ir, runtime) if ir else "PARSE_FAIL"
        print(f"\nQ: {q}")
        print(f"A: {expr} → {result}")


if __name__ == "__main__":
    main()
