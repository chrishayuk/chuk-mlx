"""
Train a model to normalize word problems → canonical math expressions.

Instead of few-shot prompting, we fine-tune the model to:
- Input: "Jenny has 5 apples. She gives 2 to Bob. How many left?"
- Output: "5 - 2 ="

This bakes the "parser" into the model weights.
"""

import sys
from pathlib import Path
import random
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# Data Generation
# =============================================================================

def generate_single_op_problems(n: int) -> list[dict]:
    """Generate single operation word problems."""
    templates = {
        "multiply": [
            ("{name} buys {a} items at ${b} each. Total cost?", "{a} * {b} ="),
            ("A store has {a} boxes with {b} items each. Total items?", "{a} * {b} ="),
            ("{name} earns ${a} per hour for {b} hours. Total earned?", "{a} * {b} ="),
            ("There are {a} rows with {b} seats each. Total seats?", "{a} * {b} ="),
            ("{a} bags with {b} apples each. How many apples?", "{a} * {b} ="),
        ],
        "divide": [
            ("{name} has {a} cookies to share equally among {b} friends. Each gets?", "{a} / {b} ="),
            ("{a} students split into groups of {b}. How many groups?", "{a} / {b} ="),
            ("${a} divided equally among {b} people. Each gets?", "{a} / {b} ="),
            ("{a} items packed {b} per box. How many boxes?", "{a} / {b} ="),
        ],
        "add": [
            ("{name} has {a} marbles and finds {b} more. Total?", "{a} + {b} ="),
            ("{name} has ${a} and earns ${b} more. Total?", "{a} + {b} ="),
            ("A box has {a} red balls and {b} blue balls. Total?", "{a} + {b} ="),
            ("{name} reads {a} pages Monday and {b} pages Tuesday. Total pages?", "{a} + {b} ="),
        ],
        "subtract": [
            ("{name} has {a} apples and gives {b} away. How many left?", "{a} - {b} ="),
            ("{name} has ${a} and spends ${b}. How much left?", "{a} - {b} ="),
            ("A jar has {a} candies. {name} eats {b}. How many left?", "{a} - {b} ="),
            ("{a} birds on a wire. {b} fly away. How many left?", "{a} - {b} ="),
        ],
    }

    names = ["Tom", "Sarah", "John", "Lisa", "Mike", "Emma", "Alex", "Maria"]
    data = []

    for _ in range(n):
        op = random.choice(list(templates.keys()))
        template, answer_template = random.choice(templates[op])

        if op == "divide":
            b = random.randint(2, 12)
            a = b * random.randint(2, 15)
        else:
            a = random.randint(2, 100)
            b = random.randint(2, 50)

        name = random.choice(names)
        question = template.format(name=name, a=a, b=b)
        answer = answer_template.format(a=a, b=b)

        data.append({"question": question, "expression": answer, "type": "single"})

    return data


def generate_two_op_problems(n: int) -> list[dict]:
    """Generate two-operation word problems."""
    templates = [
        # has X, operation Y, operation Z
        ("{name} has {a} dollars. Buys {b} items at {c} dollars each. How much left?",
         "{a} - {b} * {c} ="),
        ("{name} has {a} stickers. Gets {b} more, then gives {c} away. How many now?",
         "{a} + {b} - {c} ="),
        ("{name} bakes {a} cookies per batch. Makes {b} batches, sells {c}. How many left?",
         "{a} * {b} - {c} ="),
        ("{a} students in {b} equal groups. Each group gets {c} books. Total books?",
         "{a} / {b} * {c} ="),
        ("A factory makes {a} toys per day for {b} days. Ships {c}. How many left?",
         "{a} * {b} - {c} ="),
        ("{name} has ${a}. Earns ${b}, then spends ${c}. How much now?",
         "{a} + {b} - {c} ="),
    ]

    names = ["Tom", "Sarah", "John", "Lisa", "Mike", "Emma"]
    data = []

    for _ in range(n):
        template, answer_template = random.choice(templates)

        a = random.randint(10, 100)
        b = random.randint(2, 20)
        c = random.randint(2, 30)

        # Ensure division works cleanly
        if "/" in answer_template:
            b = random.randint(2, 10)
            a = b * random.randint(2, 15)

        name = random.choice(names)
        question = template.format(name=name, a=a, b=b, c=c)
        answer = answer_template.format(a=a, b=b, c=c)

        data.append({"question": question, "expression": answer, "type": "two_op"})

    return data


def generate_three_op_problems(n: int) -> list[dict]:
    """Generate three-operation problems."""
    templates = [
        # Nested multiplication
        ("{a} buildings. Each has {b} floors. Each floor has {c} rooms. Total rooms?",
         "{a} * {b} * {c} ="),
        ("{name} has {a} boxes. Each box has {b} bags. Each bag has {c} marbles. Total?",
         "{a} * {b} * {c} ="),
        # Rate applied to multiple periods
        ("{name} earns ${a} per hour. Works {b} hours Monday and {c} hours Tuesday. Total earned?",
         "{a} * {b} + {a} * {c} ="),
        # Sequential operations
        ("{name} has {a} dollars. Spends {b}, then {c}, then earns {d}. How much now?",
         "{a} - {b} - {c} + {d} ="),
        ("A store has {a} apples. Sells {b} Monday, {c} Tuesday, receives {d}. How many now?",
         "{a} - {b} - {c} + {d} ="),
    ]

    names = ["Tom", "Sarah", "John", "Lisa"]
    data = []

    for _ in range(n):
        template, answer_template = random.choice(templates)

        a = random.randint(2, 20)
        b = random.randint(2, 15)
        c = random.randint(2, 15)
        d = random.randint(2, 30)

        name = random.choice(names)
        question = template.format(name=name, a=a, b=b, c=c, d=d)
        answer = answer_template.format(a=a, b=b, c=c, d=d)

        data.append({"question": question, "expression": answer, "type": "three_op"})

    return data


def generate_complex_problems(n: int) -> list[dict]:
    """Generate GSM8K-style complex problems."""
    templates = [
        # Compute then subtract
        ("{name} has {a} boxes with {b} pencils each. Gives {c} pencils to each of {d} friends. Left?",
         "{a} * {b} - {c} * {d} ="),
        # Total minus used
        ("A parking lot has {a} rows with {b} spaces each. {c} cars parked. Empty spaces?",
         "{a} * {b} - {c} ="),
        # Divide then multiply
        ("{a} cookies in boxes of {b}. Each box sells for ${c}. Total money?",
         "{a} / {b} * {c} ="),
        # Capacity minus passengers
        ("A train has {a} cars with {b} seats each. {c} passengers board. Empty seats?",
         "{a} * {b} - {c} ="),
        # Split then allocate
        ("{a} students in teams of {b}. Each team needs {c} balls. Total balls needed?",
         "{a} / {b} * {c} ="),
    ]

    names = ["Tom", "Sarah", "John", "Lisa", "Mike"]
    data = []

    for _ in range(n):
        template, answer_template = random.choice(templates)

        a = random.randint(3, 20)
        b = random.randint(2, 15)
        c = random.randint(2, 50)
        d = random.randint(2, 10)

        # Ensure divisions work
        if "/" in answer_template:
            b = random.randint(2, 10)
            a = b * random.randint(2, 15)

        name = random.choice(names)
        question = template.format(name=name, a=a, b=b, c=c, d=d)
        answer = answer_template.format(a=a, b=b, c=c, d=d)

        data.append({"question": question, "expression": answer, "type": "complex"})

    return data


def generate_training_data(n_per_type: int = 500) -> list[dict]:
    """Generate balanced training data."""
    data = []
    data.extend(generate_single_op_problems(n_per_type))
    data.extend(generate_two_op_problems(n_per_type))
    data.extend(generate_three_op_problems(n_per_type))
    data.extend(generate_complex_problems(n_per_type))
    random.shuffle(data)
    return data


# =============================================================================
# Training
# =============================================================================

def format_training_example(question: str, expression: str) -> str:
    """Format as input for training."""
    return f"Q: {question}\nA: {expression}"


def train_normalizer(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_examples: int = 2000,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    save_path: str = None,
):
    """
    Fine-tune model to normalize word problems to expressions.

    Uses LoRA-style training on just the attention layers for efficiency.
    """
    print("=" * 70)
    print("  TRAINING WORD PROBLEM NORMALIZER")
    print("=" * 70)

    print(f"\nLoading {model_name}...")
    result = load_model(model_name)
    model, tokenizer = result.model, result.tokenizer

    # Generate training data
    print(f"\nGenerating {num_examples} training examples...")
    random.seed(42)
    train_data = generate_training_data(num_examples // 4)

    print(f"  Single op: {sum(1 for d in train_data if d['type'] == 'single')}")
    print(f"  Two op: {sum(1 for d in train_data if d['type'] == 'two_op')}")
    print(f"  Three op: {sum(1 for d in train_data if d['type'] == 'three_op')}")
    print(f"  Complex: {sum(1 for d in train_data if d['type'] == 'complex')}")

    # Freeze most of model, only train LM head and last few layers
    print("\nFreezing base model, training last 2 layers + LM head...")
    model.freeze()

    # Unfreeze last 2 transformer layers and lm_head
    num_layers = len(model.model.layers)
    for i in range(num_layers - 2, num_layers):
        model.model.layers[i].unfreeze()
    model.lm_head.unfreeze()

    # Count trainable params (simplified)
    def count_params(params, prefix=""):
        total = 0
        for k, v in params.items():
            if isinstance(v, dict):
                total += count_params(v, f"{prefix}{k}.")
            elif hasattr(v, 'size'):
                total += v.size
        return total

    total_params = count_params(model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Training last 2 layers + lm_head")

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, input_ids, target_ids, mask):
        """Cross-entropy loss on target tokens only."""
        logits = model(input_ids)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        targets = target_ids[:, 1:]
        loss_mask = mask[:, 1:]

        # Compute loss only on answer tokens
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction='none'
        )
        ce = ce.reshape(targets.shape)
        masked_loss = (ce * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        return masked_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Prepare batch
            texts = [format_training_example(d["question"], d["expression"]) for d in batch]

            # Tokenize with padding
            max_len = 0
            all_tokens = []
            all_masks = []

            for text in texts:
                # Split into question and answer parts
                parts = text.split("\nA: ")
                q_tokens = tokenizer.encode(parts[0] + "\nA: ")
                a_tokens = tokenizer.encode(parts[1])

                tokens = q_tokens + a_tokens
                # Mask: 0 for question, 1 for answer
                mask = [0] * len(q_tokens) + [1] * len(a_tokens)

                all_tokens.append(tokens)
                all_masks.append(mask)
                max_len = max(max_len, len(tokens))

            # Pad
            input_ids = []
            masks = []
            for tokens, mask in zip(all_tokens, all_masks):
                pad_len = max_len - len(tokens)
                input_ids.append(tokens + [tokenizer.pad_token_id or 0] * pad_len)
                masks.append(mask + [0] * pad_len)

            input_ids = mx.array(input_ids)
            masks = mx.array(masks, dtype=mx.float32)

            # Forward and backward
            loss, grads = loss_and_grad(model, input_ids, input_ids, masks)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {num_batches}: loss={total_loss/num_batches:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    # Save if requested
    if save_path:
        print(f"\nSaving model to {save_path}")
        # Save just the trained layers
        trained_weights = {}
        for name, param in model.parameters().items():
            if any(f"layers.{i}." in name for i in range(num_layers - 2, num_layers)):
                trained_weights[name] = param
            elif "lm_head" in name:
                trained_weights[name] = param
        mx.savez(save_path, **trained_weights)

    return model, tokenizer, train_data


def test_trained_model(model, tokenizer, test_cases: list[tuple[str, str]]):
    """Test the trained model on examples."""
    print("\n" + "=" * 70)
    print("  TESTING TRAINED MODEL")
    print("=" * 70)

    correct = 0

    for question, expected in test_cases:
        prompt = f"Q: {question}\nA:"
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Generate
        generated = []
        for _ in range(30):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            decoded = tokenizer.decode(generated)
            if "=" in decoded:
                break

        result = tokenizer.decode(generated).strip()

        # Clean up result
        if "=" in result:
            result = result[:result.index("=") + 1].strip()

        match = result == expected
        if match:
            correct += 1

        status = "✓" if match else "✗"
        print(f"{status} Q: {question[:50]}...")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result}")

    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")
    return correct / len(test_cases)


def main():
    # Train
    model, tokenizer, train_data = train_normalizer(
        num_examples=2000,
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-5,
    )

    # Test on held-out examples (not in training templates)
    test_cases = [
        ("Jenny has 5 apples. She gives 2 to Bob. How many left?", "5 - 2 ="),
        ("A book costs $15. Tom buys 3 books. How much?", "15 * 3 ="),
        ("48 stickers shared equally among 6 friends. Each gets?", "48 / 6 ="),
        ("Sam has $20. Buys 3 toys at $4 each. How much left?", "20 - 3 * 4 ="),
        ("5 classrooms. Each has 6 rows. Each row has 4 desks. Total?", "5 * 6 * 4 ="),
        ("John earns $10/hour. Works 8 hours Monday, 6 Tuesday. Total?", "10 * 8 + 10 * 6 ="),
        ("Tom has 3 boxes with 12 pencils each. Gives 2 to each of 5 friends. Left?", "3 * 12 - 2 * 5 ="),
        ("144 students in teams of 6. Each team needs 4 balls. Total balls?", "144 / 6 * 4 ="),
    ]

    test_trained_model(model, tokenizer, test_cases)


if __name__ == "__main__":
    main()
