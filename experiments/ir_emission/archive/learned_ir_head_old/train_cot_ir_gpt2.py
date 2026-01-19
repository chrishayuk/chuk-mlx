"""
Train CoT→IR using TinyLlama (pretrained).

The pretrained model already understands language - we just need
to fine-tune it to output IR after reasoning.
"""

import re
import json
import random
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.lora import LoRALinear
import numpy as np


# =============================================================================
# IR EXECUTION
# =============================================================================

def execute_ir(ir_code: str) -> float:
    """Execute IR and return final value."""
    env = {}
    steps = ir_code.strip().split('|')

    for step in steps:
        step = step.strip()
        if step == '[END]' or not step:
            continue

        if '=' not in step:
            continue

        var, expr = step.split('=', 1)
        var = var.strip()
        expr = expr.strip()

        # Substitute previous steps
        for prev_var, prev_val in env.items():
            expr = expr.replace(prev_var, str(prev_val))

        try:
            result = eval(expr)
            env[var] = result
        except:
            return None

    if not env:
        return None
    return list(env.values())[-1]


def extract_ir_from_output(text: str) -> str:
    """Extract IR code from model output."""
    # Try [IR] ... [ANSWER] format
    match = re.search(r'\[IR\]\s*(.+?)\s*\[ANSWER\]', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try [IR] ... [END] format
    match = re.search(r'\[IR\]\s*(.+?\[END\])', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try just after [IR]
    match = re.search(r'\[IR\]\s*(.+?)(?:\n\n|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(n_train: int = 200, n_test: int = 50):
    """Load and format data for training."""
    train_path = Path(__file__).parent / "cot_ir_data" / "train.json"
    test_path = Path(__file__).parent / "cot_ir_data" / "test.json"

    with open(train_path) as f:
        train_data = json.load(f)

    with open(test_path) as f:
        test_data = json.load(f)

    random.shuffle(train_data)
    random.shuffle(test_data)

    # Format for training
    def format_example(ex):
        # Shorter format - just question and IR output
        return {
            'text': f"Q: {ex['input']}\nA: {ex['cot']} [IR] {ex['ir']} [ANSWER] {ex['answer']}<|endoftext|>",
            'answer': ex['answer'],
            'ir': ex['ir'],
        }

    train_formatted = [format_example(ex) for ex in train_data[:n_train]]
    test_formatted = [format_example(ex) for ex in test_data[:n_test]]

    return train_formatted, test_formatted


def save_training_data(data: list, path: Path):
    """Save data in JSONL format for mlx-lm trainer."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps({'text': item['text']}) + '\n')


# =============================================================================
# SIMPLE FINE-TUNING (without mlx-lm trainer)
# =============================================================================

def simple_finetune(model, tokenizer, train_data: list, n_epochs: int = 5,
                    batch_size: int = 4, lr: float = 1e-4):
    """Simple fine-tuning loop."""

    optimizer = optim.AdamW(learning_rate=lr)

    # Tokenize all training data
    print("Tokenizing training data...")
    tokenized = []
    for item in train_data:
        tokens = tokenizer.encode(item['text'])
        if len(tokens) <= 512:  # Skip too long
            tokenized.append(mx.array(tokens))

    print(f"Training on {len(tokenized)} examples")

    def compute_loss(model, tokens):
        """Compute next-token prediction loss."""
        inputs = tokens[:-1]
        targets = tokens[1:]

        # Get logits
        logits = model(inputs[None, :])[0]

        # Cross-entropy
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        losses = -log_probs[mx.arange(len(targets)), targets]

        return losses.mean()

    # Training loop
    for epoch in range(n_epochs):
        random.shuffle(tokenized)
        total_loss = 0
        n_batches = 0

        for tokens in tokenized:
            loss, grads = nn.value_and_grad(model, compute_loss)(model, tokens)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += float(loss)
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, tokenizer, test_data: list, n_samples: int = 20):
    """Evaluate model on test examples."""

    correct = 0
    valid_ir = 0
    total = 0
    results = []

    for item in test_data[:n_samples]:
        # Get prompt (question only)
        prompt = f"Q: {item['text'].split('Q: ')[1].split('A: ')[0].strip()}\nA:"

        # Generate
        tokens = tokenizer.encode(prompt)
        prompt_arr = mx.array(tokens)[None, :]

        # Simple greedy generation
        generated = list(tokens)
        for _ in range(300):
            logits = model(mx.array([generated]))[0, -1, :]
            next_token = int(mx.argmax(logits))
            generated.append(next_token)

            # Stop at end token
            if next_token == tokenizer.eos_token_id:
                break

        output = tokenizer.decode(generated)

        # Extract IR
        ir = extract_ir_from_output(output)
        expected_answer = item['answer']

        ir_result = None
        is_correct = False

        if ir:
            valid_ir += 1
            ir_result = execute_ir(ir)
            if ir_result is not None:
                is_correct = abs(ir_result - expected_answer) < 0.01
                if is_correct:
                    correct += 1

        results.append({
            'question': item['text'][:80],
            'expected': expected_answer,
            'generated_ir': ir,
            'ir_result': ir_result,
            'correct': is_correct,
        })

        total += 1

    return {
        'total': total,
        'valid_ir': valid_ir,
        'correct': correct,
        'valid_ir_rate': valid_ir / total if total > 0 else 0,
        'accuracy': correct / total if total > 0 else 0,
        'results': results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    mx.random.seed(42)

    print("=" * 70)
    print("  COT→IR TRAINING WITH GPT-2")
    print("  Fine-tuning pretrained model on GSM8K CoT→IR")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_data, test_data = load_and_prepare_data(n_train=200, n_test=30)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Show example
    print("\nExample training format:")
    print(train_data[0]['text'][:200] + "...")

    # Load TinyLlama
    print("\nLoading TinyLlama...")
    model, tokenizer = load("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")

    # Quick baseline eval
    print("\n" + "=" * 70)
    print("BASELINE (before fine-tuning)")
    print("=" * 70)

    baseline = evaluate_model(model, tokenizer, test_data, n_samples=10)
    print(f"Valid IR: {baseline['valid_ir_rate']:.1%}")
    print(f"Accuracy: {baseline['accuracy']:.1%}")

    # Fine-tune
    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print("=" * 70)

    model = simple_finetune(model, tokenizer, train_data,
                            n_epochs=10, batch_size=1, lr=5e-5)

    # Final eval
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final = evaluate_model(model, tokenizer, test_data, n_samples=len(test_data))

    print(f"\nTotal: {final['total']}")
    print(f"Valid IR: {final['valid_ir']} ({final['valid_ir_rate']:.1%})")
    print(f"Correct: {final['correct']} ({final['accuracy']:.1%})")

    # Show results
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    print("\nCorrect predictions:")
    for r in [x for x in final['results'] if x['correct']][:5]:
        print(f"\n  Q: {r['question'][:60]}...")
        print(f"  IR: {r['generated_ir']}")
        print(f"  Result: {r['ir_result']} (expected: {r['expected']})")

    print("\nIncorrect predictions:")
    for r in [x for x in final['results'] if not x['correct']][:5]:
        print(f"\n  Q: {r['question'][:60]}...")
        print(f"  IR: {r['generated_ir']}")
        print(f"  Result: {r['ir_result']} (expected: {r['expected']})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Baseline:
    Valid IR: {baseline['valid_ir_rate']:.1%}
    Accuracy: {baseline['accuracy']:.1%}

  After fine-tuning:
    Valid IR: {final['valid_ir_rate']:.1%}
    Accuracy: {final['accuracy']:.1%}

  Improvement: {final['accuracy'] - baseline['accuracy']:.1%}
""")


if __name__ == "__main__":
    main()
