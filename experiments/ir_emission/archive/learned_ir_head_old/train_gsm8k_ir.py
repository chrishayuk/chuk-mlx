"""
Train model to emit IR format for GSM8K problems.

Uses RL with verifiable rewards - execution IS the reward signal.
"""

import sys
from pathlib import Path
import random
import json
import re

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# IR EXECUTION (Verifiable Reward)
# =============================================================================

def execute_ir(code: str) -> tuple[float | None, str, dict]:
    """
    Execute IR code and return result.

    Format:
      step1 = 48/2
      step2 = 48+step1
      [END]

    Returns: (result, reason, variables)
    """
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty", {}

    variables = {}
    last_result = None

    for line in lines:
        # Parse: var = expr
        match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not match:
            return None, f"parse_fail:{line[:20]}", variables

        var_name, expr = match.groups()
        expr = expr.strip()

        # Try to evaluate the expression
        try:
            # Replace variable references with their values
            eval_expr = expr
            for v, val in variables.items():
                eval_expr = re.sub(rf'\b{v}\b', str(val), eval_expr)

            # Evaluate (handles +, -, *, /)
            result = eval(eval_expr)
            variables[var_name] = result
            last_result = result

        except Exception as e:
            return None, f"eval_fail:{str(e)[:20]}", variables

    return last_result, "ok", variables


def compute_reward(ir_text: str, expected: int) -> tuple[float, str]:
    """Compute verifiable reward via execution."""
    result, reason, _ = execute_ir(ir_text)

    if result is None:
        return 0.0, reason

    # Handle float results (GSM8K sometimes has decimal intermediates)
    if isinstance(result, float):
        # Check if it's close to expected (within 0.01)
        if abs(result - expected) < 0.01:
            return 1.0, "correct"
        # Check if rounded matches
        if round(result) == expected:
            return 1.0, "correct_rounded"
    elif result == expected:
        return 1.0, "correct"

    return 0.3, f"wrong:{result}"


# =============================================================================
# DATA & GENERATION
# =============================================================================

def load_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_prompt(q: str) -> str:
    return f"Q: {q}\nIR:\n"


def format_target(q: str, ir: str) -> str:
    return f"Q: {q}\nIR:\n{ir}"


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if '[END]' in decoded or '\n\n\n' in decoded or '\nQ:' in decoded:
            break

    text = tokenizer.decode(generated).strip()
    if '\n\n\n' in text:
        text = text[:text.index('\n\n\n')]
    if '\nQ:' in text:
        text = text[:text.index('\nQ:')]

    return text


# =============================================================================
# TRAINING
# =============================================================================

def sft_batch(model, tokenizer, batch: list[dict], optimizer):
    def compute_loss(model, all_tokens, all_masks):
        total_loss = mx.array(0.0)

        for i in range(len(all_tokens)):
            tokens = mx.array([all_tokens[i]])
            mask = mx.array([all_masks[i]], dtype=mx.float32)

            logits = model(tokens)
            if hasattr(logits, 'logits'):
                logits = logits.logits

            logits = logits[:, :-1, :]
            targets = tokens[:, 1:]
            masks = mask[:, 1:]

            vocab_size = logits.shape[-1]
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction='none'
            ).reshape(targets.shape)

            total_loss = total_loss + (ce * masks).sum() / (masks.sum() + 1e-8)

        return total_loss / len(all_tokens)

    all_tokens = []
    all_masks = []

    for item in batch:
        full_text = format_target(item["q"], item["ir"])
        prompt = format_prompt(item["q"])

        full_tokens = tokenizer.encode(full_text)[:300]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

        all_tokens.append(full_tokens)
        all_masks.append(mask)

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, all_tokens, all_masks)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, tokenizer, data: list[dict], show: int = 0) -> dict:
    results = {"total": 0, "correct": 0, "parse_fail": 0, "wrong": 0}

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])

        results["total"] += 1
        if reward == 1.0:
            results["correct"] += 1
        elif reward == 0.0:
            results["parse_fail"] += 1
        else:
            results["wrong"] += 1

        if show > 0 and results["total"] <= show:
            status = "✓" if reward == 1.0 else f"✗ {reason}"
            q_short = item["q"][:60] + "..." if len(item["q"]) > 60 else item["q"]
            print(f"    Q: {q_short}")
            print(f"    Out: {output.replace(chr(10), ' | ')[:80]}...")
            print(f"    Status: {status} (ans={item['ans']})")
            print()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GSM8K IR TRAINING")
    print("  Learning to emit IR format from real GSM8K problems")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "gsm8k_ir_data"
    train_data = load_data(data_dir / "train.jsonl")
    val_data = load_data(data_dir / "val.jsonl")
    test_data = load_data(data_dir / "test.jsonl")

    print(f"\nData: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Baseline
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE")
    print("=" * 70)
    model.freeze()
    results = evaluate(model, tokenizer, test_data[:10], show=3)
    print(f"Baseline: {results['correct']}/{results['total']}")

    # Unfreeze last 4 layers + lm_head
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.model.layers[-4].unfreeze()
    model.lm_head.unfreeze()

    # SFT Phase
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (8 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    batch_size = 8

    # Use subset for faster iteration
    train_subset = train_data[:2000]

    for epoch in range(8):
        random.shuffle(train_subset)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_subset), batch_size):
            batch = train_subset[i:i+batch_size]
            loss = sft_batch(model, tokenizer, batch, sft_opt)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Quick eval
        train_results = evaluate(model, tokenizer, train_subset[:50])
        val_results = evaluate(model, tokenizer, val_data[:50])

        train_acc = train_results['correct'] / train_results['total']
        val_acc = val_results['correct'] / val_results['total']

        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}  train={train_acc:.0%}  val={val_acc:.0%}")

    # Detailed eval
    print("\n" + "=" * 70)
    print("DETAILED EVALUATION")
    print("=" * 70)

    print("\nSample outputs:")
    final = evaluate(model, tokenizer, test_data[:100], show=8)

    print(f"\nTest (100): {final['correct']}/{final['total']} = {final['correct']/final['total']:.0%}")
    print(f"  Parse fail: {final['parse_fail']}")
    print(f"  Wrong answer: {final['wrong']}")

    # Full test evaluation
    print("\n" + "=" * 70)
    print("FULL TEST SET EVALUATION")
    print("=" * 70)

    full_test = evaluate(model, tokenizer, test_data)
    print(f"\nFull test ({len(test_data)}): {full_test['correct']}/{full_test['total']} = {full_test['correct']/full_test['total']:.0%}")
    print(f"  Parse fail: {full_test['parse_fail']} ({full_test['parse_fail']/full_test['total']:.0%})")
    print(f"  Wrong answer: {full_test['wrong']} ({full_test['wrong']/full_test['total']:.0%})")


if __name__ == "__main__":
    main()
