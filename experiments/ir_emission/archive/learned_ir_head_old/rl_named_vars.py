"""
Named Variable Training - Multi-variable IR format.

Format:
  cost = 10 * 8
  remaining = 10 - 4
  revenue = remaining * 11
  profit = revenue - cost
  [END]
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
from gen_named_vars import execute_named, compute_reward


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
    return f"Q: {q}\nA:\n"


def format_target(q: str, expr: str) -> str:
    return f"Q: {q}\nA:\n{expr}"


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 150) -> str:
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
    # Keep [END] for parsing but remove extra stuff
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
        full_text = format_target(item["q"], item["expr"])
        prompt = format_prompt(item["q"])

        full_tokens = tokenizer.encode(full_text)[:200]
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
    from collections import defaultdict

    results = {"total": 0, "correct": 0, "by_pattern": defaultdict(lambda: {"total": 0, "correct": 0})}

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])

        results["total"] += 1
        if reward == 1.0:
            results["correct"] += 1

        pattern = item.get("pattern", "unknown")
        results["by_pattern"][pattern]["total"] += 1
        if reward == 1.0:
            results["by_pattern"][pattern]["correct"] += 1

        if show > 0 and results["total"] <= show:
            status = "✓" if reward == 1.0 else f"✗ {reason}"
            q_short = item["q"][:55] + "..." if len(item["q"]) > 55 else item["q"]
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
    print("  NAMED VARIABLE TRAINING")
    print("  Multi-variable IR format: var = expr")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "named_var_data"
    train_data = load_data(data_dir / "train.jsonl")
    test_data = load_data(data_dir / "test.jsonl")

    print(f"\nData: {len(train_data)} train, {len(test_data)} test")

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

    # Unfreeze
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.model.layers[-4].unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (12 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    batch_size = 8

    for epoch in range(12):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            loss = sft_batch(model, tokenizer, batch, sft_opt)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Quick eval
        train_results = evaluate(model, tokenizer, train_data[:30])
        test_results = evaluate(model, tokenizer, test_data[:30])

        train_acc = train_results['correct'] / train_results['total']
        test_acc = test_results['correct'] / test_results['total']

        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}  train={train_acc:.0%}  test={test_acc:.0%}")

    # Detailed eval
    print("\n" + "=" * 70)
    print("DETAILED EVALUATION")
    print("=" * 70)

    print("\nSample outputs:")
    final = evaluate(model, tokenizer, test_data, show=8)

    print(f"\nFull test: {final['correct']}/{final['total']} = {final['correct']/final['total']:.0%}")

    # By pattern
    print("\nAccuracy by pattern:")
    for pattern, stats in sorted(final["by_pattern"].items(), key=lambda x: -x[1]["total"]):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            marker = "✓" if acc >= 0.8 else "△" if acc >= 0.5 else "✗"
            print(f"  {marker} {pattern}: {stats['correct']}/{stats['total']} = {acc:.0%}")


if __name__ == "__main__":
    main()
