"""
Expression-Only Training at Scale.

Uses 500 training examples with diverse multi-step patterns.
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
# EXPRESSION EXECUTION
# =============================================================================

def execute_chain(chain_text: str) -> tuple[int | None, str]:
    """Execute expression chain - verifier does the math."""
    lines = [l.strip() for l in chain_text.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty"

    current = None

    for line in lines:
        match = re.match(r'(\d+|_)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)?', line)
        if not match:
            return None, f"parse_fail"

        left, op, right, _ = match.groups()
        right = int(right)

        if left == '_':
            if current is None:
                return None, "chain_break"
            left_val = current
        else:
            left_val = int(left)

        if op == '+':
            result = left_val + right
        elif op == '-':
            result = left_val - right
        elif op == '*':
            result = left_val * right
        elif op == '/':
            if right == 0:
                return None, "div_zero"
            result = left_val // right
        else:
            return None, f"bad_op"

        current = result

    return current, "ok"


def compute_reward(chain_text: str, expected: int) -> tuple[float, str]:
    result, reason = execute_chain(chain_text)

    if result is None:
        return 0.0, reason

    if result == expected:
        return 1.0, "correct"

    return 0.3, f"wrong:{result}"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:\n"

def format_target(question: str, expr: str) -> str:
    return f"Q: {question}\nA:\n{expr}"

def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 60) -> str:
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
        if '[END]' in decoded or '\n\n' in decoded or '\nQ:' in decoded:
            break

    text = tokenizer.decode(generated).strip()
    for stop in ['[END]', '\n\n', '\nQ:']:
        if stop in text:
            text = text[:text.index(stop)].strip()

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

    # Prepare batch
    all_tokens = []
    all_masks = []

    for item in batch:
        full_text = format_target(item["q"], item["expr"])
        prompt = format_prompt(item["q"])

        full_tokens = tokenizer.encode(full_text)[:100]
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
    """Evaluate and return stats by pattern."""
    results = {"total": 0, "correct": 0, "by_pattern": {}}

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])

        results["total"] += 1
        if reward == 1.0:
            results["correct"] += 1

        pattern = item.get("pattern", "unknown")
        if pattern not in results["by_pattern"]:
            results["by_pattern"][pattern] = {"total": 0, "correct": 0}
        results["by_pattern"][pattern]["total"] += 1
        if reward == 1.0:
            results["by_pattern"][pattern]["correct"] += 1

        if show > 0 and results["total"] <= show:
            status = "✓" if reward == 1.0 else f"✗ {reason}"
            q_short = item["q"][:50] + "..." if len(item["q"]) > 50 else item["q"]
            print(f"    Q: {q_short}")
            print(f"    Out: {output.replace(chr(10), ' | ')}")
            print(f"    Expected: {item['expr'].replace(chr(10), ' | ')}")
            print(f"    Status: {status} (ans={item['ans']})")
            print()

    return results


def rl_iteration(model, tokenizer, data: list[dict], optimizer, baseline: float, batch_size: int = 16):
    batch = random.sample(data, min(batch_size, len(data)))
    samples = []

    for item in batch:
        prompt = format_prompt(item["q"])

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        generated = []

        for _ in range(60):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            logits = logits[:, -1, :] / 0.7

            next_token = int(mx.random.categorical(logits).item())

            if next_token == tokenizer.eos_token_id:
                break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            decoded = tokenizer.decode(generated)
            if '[END]' in decoded or '\n\n' in decoded:
                break

        output_text = tokenizer.decode(generated).strip()
        for stop in ['[END]', '\n\n', '\nQ:']:
            if stop in output_text:
                output_text = output_text[:output_text.index(stop)].strip()

        reward, _ = compute_reward(output_text, item["ans"])
        samples.append((item, reward))

    avg_reward = sum(s[1] for s in samples) / len(samples)
    perfect = sum(1 for s in samples if s[1] == 1.0)

    def compute_loss(model):
        total_loss = mx.array(0.0)

        for item, reward in samples:
            prompt = format_prompt(item["q"])
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            gen_log_prob = mx.array(0.0)
            for _ in range(30):
                output = model(input_ids)
                logits = output.logits if hasattr(output, 'logits') else output
                logits = logits[:, -1, :]

                probs = mx.softmax(logits, axis=-1)
                next_token = mx.argmax(logits, axis=-1)
                log_prob = mx.log(mx.max(probs) + 1e-10)
                gen_log_prob = gen_log_prob + log_prob

                input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            advantage = reward - baseline
            total_loss = total_loss - gen_log_prob * advantage

        return total_loss / len(samples)

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    new_baseline = 0.9 * baseline + 0.1 * avg_reward

    return avg_reward, new_baseline, perfect


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPRESSION-ONLY TRAINING AT SCALE")
    print("  500 train examples, 60% multi-step")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "expr_data"
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
    results = evaluate(model, tokenizer, test_data[:20], show=3)
    print(f"Baseline: {results['correct']}/{results['total']}")

    # Unfreeze
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (8 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    batch_size = 8

    for epoch in range(8):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            loss = sft_batch(model, tokenizer, batch, sft_opt)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Quick eval on subset
        train_results = evaluate(model, tokenizer, train_data[:50])
        test_results = evaluate(model, tokenizer, test_data[:30])

        train_acc = train_results['correct'] / train_results['total']
        test_acc = test_results['correct'] / test_results['total']

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  train={train_acc:.0%}  test={test_acc:.0%}")

    # Detailed eval after SFT
    print("\nSample outputs (after SFT):")
    test_results = evaluate(model, tokenizer, test_data[:50], show=5)
    print(f"\nTest subset: {test_results['correct']}/{test_results['total']}")

    # Pattern breakdown
    print("\nBy pattern:")
    for pattern, stats in sorted(test_results["by_pattern"].items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {pattern}: {stats['correct']}/{stats['total']} = {acc:.0%}")

    # RL
    print("\n" + "=" * 70)
    print("PHASE 2: RL (30 iterations)")
    print("=" * 70)

    rl_opt = optim.Adam(learning_rate=5e-7)
    baseline = 0.5

    for i in range(30):
        avg_reward, baseline, perfect = rl_iteration(
            model, tokenizer, train_data, rl_opt, baseline, batch_size=16
        )

        if (i + 1) % 5 == 0:
            test_results = evaluate(model, tokenizer, test_data[:30])
            test_acc = test_results['correct'] / test_results['total']
            print(f"  Iter {i+1:2d}: reward={avg_reward:.2f}  perfect={perfect}/16  test={test_acc:.0%}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nSample outputs:")
    final_results = evaluate(model, tokenizer, test_data, show=6)

    print(f"\nFull test: {final_results['correct']}/{final_results['total']} = {final_results['correct']/final_results['total']:.0%}")

    print("\nBy pattern:")
    for pattern, stats in sorted(final_results["by_pattern"].items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {pattern}: {stats['correct']}/{stats['total']} = {acc:.0%}")


if __name__ == "__main__":
    main()
