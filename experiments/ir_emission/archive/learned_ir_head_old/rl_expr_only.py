"""
Expression-Only Chain Format - Model doesn't compute, just emits expressions.

Model output:
  25 - 8 =
  _ - 9 =
  [END]

Verifier executes each line and chains results.
Model doesn't need to know arithmetic!
"""

import sys
from pathlib import Path
import random
import re

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# EXPRESSION-ONLY FORMAT
# =============================================================================

def execute_chain(chain_text: str) -> tuple[int | None, str]:
    """
    Execute expression chain. Model outputs expressions, we compute results.

    Returns (final_result, reason)
    """
    lines = [l.strip() for l in chain_text.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty"

    current = None

    for line in lines:
        # Pattern: [number or _] [op] [number] =
        # With or without result on right side
        match = re.match(r'(\d+|_)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)?', line)
        if not match:
            return None, f"parse_fail:{line[:20]}"

        left, op, right, claimed = match.groups()
        right = int(right)

        # Handle left operand
        if left == '_':
            if current is None:
                return None, "chain_break"
            left_val = current
        else:
            left_val = int(left)

        # Compute result (this is the key - WE compute, not model)
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
            return None, f"bad_op:{op}"

        current = result

    return current, "ok"


def compute_reward(chain_text: str, expected: int) -> tuple[float, str]:
    """
    Compute reward for expression chain.

    1.0 = final result matches expected
    0.5 = parsed but wrong result
    0.0 = parse failure
    """
    result, reason = execute_chain(chain_text)

    if result is None:
        return 0.0, reason

    if result == expected:
        return 1.0, "correct"

    return 0.3, f"wrong:{result}!={expected}"


# =============================================================================
# CURATED DATASET - Expression only (no results in target)
# =============================================================================

TRAIN = [
    # Single step
    {"q": "Tom has 20 apples. Eats 5. How many left?",
     "expr": "20 - 5 =\n[END]", "ans": 15},

    {"q": "Lisa has 8 boxes with 4 items each. Total items?",
     "expr": "8 * 4 =\n[END]", "ans": 32},

    {"q": "Sam has 30 coins. Gives away 12. Left?",
     "expr": "30 - 12 =\n[END]", "ans": 18},

    {"q": "5 shelves with 9 books each. Total books?",
     "expr": "5 * 9 =\n[END]", "ans": 45},

    {"q": "Has 15 and gets 7 more. Total?",
     "expr": "15 + 7 =\n[END]", "ans": 22},

    {"q": "Divides 36 among 4. Each gets?",
     "expr": "36 / 4 =\n[END]", "ans": 9},

    # Multi-step
    {"q": "Has 50. Spends 15, then 10 more. Left?",
     "expr": "50 - 15 =\n_ - 10 =\n[END]", "ans": 25},

    {"q": "Has 100. Loses 30, loses 20. Left?",
     "expr": "100 - 30 =\n_ - 20 =\n[END]", "ans": 50},

    {"q": "7 rows of 6. Sells 10. Left?",
     "expr": "7 * 6 =\n_ - 10 =\n[END]", "ans": 32},

    {"q": "Has 40. Uses 8. Sells rest at $2 each. Revenue?",
     "expr": "40 - 8 =\n_ * 2 =\n[END]", "ans": 64},

    # More variety
    {"q": "3 packs with 12 each. Total?",
     "expr": "3 * 12 =\n[END]", "ans": 36},

    {"q": "Has 45, loses 18. Left?",
     "expr": "45 - 18 =\n[END]", "ans": 27},
]

TEST = [
    {"q": "Sam has 25 candies. Eats 8. Left?", "ans": 17},
    {"q": "6 bags with 11 items. Total?", "ans": 66},
    {"q": "Has 40. Loses 12, loses 9. Left?", "ans": 19},
    {"q": "Has 24. Uses 5, uses 3. Sells rest at $3 each. Revenue?", "ans": 48},
    {"q": "9 rows of 8. Remove 15. Left?", "ans": 57},
    {"q": "Has 60, gets 25 more. Total?", "ans": 85},
]


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:\n"

def format_target(question: str, expr: str) -> str:
    return f"Q: {question}\nA:\n{expr}"

def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
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

def sft_epoch(model, tokenizer, data: list[dict], optimizer):
    def compute_loss(model, tokens, mask):
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

        loss = (ce * masks).sum() / (masks.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    total_loss = 0
    for item in data:
        full_text = format_target(item["q"], item["expr"])
        prompt = format_prompt(item["q"])

        full_tokens = tokenizer.encode(full_text)[:80]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

        tokens = mx.array([full_tokens])
        masks = mx.array([mask], dtype=mx.float32)

        loss, grads = loss_and_grad(model, tokens, masks)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

    return total_loss / len(data)


def evaluate(model, tokenizer, data: list[dict], show: bool = False) -> tuple[int, int]:
    correct = 0
    results = []

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])

        if reward == 1.0:
            correct += 1

        results.append((item, output, reward, reason))

    if show:
        for item, output, reward, reason in results[:4]:
            status = "✓" if reward == 1.0 else f"✗ {reason}"
            q_short = item["q"][:45] + "..." if len(item["q"]) > 45 else item["q"]
            print(f"    Q: {q_short}")
            print(f"    Out: {output.replace(chr(10), ' | ')}")
            print(f"    Status: {status} (expected {item['ans']})")
            print()

    return correct, len(data)


def rl_iteration(model, tokenizer, data: list[dict], optimizer, baseline: float):
    samples = []

    for item in data:
        prompt = format_prompt(item["q"])

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        generated = []

        for _ in range(50):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            logits = logits[:, -1, :] / 0.7  # temperature

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
        samples.append((item, output_text, reward))

    avg_reward = sum(s[2] for s in samples) / len(samples)
    perfect = sum(1 for s in samples if s[2] == 1.0)

    def compute_loss(model):
        total_loss = mx.array(0.0)

        for item, _, reward in samples:
            prompt = format_prompt(item["q"])
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            gen_log_prob = mx.array(0.0)
            for _ in range(25):
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
    print("  EXPRESSION-ONLY CHAIN FORMAT")
    print("  Model emits: 25 - 8 =")
    print("  Verifier executes to get result")
    print("=" * 70)

    print(f"\nTrain: {len(TRAIN)} examples")
    print(f"Test: {len(TEST)} examples")

    # Test reward function
    print("\n--- Testing reward function ---")
    test_cases = [
        ("25 - 8 =", 17),     # Should compute 17, expect 17 → correct
        ("6 * 11 =", 66),     # Should compute 66, expect 66 → correct
        ("40 - 12 =\n_ - 9 =", 19),  # 40-12=28, 28-9=19 → correct
        ("25 - 8 = 15", 17),  # Has wrong result but we ignore it
        ("bad input", 10),
    ]
    for expr, expected in test_cases:
        reward, reason = compute_reward(expr, expected)
        result, _ = execute_chain(expr)
        print(f"  '{expr[:30]}...' → computed={result}, expected={expected}, reward={reward:.1f} ({reason})")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Baseline
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE")
    print("=" * 70)
    model.freeze()
    c, t = evaluate(model, tokenizer, TEST, show=True)
    print(f"Baseline: {c}/{t}")

    # Unfreeze
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (15 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)

    for epoch in range(15):
        loss = sft_epoch(model, tokenizer, TRAIN, sft_opt)
        train_c, train_t = evaluate(model, tokenizer, TRAIN)
        test_c, test_t = evaluate(model, tokenizer, TEST)
        print(f"  Epoch {epoch+1:2d}: loss={loss:.4f}  train={train_c}/{train_t}  test={test_c}/{test_t}")

    print("\nSample outputs (after SFT):")
    evaluate(model, tokenizer, TEST, show=True)

    # RL
    print("\n" + "=" * 70)
    print("PHASE 2: RL (15 iterations)")
    print("=" * 70)

    rl_opt = optim.Adam(learning_rate=5e-7)
    baseline = 0.5

    for i in range(15):
        avg_reward, baseline, perfect = rl_iteration(model, tokenizer, TRAIN, rl_opt, baseline)
        test_c, test_t = evaluate(model, tokenizer, TEST)
        print(f"  Iter {i+1:2d}: reward={avg_reward:.2f}  perfect={perfect}/{len(TRAIN)}  test={test_c}/{test_t}")

        if (i + 1) % 5 == 0:
            print("\n  Samples:")
            evaluate(model, tokenizer, TEST[:2], show=True)

    # Final
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    train_c, _ = evaluate(model, tokenizer, TRAIN)
    test_c, _ = evaluate(model, tokenizer, TEST)

    print(f"\n  Train: {train_c}/{len(TRAIN)}")
    print(f"  Test:  {test_c}/{len(TEST)}")

    print("\nAll test outputs:")
    evaluate(model, tokenizer, TEST, show=True)


if __name__ == "__main__":
    main()
