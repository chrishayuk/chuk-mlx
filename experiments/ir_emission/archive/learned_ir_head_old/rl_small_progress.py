"""
Small-Set RL Training with Visible Progress.

Uses 10 curated examples to show iteration-by-iteration progress.
"""

import sys
from pathlib import Path
import random

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from chain_cot_generator import compute_reward


# =============================================================================
# CURATED SMALL DATASET (10 examples)
# =============================================================================

TRAIN = [
    # 2-step problems
    {"q": "Tom has 20 apples. Eats 5. How many left?",
     "chain": "20 - 5 = 15\n[END]", "ans": 15},

    {"q": "Lisa has 8 boxes with 4 items each. Total items?",
     "chain": "8 * 4 = 32\n[END]", "ans": 32},

    {"q": "Sam has 30 coins. Gives away 12. Left?",
     "chain": "30 - 12 = 18\n[END]", "ans": 18},

    {"q": "5 shelves with 9 books each. Total books?",
     "chain": "5 * 9 = 45\n[END]", "ans": 45},

    # 3-step problems
    {"q": "Has 50. Spends 15, then 10 more. Left?",
     "chain": "50 - 15 = 35\n_ - 10 = 25\n[END]", "ans": 25},

    {"q": "Has 100. Loses 30, loses 20. Left?",
     "chain": "100 - 30 = 70\n_ - 20 = 50\n[END]", "ans": 50},

    {"q": "7 rows of 6. Sells 10. Left?",
     "chain": "7 * 6 = 42\n_ - 10 = 32\n[END]", "ans": 32},

    {"q": "Has 40. Uses 8. Sells rest at $2 each. Revenue?",
     "chain": "40 - 8 = 32\n_ * 2 = 64\n[END]", "ans": 64},

    # 4-step problem
    {"q": "Has 60. Gives 15, gives 10. Sells rest at $3 each. Revenue?",
     "chain": "60 - 15 = 45\n_ - 10 = 35\n_ * 3 = 105\n[END]", "ans": 105},

    {"q": "Has 80. Loses 20, loses 15. Divides rest among 9. Each gets?",
     "chain": "80 - 20 = 60\n_ - 15 = 45\n_ / 9 = 5\n[END]", "ans": 5},
]

TEST = [
    {"q": "Sam has 25 candies. Eats 8. Left?", "ans": 17},
    {"q": "6 bags with 11 items. Total?", "ans": 66},
    {"q": "Has 40. Loses 12, loses 9. Left?", "ans": 19},
    {"q": "Has 24. Uses 5, uses 3. Sells rest at $3 each. Revenue?", "ans": 48},
]


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:\n"

def format_target(question: str, chain: str) -> str:
    return f"Q: {question}\nA:\n{chain}"

def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 60) -> str:
    """Greedy generation with [END] stopping."""
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
# SFT TRAINING
# =============================================================================

def sft_epoch(model, tokenizer, data: list[dict], optimizer, epoch: int):
    """Single SFT epoch with progress."""

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
        full_text = format_target(item["q"], item["chain"])
        prompt = format_prompt(item["q"])

        full_tokens = tokenizer.encode(full_text)[:100]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

        tokens = mx.array([full_tokens])
        masks = mx.array([mask], dtype=mx.float32)

        loss, grads = loss_and_grad(model, tokens, masks)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

    return total_loss / len(data)


def evaluate_quick(model, tokenizer, data: list[dict]) -> tuple[int, int]:
    """Quick evaluation returning (correct, total)."""
    correct = 0
    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, _ = compute_reward(output, item["ans"])
        if reward == 1.0:
            correct += 1
    return correct, len(data)


def show_samples(model, tokenizer, data: list[dict], n: int = 3):
    """Show sample outputs."""
    for item in data[:n]:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])
        status = "✓" if reward == 1.0 else f"✗ {reason}"

        print(f"    Q: {item['q'][:50]}...")
        print(f"    Got: {output.replace(chr(10), ' | ')}")
        print(f"    Status: {status}")
        print()


# =============================================================================
# RL TRAINING
# =============================================================================

def rl_iteration(model, tokenizer, data: list[dict], optimizer, baseline: float,
                 temperature: float = 0.7) -> tuple[float, float, int]:
    """Single RL iteration with REINFORCE."""

    # Sample from each training example
    samples = []
    for item in data:
        prompt = format_prompt(item["q"])

        # Sample generation
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        generated = []

        for _ in range(60):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            logits = logits[:, -1, :] / temperature

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

    # Compute statistics
    avg_reward = sum(s[2] for s in samples) / len(samples)
    perfect = sum(1 for s in samples if s[2] == 1.0)

    # Compute policy gradient loss
    def compute_loss(model):
        total_loss = mx.array(0.0)

        for item, _, reward in samples:
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
    print("  SMALL-SET RL WITH VISIBLE PROGRESS")
    print("  10 train examples, iteration-by-iteration progress")
    print("=" * 70)

    print(f"\nTraining: {len(TRAIN)} examples")
    print(f"Test: {len(TEST)} examples")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Baseline
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE")
    print("=" * 70)
    model.freeze()
    correct, total = evaluate_quick(model, tokenizer, TEST)
    print(f"\nBaseline test: {correct}/{total}")
    print("\nSample outputs (baseline):")
    show_samples(model, tokenizer, TEST, n=2)

    # Unfreeze layers
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.lm_head.unfreeze()

    # SFT Phase
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (10 epochs)")
    print("=" * 70)

    sft_optimizer = optim.Adam(learning_rate=2e-5)

    for epoch in range(10):
        loss = sft_epoch(model, tokenizer, TRAIN, sft_optimizer, epoch)
        train_c, train_t = evaluate_quick(model, tokenizer, TRAIN)
        test_c, test_t = evaluate_quick(model, tokenizer, TEST)

        print(f"  Epoch {epoch+1:2d}: loss={loss:.4f}  train={train_c}/{train_t}  test={test_c}/{test_t}")

    print("\nSample outputs (after SFT):")
    show_samples(model, tokenizer, TEST)

    # RL Phase
    print("\n" + "=" * 70)
    print("PHASE 2: RL (20 iterations)")
    print("=" * 70)

    rl_optimizer = optim.Adam(learning_rate=5e-7)
    baseline = 0.5

    for iteration in range(20):
        avg_reward, baseline, perfect = rl_iteration(
            model, tokenizer, TRAIN, rl_optimizer, baseline, temperature=0.7
        )

        # Evaluate on test
        test_c, test_t = evaluate_quick(model, tokenizer, TEST)

        print(f"  Iter {iteration+1:2d}: reward={avg_reward:.2f}  perfect={perfect}/{len(TRAIN)}  test={test_c}/{test_t}")

        # Show samples every 5 iterations
        if (iteration + 1) % 5 == 0:
            print("\n  Sample outputs:")
            show_samples(model, tokenizer, TEST[:2])

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    train_c, train_t = evaluate_quick(model, tokenizer, TRAIN)
    test_c, test_t = evaluate_quick(model, tokenizer, TEST)

    print(f"\n  Train: {train_c}/{train_t} = {train_c/train_t:.0%}")
    print(f"  Test:  {test_c}/{test_t} = {test_c/test_t:.0%}")

    print("\nAll test outputs:")
    show_samples(model, tokenizer, TEST, n=len(TEST))


if __name__ == "__main__":
    main()
