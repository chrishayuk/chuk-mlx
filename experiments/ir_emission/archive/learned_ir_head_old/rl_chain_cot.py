"""
RL with Chain CoT Format.

Format:
  16 - 3 = 13
  _ - 4 = 9
  _ * 2 = 18

Train with SFT cold start, then RL with verifiable rewards.
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
from chain_cot_generator import verify_chain, compute_reward


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


def format_target(question: str, chain: str) -> str:
    return f"Q: {question}\nA:\n{chain}"


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 80) -> str:
    """Greedy generation."""
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
        # Stop at [END] marker, double newline, or Q:
        if '[END]' in decoded or '\n\n' in decoded or '\nQ:' in decoded:
            break

    text = tokenizer.decode(generated).strip()
    # Clean up - remove [END] and anything after
    if '[END]' in text:
        text = text[:text.index('[END]')].strip()
    if '\n\n' in text:
        text = text[:text.index('\n\n')]
    if '\nQ:' in text:
        text = text[:text.index('\nQ:')]

    return text


def sample_from_model(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 80):
    """Sample with temperature."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    log_probs = []

    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        logits = logits[:, -1, :] / temperature

        probs = mx.softmax(logits, axis=-1)
        next_token = int(mx.random.categorical(logits).item())

        if next_token == tokenizer.eos_token_id:
            break

        log_prob = mx.log(probs[0, next_token] + 1e-10)
        log_probs.append(log_prob)

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if '[END]' in decoded or '\n\n' in decoded or '\nQ:' in decoded:
            break

    text = tokenizer.decode(generated).strip()
    if '[END]' in text:
        text = text[:text.index('[END]')].strip()
    if '\n\n' in text:
        text = text[:text.index('\n\n')]
    if '\nQ:' in text:
        text = text[:text.index('\nQ:')]

    return text, log_probs


# =============================================================================
# SFT TRAINING
# =============================================================================

def sft_train(model, tokenizer, data: list[dict], epochs: int = 5, lr: float = 2e-5, batch_size: int = 4):
    """SFT on chain format."""
    print(f"\n  SFT Training: {epochs} epochs, {len(data)} examples, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)

    def compute_loss(model, batch_tokens, batch_masks):
        logits = model(batch_tokens)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        logits = logits[:, :-1, :]
        targets = batch_tokens[:, 1:]
        masks = batch_masks[:, 1:]

        vocab_size = logits.shape[-1]
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        ).reshape(targets.shape)

        loss = (ce * masks).sum() / (masks.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]

            all_tokens = []
            all_masks = []

            for item in batch:
                full_text = format_target(item["q"], item["chain"])
                prompt = format_prompt(item["q"])

                full_tokens = tokenizer.encode(full_text)[:150]
                prompt_tokens = tokenizer.encode(prompt)

                mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

                all_tokens.append(full_tokens)
                all_masks.append(mask)

            max_len = max(len(t) for t in all_tokens)
            padded_tokens = [t + [0] * (max_len - len(t)) for t in all_tokens]
            padded_masks = [m + [0] * (max_len - len(m)) for m in all_masks]

            batch_tokens = mx.array(padded_tokens)
            batch_masks = mx.array(padded_masks, dtype=mx.float32)

            loss, grads = loss_and_grad(model, batch_tokens, batch_masks)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model


# =============================================================================
# RL TRAINING
# =============================================================================

def rl_train(model, tokenizer, data: list[dict], iterations: int = 20, lr: float = 5e-7,
             batch_size: int = 8, temperature: float = 0.7):
    """REINFORCE with verifiable rewards."""
    print(f"\n  RL Training: {iterations} iterations, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    baseline = 0.5

    for iteration in range(iterations):
        batch = random.sample(data, min(batch_size, len(data)))
        samples = []

        for item in batch:
            prompt = format_prompt(item["q"])
            output, log_probs = sample_from_model(model, tokenizer, prompt, temperature)
            reward, reason = compute_reward(output, item["ans"])
            samples.append((item, output, log_probs, reward, reason))

        avg_reward = sum(s[3] for s in samples) / len(samples)
        perfect = sum(1 for s in samples if s[3] == 1.0)

        def compute_loss(model):
            total_loss = mx.array(0.0)

            for item, _, _, reward, _ in samples:
                prompt = format_prompt(item["q"])
                tokens = tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

                gen_log_prob = mx.array(0.0)
                for _ in range(40):
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

        baseline = 0.9 * baseline + 0.1 * avg_reward

        if (iteration + 1) % 5 == 0:
            print(f"    Iter {iteration+1}: reward={avg_reward:.2f}, perfect={perfect}/{batch_size}")

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, tokenizer, data: list[dict], label: str = "", verbose: bool = True):
    """Evaluate with verification."""
    perfect = 0
    partial = 0

    if verbose:
        print(f"\n  {label}")
        print("-" * 80)

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = compute_reward(output, item["ans"])

        if reward == 1.0:
            perfect += 1
            status = "✓"
        elif reward >= 0.3:
            partial += 1
            status = f"~ {reason}"
        else:
            status = f"✗ {reason}"

        if verbose:
            q_short = item["q"][:55] + "..." if len(item["q"]) > 55 else item["q"]
            print(f"\n  Q: {q_short}")
            print(f"  Expected: {item['chain'].replace(chr(10), ' | ')}")
            print(f"  Got:      {output.replace(chr(10), ' | ')}")
            print(f"  Status:   {status}")

    accuracy = perfect / len(data)
    if verbose:
        print(f"\n  Perfect: {perfect}/{len(data)} = {accuracy:.0%}")
        print(f"  Partial: {partial}/{len(data)}")

    return accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  RL WITH CHAIN COT FORMAT")
    print("  X op Y = Z  |  _ op Y = Z")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "chain_data"
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
    baseline_acc = evaluate(model, tokenizer, test_data[:5], "Baseline (5 examples)")

    # SFT
    print("\n" + "=" * 70)
    print("PHASE 1: SFT COLD START")
    print("=" * 70)

    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.lm_head.unfreeze()

    # Use subset for faster iteration
    model = sft_train(model, tokenizer, train_data[:500], epochs=5, lr=2e-5, batch_size=4)
    sft_acc = evaluate(model, tokenizer, test_data[:10], "After SFT (10 examples)")

    # RL
    print("\n" + "=" * 70)
    print("PHASE 2: RL FINE-TUNE")
    print("=" * 70)
    model = rl_train(model, tokenizer, train_data[:200], iterations=20, lr=5e-7, batch_size=8)
    rl_acc = evaluate(model, tokenizer, test_data[:10], "After RL (10 examples)")

    # Full test
    print("\n" + "=" * 70)
    print("FULL TEST SET EVALUATION")
    print("=" * 70)
    final_acc = evaluate(model, tokenizer, test_data, "Full Test", verbose=False)
    print(f"\n  Full test accuracy: {final_acc:.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
    Chain CoT Format:
      16 - 3 = 13
      _ - 4 = 9
      _ * 2 = 18

    Each step verifiable independently.
    _ = previous result.

    Results:
      Baseline: {baseline_acc:.0%}
      After SFT: {sft_acc:.0%}
      After RL:  {rl_acc:.0%}
      Full test: {final_acc:.0%}
    """)


if __name__ == "__main__":
    main()
