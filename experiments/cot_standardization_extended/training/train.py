#!/usr/bin/env python3
"""
Train Extended CoT Format for Verifiable Reasoning.

Based on cot_standardization, but with YAML trace format:

```yaml
expert: entity_track
trace:
  - {init: eggs, value: 16}
  - {consume: {entity: eggs, amount: 3}}
  - {state: {eggs: 13}}
answer: 13
```

Usage:
    python train.py --minimal-sft
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from trace.verifier import verify_yaml_output
from data.generate import generate_all, format_yaml


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You convert math problems to verifiable YAML traces.

Format:
```yaml
expert: <type>
trace:
  - {step}
  - {step}
answer: <number>
```

Expert types: entity_track, arithmetic, rate_equation, comparison, percentage

Example:
Q: Bob has 20 apples. He gives 5 to Carol. How many does Bob have?
```yaml
expert: entity_track
trace:
  - {init: bob.apples, value: 20}
  - {consume: {entity: bob.apples, amount: 5}}
  - {state: {bob.apples: 15}}
answer: 15
```"""


# =============================================================================
# FORMAT
# =============================================================================

def format_chat_prompt(question: str) -> str:
    """Format using TinyLlama's chat template."""
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n```yaml\n"


def format_target(example: dict) -> str:
    """Format target as YAML."""
    yaml_str = yaml.dump({
        "expert": example["expert"],
        "trace": example["trace"],
        "answer": example["answer"],
    }, default_flow_style=None, sort_keys=False)
    return yaml_str + "```"


def format_full_target(example: dict) -> str:
    """Full training target."""
    return format_chat_prompt(example["question"]) + format_target(example) + "</s>"


# =============================================================================
# REWARD
# =============================================================================

def compute_reward(output: str, example: dict) -> tuple[float, str]:
    """Compute verified reward."""
    # Extract YAML from output
    yaml_str = output.strip()
    if yaml_str.startswith("```yaml"):
        yaml_str = yaml_str[7:]
    if yaml_str.startswith("```"):
        yaml_str = yaml_str[3:]
    if "```" in yaml_str:
        yaml_str = yaml_str.split("```")[0]

    # Verify
    result = verify_yaml_output(yaml_str, expected_answer=example["answer"])

    if not result["parsed"]:
        return 0.0, "parse_fail"

    if result["expert"] != example["expert"]:
        return 0.3, f"wrong_expert:{result['expert']}"

    if not result["trace_valid"]:
        return 0.5, f"invalid_trace:{result['trace_error']}"

    if not result["answer_correct"]:
        return 0.7, f"wrong_answer:{result['answer']}"

    return 1.0, "correct"


# =============================================================================
# GENERATION
# =============================================================================

def generate(model, tokenizer, prompt: str, max_tokens: int = 200, greedy: bool = True, temp: float = 0.7) -> str:
    """Generate completion."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if greedy:
            next_token = int(mx.argmax(logits).item())
        else:
            next_token = int(mx.random.categorical(logits / temp).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        text = tokenizer.decode(generated)
        if '</s>' in text:
            break
        # Stop at end of YAML block
        if '```' in text and text.count('```') >= 1 and not text.strip().endswith('```yaml'):
            break

    output = tokenizer.decode(generated).strip()
    output = output.replace('</s>', '').strip()
    return output


# =============================================================================
# TRAINING
# =============================================================================

def sft_step(model, tokenizer, batch: list[dict], optimizer, max_len: int = 512):
    """One SFT step."""

    def loss_fn(model, tokens_list, masks_list):
        total = mx.array(0.0)
        for tokens, mask in zip(tokens_list, masks_list):
            toks = mx.array([tokens])
            logits = model(toks)
            if hasattr(logits, "logits"):
                logits = logits.logits

            logits = logits[:, :-1, :]
            targets = toks[:, 1:]
            m = mx.array([mask[1:]], dtype=mx.float32)

            ce = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none"
            ).reshape(targets.shape)

            total = total + (ce * m).sum() / (m.sum() + 1e-8)
        return total / len(tokens_list)

    tokens_list, masks_list = [], []
    for ex in batch:
        full = format_full_target(ex)
        prompt = format_chat_prompt(ex["question"])

        full_toks = tokenizer.encode(full)[:max_len]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_toks) - prompt_len)
        tokens_list.append(full_toks)
        masks_list.append(mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens_list, masks_list)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, tokenizer, data: list[dict], show: int = 0) -> dict:
    """Evaluate examples."""
    stats = {"total": 0, "correct": 0, "parsed": 0, "valid_trace": 0, "by_expert": {}}

    for ex in data:
        prompt = format_chat_prompt(ex["question"])
        output = generate(model, tokenizer, prompt, greedy=True)
        reward, reason = compute_reward(output, ex)

        stats["total"] += 1
        if reward >= 0.7:
            stats["correct"] += 1
        if reward >= 0.0 and "parse" not in reason:
            stats["parsed"] += 1
        if reward >= 0.5:
            stats["valid_trace"] += 1

        # Track by expert
        expert = ex["expert"]
        if expert not in stats["by_expert"]:
            stats["by_expert"][expert] = {"total": 0, "correct": 0}
        stats["by_expert"][expert]["total"] += 1
        if reward >= 0.7:
            stats["by_expert"][expert]["correct"] += 1

        if show > 0 and stats["total"] <= show:
            mark = "✓" if reward >= 0.7 else "✗"
            q = ex["question"][:50] + "..." if len(ex["question"]) > 50 else ex["question"]
            print(f"  {mark} [{ex['expert']:15}] {q}")
            print(f"      Output: {output[:60]}...")
            print(f"      Reason: {reason}")

    return stats


def rl_step(model, tokenizer, data: list[dict], optimizer, baseline: float, batch_size: int = 8, temp: float = 0.7, max_tokens: int = 200):
    """One REINFORCE step with proper policy gradient."""
    batch = random.sample(data, min(batch_size, len(data)))

    # Sample outputs and collect token sequences
    samples = []
    for ex in batch:
        prompt = format_chat_prompt(ex["question"])
        prompt_tokens = tokenizer.encode(prompt)

        # Generate with sampling
        input_ids = mx.array([prompt_tokens])
        generated_tokens = []

        for _ in range(max_tokens):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            logits = logits[:, -1, :] / temp

            next_token = int(mx.random.categorical(logits).item())
            generated_tokens.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            text = tokenizer.decode(generated_tokens)
            if '</s>' in text:
                break
            if '```' in text and text.count('```') >= 1 and not text.strip().endswith('```yaml'):
                break

        output_text = tokenizer.decode(generated_tokens).replace('</s>', '').strip()
        reward, reason = compute_reward(output_text, ex)
        samples.append((ex, prompt_tokens, generated_tokens, reward, reason))

    avg_reward = sum(s[3] for s in samples) / len(samples)

    # Compute policy gradient loss
    def loss_fn(model):
        total = mx.array(0.0)
        for ex, prompt_tokens, gen_tokens, reward, _ in samples:
            if not gen_tokens:
                continue

            full_tokens = prompt_tokens + gen_tokens
            input_ids = mx.array([full_tokens[:-1]])
            targets = mx.array([full_tokens[1:]])

            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            gen_start = len(prompt_tokens) - 1
            gen_logits = logits[:, gen_start:, :]
            gen_targets = targets[:, gen_start:]

            log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)
            chosen_log_probs = mx.take_along_axis(
                log_probs,
                gen_targets[:, :, None],
                axis=-1
            ).squeeze(-1)

            seq_log_prob = chosen_log_probs.sum()
            advantage = reward - baseline
            total = total - seq_log_prob * advantage

        return total / max(len(samples), 1)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    new_baseline = 0.9 * baseline + 0.1 * avg_reward
    correct = sum(1 for s in samples if s[3] >= 0.7)

    return avg_reward, new_baseline, correct


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--rl-iters", type=int, default=20)
    parser.add_argument("--minimal-sft", action="store_true", help="1 epoch SFT + more RL")
    parser.add_argument("--fast", action="store_true", help="Faster RL with smaller batches")
    parser.add_argument("--examples", type=int, default=235, help="Total examples to generate")
    args = parser.parse_args()

    # Config
    rl_batch_size = 4 if args.fast else 8
    max_gen_tokens = 150 if args.fast else 200

    if args.minimal_sft:
        args.sft_epochs = 1
        if args.rl_iters < 20:
            args.rl_iters = 20

    print("=" * 70)
    print("  EXTENDED COT TRAINING: YAML TRACE FORMAT")
    print("=" * 70)

    # Generate data
    train_data = generate_all(
        n_entity=100,
        n_arithmetic=40,
        n_rate=40,
        n_comparison=40,
        n_percentage=15,
    )
    eval_data = generate_all(
        n_entity=20,
        n_arithmetic=10,
        n_rate=10,
        n_comparison=10,
        n_percentage=5,
    )

    # Load model
    print(f"\nLoading {args.model}...")
    result = load_model(args.model)
    model, tokenizer = result.model, result.tokenizer

    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE (before training)")
    print("=" * 70)
    model.freeze()
    stats = evaluate(model, tokenizer, eval_data[:10], show=3)
    print(f"\nBaseline: {stats['correct']}/{stats['total']} correct")

    # Unfreeze layers
    for layer in model.model.layers[-6:]:
        layer.unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print(f"SFT ({args.sft_epochs} epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    for epoch in range(args.sft_epochs):
        random.shuffle(train_data)
        losses = []
        for i in range(0, len(train_data), 4):
            loss = sft_step(model, tokenizer, train_data[i:i+4], sft_opt)
            losses.append(loss)

        stats = evaluate(model, tokenizer, eval_data)
        acc = stats["correct"] / stats["total"]
        print(f"  Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f} acc={acc:.0%} parse={stats['parsed']/stats['total']:.0%}")

    print("\nAfter SFT:")
    evaluate(model, tokenizer, eval_data[:5], show=5)

    # RL
    if args.rl_iters > 0:
        print("\n" + "=" * 70)
        print(f"RL ({args.rl_iters} iterations, batch={rl_batch_size}, max_tokens={max_gen_tokens})")
        print("=" * 70)

        rl_opt = optim.Adam(learning_rate=5e-7)
        baseline = 0.5

        for i in range(args.rl_iters):
            avg_r, baseline, correct = rl_step(model, tokenizer, train_data, rl_opt, baseline, batch_size=rl_batch_size, max_tokens=max_gen_tokens)
            if (i + 1) % 5 == 0:
                stats = evaluate(model, tokenizer, eval_data)
                print(f"  Iter {i+1}: reward={avg_r:.2f} batch={correct}/8 eval={stats['correct']}/{stats['total']}")

    # Final
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final = evaluate(model, tokenizer, eval_data, show=5)
    print(f"\nOverall: {final['correct']}/{final['total']} ({final['correct']/final['total']:.0%})")
    print(f"Parsed: {final['parsed']}/{final['total']} ({final['parsed']/final['total']:.0%})")
    print(f"Valid traces: {final['valid_trace']}/{final['total']} ({final['valid_trace']/final['total']:.0%})")

    print("\nBy expert:")
    for expert, s in final["by_expert"].items():
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"  {expert:15} {acc:.0%} ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
