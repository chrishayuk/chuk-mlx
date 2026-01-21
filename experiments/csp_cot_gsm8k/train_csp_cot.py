#!/usr/bin/env python3
"""
CSP-CoT Training with Verifiable Rewards.

Fine-tunes a small model to emit structured actions for GSM-8K problems.
Uses trace verification as the reward signal.

Reward structure:
- 1.0: Valid trace + correct answer
- 0.5: Valid trace + wrong answer (model understands format, solver works)
- 0.0: Invalid JSON or failed trace verification

Usage:
    python train_csp_cot.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from decimal import Decimal
from pathlib import Path
import functools

print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model

# CSP-CoT imports
from experiments.csp_cot_gsm8k.expert_standalone import MathSolver
from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import get_sample_problems


# =============================================================================
# VERIFIABLE REWARD
# =============================================================================

def compute_reward(
    output_text: str,
    expected_answer: float,
    expert: MathSolver,
) -> tuple[float, str, dict | None]:
    """
    Compute reward from model output using trace verification.

    Returns:
        (reward, reason, parsed_action)
    """
    # Try to extract JSON from output
    action_dict = extract_json(output_text)

    if action_dict is None:
        return 0.0, "parse_fail", None

    # Check if it's a passthrough (not a math problem)
    if action_dict.get("expert") == "none":
        return 0.0, "passthrough", action_dict

    # Check if it targets our expert
    if action_dict.get("expert") != "math_word_problem":
        return 0.0, "wrong_expert", action_dict

    # Try to execute via expert
    try:
        parameters = action_dict.get("parameters", {})
        result = expert.solve(**parameters)

        if not result.get("success"):
            return 0.0, f"exec_fail:{result.get('error', 'unknown')}", action_dict

        answer = result.get("answer")
        verified = result.get("verified", False)

        if not verified:
            return 0.0, "invalid_trace", action_dict

        # Valid trace - check answer
        if answer is not None and abs(answer - expected_answer) < 0.01:
            return 1.0, "correct", action_dict
        else:
            return 0.5, f"wrong:{answer}", action_dict

    except Exception as e:
        return 0.0, f"exception:{str(e)[:50]}", action_dict


def extract_json(text: str) -> dict | None:
    """Extract JSON object from model output."""
    try:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        end = start
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_str = text[start:end]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None


# =============================================================================
# DATA
# =============================================================================

def load_training_data() -> list[dict]:
    """Load CoT examples for training."""
    path = Path(__file__).parent / "cot_examples.json"
    with open(path) as f:
        data = json.load(f)

    examples = []
    for ex in data["examples"]:
        if ex["action"]["expert"] == "math_word_problem":
            examples.append({
                "question": ex["query"],
                "action": ex["action"],
                "answer": compute_expected_answer(ex["action"]),
            })

    return examples


def compute_expected_answer(action: dict) -> float | None:
    """Compute expected answer from action using expert."""
    expert = MathSolver()
    params = action.get("parameters", {})
    result = expert.solve(**params)
    return result.get("answer") if result.get("success") else None


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a math problem parser. Extract structured parameters for solving.

## Schema
- problem_type: entity_tracking | arithmetic_chain | comparison | allocation
- entities: [{"name": "<name>", "initial_value": <NUMBER>}, ...]
- operations: [{"type": "add|subtract|multiply|divide|transfer", "target": "<entity>", "amount": <NUMBER>}, ...]
- query: {"target": "<entity>"}

## Key Rules
1. "amount" and "factor" must be NUMBERS, not entity names
2. Use "amount" for add/subtract, "factor" for multiply/divide
3. Chain operations on a SINGLE entity for arithmetic
4. Query target should match the entity you want the final value of

Respond with ONLY valid JSON."""


def format_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQ: {question}\nAction:"


def format_target(question: str, action: dict) -> str:
    action_json = json.dumps(action, indent=2)
    return f"{SYSTEM_PROMPT}\n\nQ: {question}\nAction: {action_json}"


# =============================================================================
# GENERATION
# =============================================================================

def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
    """Greedy decoding for evaluation."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        # Stop if we have complete JSON
        if decoded.count("{") > 0 and decoded.count("{") == decoded.count("}"):
            break

    return tokenizer.decode(generated).strip()


def sample_generate(model, tokenizer, prompt: str, max_tokens: int = 300, temp: float = 0.7) -> str:
    """Sampling for RL."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :] / temp

        next_token = int(mx.random.categorical(logits).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if decoded.count("{") > 0 and decoded.count("{") == decoded.count("}"):
            break

    return tokenizer.decode(generated).strip()


# =============================================================================
# TRAINING
# =============================================================================

def sft_batch(model, tokenizer, batch: list[dict], optimizer):
    """Supervised fine-tuning on one batch."""

    def compute_loss(model, all_tokens, all_masks):
        total_loss = mx.array(0.0)

        for i in range(len(all_tokens)):
            tokens = mx.array([all_tokens[i]])
            mask = mx.array([all_masks[i]], dtype=mx.float32)

            logits = model(tokens)
            if hasattr(logits, "logits"):
                logits = logits.logits

            logits = logits[:, :-1, :]
            targets = tokens[:, 1:]
            masks = mask[:, 1:]

            vocab_size = logits.shape[-1]
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)

            total_loss = total_loss + (ce * masks).sum() / (masks.sum() + 1e-8)

        return total_loss / len(all_tokens)

    # Prepare batch
    all_tokens = []
    all_masks = []

    for item in batch:
        full_text = format_target(item["question"], item["action"])
        prompt = format_prompt(item["question"])

        full_tokens = tokenizer.encode(full_text)[:512]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

        all_tokens.append(full_tokens)
        all_masks.append(mask)

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, all_tokens, all_masks)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, tokenizer, expert: MathSolver, data: list[dict], show: int = 0) -> dict:
    """Evaluate model on data."""
    results = {"total": 0, "correct": 0, "valid": 0, "parse_fail": 0}

    for item in data:
        prompt = format_prompt(item["question"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason, _ = compute_reward(output, item["answer"] or 0, expert)

        results["total"] += 1
        if reward == 1.0:
            results["correct"] += 1
        if reward >= 0.5:
            results["valid"] += 1
        if reward == 0.0 and "parse" in reason:
            results["parse_fail"] += 1

        if show > 0 and results["total"] <= show:
            status = "✓" if reward == 1.0 else f"✗ {reason}"
            q_short = item["question"][:60] + "..." if len(item["question"]) > 60 else item["question"]
            print(f"    Q: {q_short}")
            print(f"    Out: {output[:100]}...")
            print(f"    Status: {status}")
            print()

    return results


def rl_iteration(
    model,
    tokenizer,
    expert: MathSolver,
    data: list[dict],
    optimizer,
    baseline: float,
    batch_size: int = 8,
):
    """One iteration of REINFORCE with verifiable rewards."""
    batch = random.sample(data, min(batch_size, len(data)))
    samples = []

    for item in batch:
        prompt = format_prompt(item["question"])
        output = sample_generate(model, tokenizer, prompt)
        reward, reason, _ = compute_reward(output, item["answer"] or 0, expert)
        samples.append((item, reward, output))

    avg_reward = sum(s[1] for s in samples) / len(samples)
    correct = sum(1 for s in samples if s[1] == 1.0)
    valid = sum(1 for s in samples if s[1] >= 0.5)

    def compute_loss(model):
        total_loss = mx.array(0.0)

        for item, reward, _ in samples:
            prompt = format_prompt(item["question"])
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            gen_log_prob = mx.array(0.0)
            for _ in range(50):  # Shorter for RL
                output = model(input_ids)
                logits = output.logits if hasattr(output, "logits") else output
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

    return avg_reward, new_baseline, correct, valid


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  CSP-CoT TRAINING WITH VERIFIABLE REWARDS")
    print("=" * 70)

    # Create expert for verification
    expert = MathSolver()
    print("Expert: MathSolver (standalone)")

    # Load training data
    train_data = load_training_data()
    print(f"Training examples: {len(train_data)}")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Baseline evaluation
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE")
    print("=" * 70)
    model.freeze()
    results = evaluate(model, tokenizer, expert, train_data[:5], show=2)
    print(f"Baseline: {results['correct']}/{results['total']} correct, {results['valid']}/{results['total']} valid")

    # Unfreeze last few layers
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.lm_head.unfreeze()

    # SFT Phase
    print("\n" + "=" * 70)
    print("PHASE 1: SFT (10 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    batch_size = 4

    for epoch in range(10):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            loss = sft_batch(model, tokenizer, batch, sft_opt)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Eval
        train_results = evaluate(model, tokenizer, expert, train_data)
        train_acc = train_results["correct"] / train_results["total"]
        valid_rate = train_results["valid"] / train_results["total"]

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  correct={train_acc:.0%}  valid={valid_rate:.0%}")

    # Show some outputs after SFT
    print("\nSample outputs (after SFT):")
    evaluate(model, tokenizer, expert, train_data[:3], show=3)

    # RL Phase
    print("\n" + "=" * 70)
    print("PHASE 2: RL (20 iterations)")
    print("=" * 70)

    rl_opt = optim.Adam(learning_rate=5e-7)
    baseline = 0.5

    for i in range(20):
        avg_reward, baseline, correct, valid = rl_iteration(
            model, tokenizer, expert, train_data, rl_opt, baseline, batch_size=8
        )

        if (i + 1) % 5 == 0:
            train_results = evaluate(model, tokenizer, expert, train_data)
            train_acc = train_results["correct"] / train_results["total"]
            print(f"  Iter {i+1:2d}: reward={avg_reward:.2f}  correct={correct}/8  train_acc={train_acc:.0%}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nSample outputs:")
    final_results = evaluate(model, tokenizer, expert, train_data, show=5)

    print(f"\nFinal: {final_results['correct']}/{final_results['total']} correct ({final_results['correct']/final_results['total']:.0%})")
    print(f"       {final_results['valid']}/{final_results['total']} valid traces ({final_results['valid']/final_results['total']:.0%})")
    print(f"       {final_results['parse_fail']}/{final_results['total']} parse failures")


if __name__ == "__main__":
    main()
