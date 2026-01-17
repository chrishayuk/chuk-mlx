"""
Train on GSM8K-style synthetic data, evaluate on REAL GSM8K.

Tests whether matching the semantic distribution transfers to real problems.
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
# IR EXECUTION
# =============================================================================

def execute_ir(code: str) -> tuple[float | None, str]:
    """Execute IR code."""
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty"

    variables = {}
    last_result = None

    for line in lines:
        match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not match:
            return None, f"parse_fail"

        var_name, expr = match.groups()
        expr = expr.strip()

        try:
            eval_expr = expr
            for v, val in variables.items():
                eval_expr = re.sub(rf'\b{v}\b', str(val), eval_expr)

            result = eval(eval_expr)
            variables[var_name] = result
            last_result = result

        except Exception as e:
            return None, f"eval_fail"

    return last_result, "ok"


def compute_reward(ir_text: str, expected: int) -> tuple[float, str]:
    """Compute reward via execution."""
    result, reason = execute_ir(ir_text)

    if result is None:
        return 0.0, reason

    if isinstance(result, float):
        if abs(result - expected) < 0.01:
            return 1.0, "correct"
        if round(result) == expected:
            return 1.0, "correct"
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


def load_gsm8k_test(n: int = 100):
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")

    data = []
    for item in ds:
        match = re.search(r'####\s*(-?[\d,]+)', item["answer"])
        if match:
            data.append({
                "q": item["question"],
                "ans": int(match.group(1).replace(",", ""))
            })
        if len(data) >= n:
            break

    return data


def format_prompt(q: str) -> str:
    return f"Q: {q}\nIR:\n"


def format_target(q: str, ir: str) -> str:
    return f"Q: {q}\nIR:\n{ir}"


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

        full_tokens = tokenizer.encode(full_text)[:256]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

        all_tokens.append(full_tokens)
        all_masks.append(mask)

    loss_and_grad = nn.value_and_grad(model, compute_loss)
    loss, grads = loss_and_grad(model, all_tokens, all_masks)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, tokenizer, data: list[dict], show: int = 0, name: str = "") -> dict:
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
            q_short = item["q"][:55] + "..." if len(item["q"]) > 55 else item["q"]
            print(f"    Q: {q_short}")
            print(f"    Out: {output.replace(chr(10), ' | ')[:70]}...")
            print(f"    Status: {status} (ans={item['ans']})")
            print()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GSM8K-STYLE TRAINING")
    print("  Train on semantic patterns, test on REAL GSM8K")
    print("=" * 70)

    # Load synthetic data
    data_dir = Path(__file__).parent / "gsm8k_style_data"
    train_data = load_data(data_dir / "train.jsonl")
    test_data = load_data(data_dir / "test.jsonl")

    print(f"\nSynthetic data: {len(train_data)} train, {len(test_data)} test")

    # Load real GSM8K
    print("Loading real GSM8K test set...")
    gsm8k_test = load_gsm8k_test(100)
    print(f"GSM8K test: {len(gsm8k_test)} examples")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Baseline on GSM8K
    print("\n" + "=" * 70)
    print("BASELINE ON REAL GSM8K")
    print("=" * 70)
    model.freeze()
    baseline = evaluate(model, tokenizer, gsm8k_test[:20], show=3)
    print(f"Baseline: {baseline['correct']}/{baseline['total']}")

    # Unfreeze
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.model.layers[-3].unfreeze()
    model.model.layers[-4].unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print("TRAINING (10 epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    batch_size = 8

    for epoch in range(10):
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            loss = sft_batch(model, tokenizer, batch, sft_opt)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate on BOTH synthetic and GSM8K
        synth_results = evaluate(model, tokenizer, test_data[:50])
        gsm8k_results = evaluate(model, tokenizer, gsm8k_test[:50])

        synth_acc = synth_results['correct'] / synth_results['total']
        gsm8k_acc = gsm8k_results['correct'] / gsm8k_results['total']

        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}  synthetic={synth_acc:.0%}  gsm8k={gsm8k_acc:.0%}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nSynthetic test (full):")
    synth_final = evaluate(model, tokenizer, test_data, show=5)
    print(f"Synthetic: {synth_final['correct']}/{synth_final['total']} = {synth_final['correct']/synth_final['total']:.0%}")

    print("\nReal GSM8K test (full):")
    gsm8k_final = evaluate(model, tokenizer, gsm8k_test, show=8)
    print(f"GSM8K: {gsm8k_final['correct']}/{gsm8k_final['total']} = {gsm8k_final['correct']/gsm8k_final['total']:.0%}")
    print(f"  Parse fail: {gsm8k_final['parse_fail']}")
    print(f"  Wrong: {gsm8k_final['wrong']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Synthetic test: {synth_final['correct']/synth_final['total']:.0%}")
    print(f"  Real GSM8K:     {gsm8k_final['correct']/gsm8k_final['total']:.0%}")
    print(f"\n  Transfer gap: {synth_final['correct']/synth_final['total'] - gsm8k_final['correct']/gsm8k_final['total']:.0%}")


if __name__ == "__main__":
    main()
