"""
Test trained model on real GSM8K problems.

Evaluates both expression-only and named-variable formats.
"""

import sys
from pathlib import Path
import random
import re
import json

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# GSM8K LOADING
# =============================================================================

def load_gsm8k(n: int = 20):
    """Load GSM8K test examples."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")

    data = []
    for item in ds:
        match = re.search(r'####\s*(-?\d+)', item["answer"])
        if match:
            data.append({
                "question": item["question"],
                "answer": int(match.group(1)),
                "solution": item["answer"]
            })
        if len(data) >= n:
            break

    return data


# =============================================================================
# EXECUTORS
# =============================================================================

def execute_expr_only(code: str) -> tuple[int | None, str]:
    """Execute expression-only format (with _ chaining)."""
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty"

    current = None

    for line in lines:
        match = re.match(r'(\d+|_)\s*([+\-*/])\s*(\d+)\s*=', line)
        if not match:
            return None, "parse_fail"

        left, op, right = match.groups()
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
            return None, "bad_op"

        current = result

    return current, "ok"


def execute_named(code: str) -> tuple[int | None, str]:
    """Execute named variable format."""
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty"

    variables = {}
    last_result = None

    for line in lines:
        match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not match:
            return None, "parse_fail"

        var_name, expr = match.groups()
        expr = expr.strip()

        expr_match = re.match(r'(\w+)\s*([+\-*/])\s*(\w+)', expr)
        if not expr_match:
            if expr.isdigit():
                variables[var_name] = int(expr)
                last_result = int(expr)
                continue
            return None, "expr_fail"

        left, op, right = expr_match.groups()

        try:
            left_val = int(left) if left.isdigit() else variables[left]
            right_val = int(right) if right.isdigit() else variables[right]
        except KeyError:
            return None, "undefined_var"

        if op == '+':
            result = left_val + right_val
        elif op == '-':
            result = left_val - right_val
        elif op == '*':
            result = left_val * right_val
        elif op == '/':
            if right_val == 0:
                return None, "div_zero"
            result = left_val // right_val
        else:
            return None, "bad_op"

        variables[var_name] = result
        last_result = result

    return last_result, "ok"


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(q: str) -> str:
    return f"Q: {q}\nA:\n"


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
# TRAINING (quick SFT)
# =============================================================================

def load_train_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def quick_sft(model, tokenizer, data: list[dict], epochs: int = 8):
    """Quick SFT training."""
    print(f"  Training on {len(data)} examples, {epochs} epochs...")

    optimizer = optim.Adam(learning_rate=2e-5)

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

        return (ce * masks).sum() / (masks.sum() + 1e-8)

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0

        for item in data:
            # Handle both formats
            expr_key = "expr" if "expr" in item else "chain"
            full_text = f"Q: {item['q']}\nA:\n{item[expr_key]}"
            prompt = f"Q: {item['q']}\nA:\n"

            full_tokens = tokenizer.encode(full_text)[:200]
            prompt_len = len(tokenizer.encode(prompt))

            mask = [0] * prompt_len + [1] * (len(full_tokens) - prompt_len)

            tokens = mx.array([full_tokens])
            masks = mx.array([mask], dtype=mx.float32)

            loss, grads = loss_and_grad(model, tokens, masks)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()

        print(f"    Epoch {epoch+1}: loss={total_loss/len(data):.4f}")

    return model


# =============================================================================
# GSM8K EVALUATION
# =============================================================================

def evaluate_gsm8k(model, tokenizer, gsm8k_data: list, executor_fn, format_name: str):
    """Evaluate on GSM8K with specified executor."""
    print(f"\n--- {format_name} Format ---")

    correct = 0
    parse_fail = 0
    wrong = 0

    results = []

    for item in gsm8k_data:
        prompt = format_prompt(item["question"])
        output = greedy_generate(model, tokenizer, prompt)

        result, reason = executor_fn(output)

        if result is None:
            parse_fail += 1
            status = f"PARSE_FAIL: {reason}"
        elif result == item["answer"]:
            correct += 1
            status = "CORRECT"
        else:
            wrong += 1
            status = f"WRONG: got {result}, expected {item['answer']}"

        results.append({
            "question": item["question"][:80] + "...",
            "expected": item["answer"],
            "output": output[:100],
            "result": result,
            "status": status
        })

    # Summary
    total = len(gsm8k_data)
    print(f"\n  Results:")
    print(f"    Correct:    {correct}/{total} = {correct/total:.0%}")
    print(f"    Parse fail: {parse_fail}/{total} = {parse_fail/total:.0%}")
    print(f"    Wrong:      {wrong}/{total} = {wrong/total:.0%}")

    # Show samples
    print(f"\n  Sample outputs:")
    for r in results[:5]:
        print(f"\n    Q: {r['question']}")
        print(f"    Expected: {r['expected']}")
        print(f"    Output: {r['output'][:60]}...")
        print(f"    Status: {r['status']}")

    return {"correct": correct, "parse_fail": parse_fail, "wrong": wrong, "total": total}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GSM8K EVALUATION")
    print("  Testing trained models on real math word problems")
    print("=" * 70)

    # Load GSM8K
    print("\nLoading GSM8K (20 examples)...")
    gsm8k = load_gsm8k(20)
    print(f"Loaded {len(gsm8k)} examples")

    # Show GSM8K samples
    print("\nGSM8K sample problems:")
    for item in gsm8k[:3]:
        print(f"\n  Q: {item['question'][:100]}...")
        print(f"  A: {item['answer']}")

    # Load model
    print("\n" + "=" * 70)
    print("Loading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Test 1: Expression-only format
    print("\n" + "=" * 70)
    print("TEST 1: EXPRESSION-ONLY FORMAT (v2 data)")
    print("=" * 70)

    expr_data_path = Path(__file__).parent / "expr_data_v2" / "train.jsonl"
    if expr_data_path.exists():
        model.freeze()
        model.model.layers[-1].unfreeze()
        model.model.layers[-2].unfreeze()
        model.model.layers[-3].unfreeze()
        model.lm_head.unfreeze()

        expr_data = load_train_data(expr_data_path)
        model = quick_sft(model, tokenizer, expr_data[:500], epochs=6)

        expr_results = evaluate_gsm8k(model, tokenizer, gsm8k, execute_expr_only, "Expression-Only")
    else:
        print("  No expr_data_v2 found, skipping...")
        expr_results = None

    # Test 2: Named variable format
    print("\n" + "=" * 70)
    print("TEST 2: NAMED VARIABLE FORMAT")
    print("=" * 70)

    # Reload model fresh
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    named_data_path = Path(__file__).parent / "named_var_data" / "train.jsonl"
    if named_data_path.exists():
        model.freeze()
        model.model.layers[-1].unfreeze()
        model.model.layers[-2].unfreeze()
        model.model.layers[-3].unfreeze()
        model.lm_head.unfreeze()

        named_data = load_train_data(named_data_path)
        model = quick_sft(model, tokenizer, named_data[:500], epochs=6)

        named_results = evaluate_gsm8k(model, tokenizer, gsm8k, execute_named, "Named Variable")
    else:
        print("  No named_var_data found, skipping...")
        named_results = None

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if expr_results:
        print(f"\n  Expression-only: {expr_results['correct']}/{expr_results['total']} = {expr_results['correct']/expr_results['total']:.0%}")
    if named_results:
        print(f"  Named variable:  {named_results['correct']}/{named_results['total']} = {named_results['correct']/named_results['total']:.0%}")

    print("\n  Key insight: GSM8K problems often require multi-variable tracking")
    print("  that expression-only format can't handle.")


if __name__ == "__main__":
    main()
