"""
GSM8K Diagnostic - Test expression-only model on real math word problems.

Track:
- Parse success rate
- Execution accuracy
- Failure patterns (what text didn't map?)

Each failure category → new template family.
"""

import sys
from pathlib import Path
import random
import re
import json
from collections import Counter, defaultdict

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# EXPRESSION EXECUTION (same as training)
# =============================================================================

def execute_chain(chain_text: str) -> tuple[int | None, str]:
    """Execute expression chain."""
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


# =============================================================================
# GSM8K LOADING
# =============================================================================

def load_gsm8k(split: str = "test", n: int = 100) -> list[dict]:
    """Load GSM8K from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=split)

        data = []
        for item in ds:
            # Extract numeric answer from "#### 42" format
            answer_text = item["answer"]
            match = re.search(r'####\s*(-?\d+)', answer_text)
            if match:
                answer = int(match.group(1))
                data.append({
                    "question": item["question"],
                    "answer": answer,
                    "full_solution": answer_text
                })

        # Sample n examples
        if len(data) > n:
            random.seed(42)
            data = random.sample(data, n)

        return data
    except ImportError:
        print("Installing datasets...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset
        return load_gsm8k(split, n)


def extract_gsm8k_steps(solution: str) -> int:
    """Count reasoning steps in GSM8K solution."""
    # Count lines with calculations
    lines = solution.split('\n')
    calc_lines = [l for l in lines if any(op in l for op in ['=', '+', '-', '*', '/'])]
    return len(calc_lines)


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:\n"


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 80) -> str:
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
# SFT TRAINING (quick, on synthetic data first)
# =============================================================================

def load_synthetic_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def sft_train(model, tokenizer, data: list[dict], epochs: int = 8):
    """Quick SFT training."""
    print(f"  SFT: {len(data)} examples, {epochs} epochs")

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
            full_text = f"Q: {item['q']}\nA:\n{item['expr']}"
            prompt = f"Q: {item['q']}\nA:\n"

            full_tokens = tokenizer.encode(full_text)[:100]
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
# FAILURE ANALYSIS
# =============================================================================

def analyze_failure(question: str, output: str, expected: int) -> dict:
    """Categorize failure type."""
    result, reason = execute_chain(output)

    analysis = {
        "parse_success": result is not None,
        "correct": result == expected if result else False,
        "result": result,
        "expected": expected,
        "reason": reason,
        "output": output,
    }

    # Analyze question patterns
    q_lower = question.lower()

    # Count expected steps (heuristic)
    step_keywords = ["then", "after", "now", "finally", "next", "later"]
    analysis["estimated_steps"] = 1 + sum(1 for kw in step_keywords if kw in q_lower)

    # Detect patterns not in our training
    missing_patterns = []

    # Comparison/conditionals
    if any(w in q_lower for w in ["more than", "less than", "twice", "half", "triple"]):
        missing_patterns.append("comparison")

    # Fractions/percentages
    if any(w in q_lower for w in ["percent", "%", "fraction", "half of", "quarter"]):
        missing_patterns.append("fractions")

    # Time-based
    if any(w in q_lower for w in ["hour", "minute", "day", "week", "month", "year"]):
        missing_patterns.append("time")

    # Rate problems
    if any(w in q_lower for w in ["per hour", "per day", "speed", "rate"]):
        missing_patterns.append("rates")

    # Multi-entity
    if len(re.findall(r'\b[A-Z][a-z]+\b', question)) > 2:
        missing_patterns.append("multi_entity")

    # Long chains
    if analysis["estimated_steps"] > 3:
        missing_patterns.append("long_chain")

    analysis["missing_patterns"] = missing_patterns

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GSM8K DIAGNOSTIC")
    print("  Testing expression-only model on real math problems")
    print("=" * 70)

    # Load GSM8K
    print("\nLoading GSM8K test set...")
    gsm8k = load_gsm8k("test", n=100)
    print(f"Loaded {len(gsm8k)} examples")

    # Analyze GSM8K complexity
    step_counts = [extract_gsm8k_steps(item["full_solution"]) for item in gsm8k]
    print(f"\nGSM8K step distribution:")
    for steps in sorted(set(step_counts)):
        count = step_counts.count(steps)
        print(f"  {steps} steps: {count} examples")

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    # Train on synthetic data first
    print("\n" + "=" * 70)
    print("PHASE 1: Train on synthetic data")
    print("=" * 70)

    data_dir = Path(__file__).parent / "expr_data"
    if data_dir.exists():
        train_data = load_synthetic_data(data_dir / "train.jsonl")

        model.freeze()
        model.model.layers[-1].unfreeze()
        model.model.layers[-2].unfreeze()
        model.model.layers[-3].unfreeze()
        model.lm_head.unfreeze()

        model = sft_train(model, tokenizer, train_data, epochs=5)
    else:
        print("No synthetic data found, using base model")

    # Test on GSM8K
    print("\n" + "=" * 70)
    print("PHASE 2: Test on GSM8K")
    print("=" * 70)

    results = []
    parse_success = 0
    correct = 0

    failure_reasons = Counter()
    missing_pattern_counts = Counter()

    for i, item in enumerate(gsm8k):
        prompt = format_prompt(item["question"])
        output = greedy_generate(model, tokenizer, prompt)

        analysis = analyze_failure(item["question"], output, item["answer"])
        results.append(analysis)

        if analysis["parse_success"]:
            parse_success += 1
        if analysis["correct"]:
            correct += 1
        else:
            failure_reasons[analysis["reason"]] += 1
            for pattern in analysis["missing_patterns"]:
                missing_pattern_counts[pattern] += 1

        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(gsm8k)}: parse={parse_success}/{i+1}, correct={correct}/{i+1}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nOverall:")
    print(f"  Parse success: {parse_success}/{len(gsm8k)} = {parse_success/len(gsm8k):.0%}")
    print(f"  Correct: {correct}/{len(gsm8k)} = {correct/len(gsm8k):.0%}")

    print(f"\nFailure reasons:")
    for reason, count in failure_reasons.most_common(10):
        print(f"  {reason}: {count}")

    print(f"\nMissing pattern categories (in failures):")
    for pattern, count in missing_pattern_counts.most_common(10):
        print(f"  {pattern}: {count}")

    # Sample failures by category
    print("\n" + "=" * 70)
    print("SAMPLE FAILURES")
    print("=" * 70)

    # Group by failure type
    failures_by_type = defaultdict(list)
    for r in results:
        if not r["correct"]:
            key = r["reason"]
            if r["missing_patterns"]:
                key = f"{r['reason']}:{r['missing_patterns'][0]}"
            failures_by_type[key].append(r)

    shown = 0
    for failure_type, failures in sorted(failures_by_type.items(), key=lambda x: -len(x[1])):
        if shown >= 10:
            break

        sample = failures[0]
        idx = results.index(sample)
        item = gsm8k[idx]

        print(f"\n[{failure_type}] ({len(failures)} total)")
        print(f"  Q: {item['question'][:100]}...")
        print(f"  Expected: {sample['expected']}")
        print(f"  Output: {sample['output'][:60]}...")
        print(f"  Steps in solution: {extract_gsm8k_steps(item['full_solution'])}")

        shown += 1

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS - New template families needed:")
    print("=" * 70)

    if missing_pattern_counts["long_chain"] > 5:
        print(f"  - Long chains (4+ steps): {missing_pattern_counts['long_chain']} failures")
    if missing_pattern_counts["comparison"] > 3:
        print(f"  - Comparisons (twice, half, more than): {missing_pattern_counts['comparison']} failures")
    if missing_pattern_counts["fractions"] > 3:
        print(f"  - Fractions/percentages: {missing_pattern_counts['fractions']} failures")
    if missing_pattern_counts["time"] > 3:
        print(f"  - Time calculations: {missing_pattern_counts['time']} failures")
    if missing_pattern_counts["rates"] > 3:
        print(f"  - Rate problems: {missing_pattern_counts['rates']} failures")
    if missing_pattern_counts["multi_entity"] > 3:
        print(f"  - Multi-entity problems: {missing_pattern_counts['multi_entity']} failures")

    if failure_reasons["parse_fail"] > 20:
        print(f"  - General parsing issues: {failure_reasons['parse_fail']} failures")
        print("    (Model outputs NL instead of expressions)")


if __name__ == "__main__":
    main()
