"""
CoT → IR: Chain-of-Thought as Semantic Rewriting to IR

Key insight from our experiments:
- Format learning (IR emission) is easy: 95-99% accuracy
- Semantic mapping is hard: 0-7% on GSM8K
- The "derivation stage" IS what CoT does

The synthesis: CoT rewrites complex linguistic relationships into
operand-operator form, but outputs executable IR instead of natural language.

This is NOT:
  Q: "5 fewer than 3 times as many"
  CoT: "3 times as many means multiply. 5 fewer means subtract."
  A: "3 * x - 5 = 15... so x = ..."  (natural language math)

This IS:
  Q: "5 fewer than 3 times as many eggs as yesterday. Yesterday had 10."
  CoT: "base = 10. 3 times means multiply: 10 * 3 = 30. 5 fewer: 30 - 5 = 25"
  IR: step1 = 10 * 3 | step2 = step1 - 5 | [END]
  Execute: step1=30, step2=25
  Answer: 25

The CoT does semantic binding (what numbers go where), IR provides verifiable execution.
"""

import re
import json
import random
from pathlib import Path
from collections import Counter

import functools
print = functools.partial(print, flush=True)


# =============================================================================
# IR EXECUTION ENGINE
# =============================================================================

def execute_ir(ir_code: str) -> dict:
    """Execute IR and return all step values."""
    env = {}
    steps = ir_code.strip().split('|')

    for step in steps:
        step = step.strip()
        if step == '[END]' or not step:
            continue

        if '=' not in step:
            continue

        var, expr = step.split('=', 1)
        var = var.strip()
        expr = expr.strip()

        # Substitute previous steps
        for prev_var, prev_val in env.items():
            expr = expr.replace(prev_var, str(prev_val))

        try:
            result = eval(expr)
            env[var] = result
        except:
            env[var] = None

    return env


def get_final_answer(ir_code: str) -> float:
    """Execute IR and return final answer."""
    env = execute_ir(ir_code)
    if not env:
        return None
    # Return last step value
    return list(env.values())[-1]


# =============================================================================
# COT REWRITE PATTERNS
# =============================================================================

# These patterns show what semantic transformations CoT must learn
COT_PATTERNS = {
    # Direct references
    "direct": {
        "example": "Tom has 5 apples. He eats 2.",
        "cot": "Start with 5. Eats means subtract 2.",
        "ir": "step1 = 5 - 2 | [END]",
    },

    # Multiplicative relationships
    "times_as_many": {
        "example": "Sara has 3 times as many books as Tom. Tom has 5 books.",
        "cot": "Tom has 5. 3 times as many means multiply: 5 * 3.",
        "ir": "step1 = 5 * 3 | [END]",
    },

    # Comparative relationships
    "fewer_than": {
        "example": "Mike has 5 fewer apples than Sara. Sara has 12 apples.",
        "cot": "Sara has 12. 5 fewer means subtract: 12 - 5.",
        "ir": "step1 = 12 - 5 | [END]",
    },

    # Combined relationships
    "times_then_fewer": {
        "example": "Emma has 5 fewer than 3 times as many eggs as yesterday. Yesterday: 10.",
        "cot": "Yesterday: 10. 3 times: 10 * 3 = 30. 5 fewer: 30 - 5.",
        "ir": "step1 = 10 * 3 | step2 = step1 - 5 | [END]",
    },

    # Rate problems
    "rate_per": {
        "example": "John earns $15 per hour. He works 8 hours.",
        "cot": "Rate: $15/hour. Hours: 8. Total: 15 * 8.",
        "ir": "step1 = 15 * 8 | [END]",
    },

    # Division/sharing
    "split_among": {
        "example": "24 cookies split among 6 kids.",
        "cot": "Total: 24. Kids: 6. Each gets: 24 / 6.",
        "ir": "step1 = 24 / 6 | [END]",
    },

    # Multi-step
    "produce_consume_sell": {
        "example": "Ducks lay 16 eggs/day. Janet eats 3 and bakes with 4. Sells rest at $2 each.",
        "cot": "Produced: 16. Eats: 3. Bakes: 4. Remaining: 16 - 3 - 4 = 9. Sells at $2: 9 * 2.",
        "ir": "step1 = 16 - 3 | step2 = step1 - 4 | step3 = step2 * 2 | [END]",
    },
}


# =============================================================================
# GSM8K COT→IR EXAMPLES
# =============================================================================

def load_gsm8k_cot_examples():
    """Load GSM8K and extract CoT→IR aligned examples."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")

    examples = []

    for item in ds:
        question = item["question"]
        answer = item["answer"]

        # Extract computations (these form the IR)
        computations = re.findall(r'<<([^>]+)>>', answer)
        if not computations:
            continue

        # Build IR from computations
        ir_steps = []
        for i, comp in enumerate(computations, 1):
            if '=' in comp:
                expr, result = comp.rsplit('=', 1)
                ir_steps.append(f"step{i} = {expr.strip()}")

        if not ir_steps:
            continue

        ir_code = " | ".join(ir_steps) + " | [END]"

        # Extract the reasoning text (CoT)
        # Remove the <<...>> annotations to get pure reasoning
        cot_text = re.sub(r'<<[^>]+>>', '', answer)
        cot_text = re.sub(r'####.*', '', cot_text).strip()

        # Extract final answer
        final_match = re.search(r'####\s*(-?[\d,]+)', answer)
        if not final_match:
            continue
        final_answer = int(final_match.group(1).replace(",", ""))

        # Verify IR execution matches
        computed = get_final_answer(ir_code)
        if computed is not None and abs(computed - final_answer) < 0.01:
            examples.append({
                'question': question,
                'cot': cot_text,
                'ir': ir_code,
                'answer': final_answer,
                'n_steps': len(ir_steps),
            })

    return examples


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_cot_to_ir_alignment(examples: list):
    """Analyze what CoT patterns map to what IR patterns."""

    print(f"\nAnalyzing {len(examples)} CoT→IR examples...")

    # Step count distribution
    step_counts = Counter(ex['n_steps'] for ex in examples)
    print("\nSteps distribution:")
    for n, count in sorted(step_counts.items()):
        print(f"  {n} steps: {count} ({count/len(examples):.1%})")

    # Operation patterns in IR
    op_patterns = Counter()
    for ex in examples:
        ir = ex['ir']
        ops = []
        if '+' in ir: ops.append('ADD')
        if '-' in ir: ops.append('SUB')
        if '*' in ir: ops.append('MUL')
        if '/' in ir: ops.append('DIV')
        op_patterns[tuple(sorted(ops))] += 1

    print("\nOperation combinations:")
    for pattern, count in op_patterns.most_common(10):
        print(f"  {pattern}: {count}")

    # Sample some 1-step and 2-step examples
    print("\n" + "=" * 70)
    print("SAMPLE 1-STEP EXAMPLES")
    print("=" * 70)

    one_step = [ex for ex in examples if ex['n_steps'] == 1][:5]
    for ex in one_step:
        print(f"\n  Q: {ex['question'][:80]}...")
        print(f"  CoT: {ex['cot'][:80]}...")
        print(f"  IR: {ex['ir']}")
        print(f"  Answer: {ex['answer']}")

    print("\n" + "=" * 70)
    print("SAMPLE 2-STEP EXAMPLES")
    print("=" * 70)

    two_step = [ex for ex in examples if ex['n_steps'] == 2][:5]
    for ex in two_step:
        print(f"\n  Q: {ex['question'][:80]}...")
        print(f"  CoT: {ex['cot'][:80]}...")
        print(f"  IR: {ex['ir']}")
        print(f"  Answer: {ex['answer']}")


def demonstrate_cot_ir_paradigm():
    """Demonstrate the CoT→IR paradigm with examples."""

    print("=" * 70)
    print("  COT→IR PARADIGM DEMONSTRATION")
    print("  CoT does semantic binding, IR provides verifiable execution")
    print("=" * 70)

    for name, pattern in COT_PATTERNS.items():
        print(f"\n{'─' * 50}")
        print(f"Pattern: {name}")
        print(f"{'─' * 50}")
        print(f"  Question: {pattern['example']}")
        print(f"  CoT:      {pattern['cot']}")
        print(f"  IR:       {pattern['ir']}")

        # Execute
        result = get_final_answer(pattern['ir'])
        print(f"  Execute:  {result}")


# =============================================================================
# TRAINING DATA FORMAT
# =============================================================================

def create_cot_ir_training_format(examples: list) -> list:
    """
    Create training data in CoT→IR format.

    Format:
      Input:  <question>
      Output: <cot_reasoning> [IR] <ir_code> [ANSWER] <number>

    The [IR] token signals transition from reasoning to executable code.
    The [ANSWER] token marks the final numeric answer.
    """

    formatted = []

    for ex in examples:
        # Clean up CoT (remove extra whitespace)
        cot = ' '.join(ex['cot'].split())

        formatted.append({
            'input': ex['question'],
            'output': f"{cot} [IR] {ex['ir']} [ANSWER] {ex['answer']}",
            'cot': cot,
            'ir': ex['ir'],
            'answer': ex['answer'],
        })

    return formatted


def verify_training_format(examples: list, n: int = 5):
    """Verify and display training format examples."""

    print("\n" + "=" * 70)
    print("TRAINING FORMAT EXAMPLES")
    print("=" * 70)

    for ex in examples[:n]:
        print(f"\n{'─' * 50}")
        print(f"INPUT:  {ex['input'][:100]}...")
        print(f"OUTPUT: {ex['output'][:150]}...")

        # Verify execution
        ir_match = re.search(r'\[IR\]\s*(.+?)\s*\[ANSWER\]', ex['output'])
        if ir_match:
            ir = ir_match.group(1)
            computed = get_final_answer(ir)
            print(f"VERIFY: IR executes to {computed}, expected {ex['answer']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)

    print("=" * 70)
    print("  COT → IR: Chain-of-Thought as Semantic Rewriting")
    print("=" * 70)

    print("""
Key insight from experiments:
  - Format learning (IR emission): 95-99% accuracy ✓
  - Semantic mapping: 0-7% on GSM8K ✗
  - The gap: Complex linguistic relationships need transformation

The solution:
  - CoT does semantic binding (what numbers go where)
  - IR provides verifiable execution (machine-checkable)
  - Reward = IR execution matches expected answer
""")

    # Demonstrate the paradigm
    demonstrate_cot_ir_paradigm()

    # Load GSM8K examples
    print("\n" + "=" * 70)
    print("LOADING GSM8K COT→IR EXAMPLES")
    print("=" * 70)

    examples = load_gsm8k_cot_examples()
    print(f"\nLoaded {len(examples)} verified CoT→IR examples")

    # Analyze
    analyze_cot_to_ir_alignment(examples)

    # Create training format
    print("\n" + "=" * 70)
    print("CREATING TRAINING DATA")
    print("=" * 70)

    formatted = create_cot_ir_training_format(examples)
    verify_training_format(formatted)

    # Save
    output_dir = Path(__file__).parent / "cot_ir_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split train/test
    random.shuffle(formatted)
    split = int(0.9 * len(formatted))
    train = formatted[:split]
    test = formatted[split:]

    for name, data in [("train", train), ("test", test)]:
        path = output_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(data)} examples to {path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Total examples: {len(examples)}
  Train: {len(train)}
  Test: {len(test)}

  Format:
    Input:  <question>
    Output: <cot> [IR] <ir_code> [ANSWER] <number>

  Training approach:
    1. Model generates CoT reasoning + IR code
    2. IR code is executed deterministically
    3. Reward = 1 if execution matches answer, 0 otherwise
    4. RL optimizes for correct semantic binding

  Why this works:
    - CoT provides flexibility for complex language
    - IR provides verifiability for reward signal
    - The model learns to translate semantics → operations
""")


if __name__ == "__main__":
    main()
