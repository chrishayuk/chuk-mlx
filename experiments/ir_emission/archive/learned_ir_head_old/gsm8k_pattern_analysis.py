"""
GSM8K Pattern Analysis - Discover missing templates.

Analyze GSM8K problems to find:
1. What operation patterns exist
2. What text signals map to operations
3. What's missing from our template library
"""

import re
from collections import Counter, defaultdict
import functools
print = functools.partial(print, flush=True)


def load_gsm8k(n: int = 200):
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")

    data = []
    for item in ds:
        answer_text = item["answer"]
        match = re.search(r'####\s*(-?\d+)', answer_text)
        if match:
            data.append({
                "question": item["question"],
                "answer": int(match.group(1)),
                "solution": answer_text
            })
        if len(data) >= n:
            break

    return data


def extract_operations(solution: str) -> list[str]:
    """Extract operation sequence from GSM8K solution."""
    ops = []
    lines = solution.split('\n')

    for line in lines:
        if '####' in line:
            continue

        # Look for patterns like "5 + 3 = 8" or "5 * 3"
        if '+' in line and '=' in line:
            ops.append('add')
        elif '-' in line and '=' in line:
            ops.append('subtract')
        elif '*' in line and '=' in line:
            ops.append('multiply')
        elif '/' in line and '=' in line:
            ops.append('divide')

    return ops


def extract_text_signals(question: str) -> dict:
    """Extract text patterns that signal operations."""
    q = question.lower()

    signals = {
        "add": [],
        "subtract": [],
        "multiply": [],
        "divide": [],
        "compare": [],
        "fraction": [],
        "time": [],
        "rate": [],
        "unknown": [],
    }

    # Addition signals
    add_patterns = [
        (r'gets?\s+(\d+)\s+more', 'gets X more'),
        (r'receives?\s+(\d+)', 'receives X'),
        (r'adds?\s+(\d+)', 'adds X'),
        (r'buys?\s+(\d+)', 'buys X'),
        (r'finds?\s+(\d+)', 'finds X'),
        (r'(\d+)\s+more\s+than', 'X more than'),
        (r'plus\s+(\d+)', 'plus X'),
        (r'and\s+(\d+)\s+more', 'and X more'),
        (r'together', 'together'),
        (r'total', 'total'),
        (r'combined', 'combined'),
        (r'in all', 'in all'),
    ]

    # Subtraction signals
    sub_patterns = [
        (r'loses?\s+(\d+)', 'loses X'),
        (r'gives?\s+away\s+(\d+)', 'gives away X'),
        (r'uses?\s+(\d+)', 'uses X'),
        (r'eats?\s+(\d+)', 'eats X'),
        (r'spends?\s+(\d+)', 'spends X'),
        (r'(\d+)\s+less\s+than', 'X less than'),
        (r'(\d+)\s+fewer', 'X fewer'),
        (r'minus\s+(\d+)', 'minus X'),
        (r'left\s+over', 'left over'),
        (r'remaining', 'remaining'),
        (r'how many.+left', 'how many left'),
        (r'takes?\s+away', 'takes away'),
        (r'removes?', 'removes'),
    ]

    # Multiplication signals
    mul_patterns = [
        (r'(\d+)\s+times', 'X times'),
        (r'(\d+)\s+each', 'X each'),
        (r'(\d+)\s+per\s+', 'X per'),
        (r'(\d+)\s+rows?\s+of\s+(\d+)', 'X rows of Y'),
        (r'(\d+)\s+groups?\s+of\s+(\d+)', 'X groups of Y'),
        (r'(\d+)\s+boxes?\s+with\s+(\d+)', 'X boxes with Y'),
        (r'(\d+)\s+bags?\s+with\s+(\d+)', 'X bags with Y'),
        (r'twice', 'twice'),
        (r'triple', 'triple'),
        (r'double', 'double'),
    ]

    # Division signals
    div_patterns = [
        (r'divides?\s+.+\s+among\s+(\d+)', 'divides among X'),
        (r'splits?\s+.+\s+among\s+(\d+)', 'splits among X'),
        (r'shares?\s+equally', 'shares equally'),
        (r'(\d+)\s+equal\s+parts', 'X equal parts'),
        (r'half\s+of', 'half of'),
        (r'quarter', 'quarter'),
        (r'per\s+person', 'per person'),
        (r'each\s+person\s+gets?', 'each person gets'),
    ]

    # Comparison/ratio signals
    cmp_patterns = [
        (r'(\d+)\s+more\s+than', 'X more than'),
        (r'(\d+)\s+less\s+than', 'X less than'),
        (r'(\d+)\s+times\s+as\s+many', 'X times as many'),
        (r'twice\s+as\s+many', 'twice as many'),
        (r'half\s+as\s+many', 'half as many'),
        (r'same\s+number', 'same number'),
        (r'difference', 'difference'),
    ]

    # Fraction/percentage signals
    frac_patterns = [
        (r'(\d+)\s*%', 'X%'),
        (r'percent', 'percent'),
        (r'half', 'half'),
        (r'quarter', 'quarter'),
        (r'third', 'third'),
        (r'fraction', 'fraction'),
    ]

    # Time signals
    time_patterns = [
        (r'(\d+)\s+hours?', 'X hours'),
        (r'(\d+)\s+minutes?', 'X minutes'),
        (r'(\d+)\s+days?', 'X days'),
        (r'(\d+)\s+weeks?', 'X weeks'),
        (r'per\s+hour', 'per hour'),
        (r'per\s+day', 'per day'),
        (r'every\s+day', 'every day'),
    ]

    # Rate signals
    rate_patterns = [
        (r'(\d+)\s+per\s+hour', 'X per hour'),
        (r'(\d+)\s+per\s+day', 'X per day'),
        (r'\$(\d+)\s+per', '$X per'),
        (r'\$(\d+)\s+each', '$X each'),
        (r'rate\s+of', 'rate of'),
        (r'speed', 'speed'),
    ]

    for pattern, label in add_patterns:
        if re.search(pattern, q):
            signals["add"].append(label)

    for pattern, label in sub_patterns:
        if re.search(pattern, q):
            signals["subtract"].append(label)

    for pattern, label in mul_patterns:
        if re.search(pattern, q):
            signals["multiply"].append(label)

    for pattern, label in div_patterns:
        if re.search(pattern, q):
            signals["divide"].append(label)

    for pattern, label in cmp_patterns:
        if re.search(pattern, q):
            signals["compare"].append(label)

    for pattern, label in frac_patterns:
        if re.search(pattern, q):
            signals["fraction"].append(label)

    for pattern, label in time_patterns:
        if re.search(pattern, q):
            signals["time"].append(label)

    for pattern, label in rate_patterns:
        if re.search(pattern, q):
            signals["rate"].append(label)

    return signals


def count_steps(solution: str) -> int:
    """Count computation steps in solution."""
    lines = solution.split('\n')
    calc_lines = [l for l in lines if '=' in l and '####' not in l]
    return len(calc_lines)


def main():
    print("=" * 70)
    print("  GSM8K PATTERN ANALYSIS")
    print("  Discovering missing templates")
    print("=" * 70)

    # Load data
    print("\nLoading GSM8K...")
    data = load_gsm8k(200)
    print(f"Loaded {len(data)} examples")

    # Analyze step counts
    print("\n" + "=" * 70)
    print("STEP DISTRIBUTION")
    print("=" * 70)

    step_counts = Counter()
    for item in data:
        steps = count_steps(item["solution"])
        step_counts[steps] += 1

    for steps in sorted(step_counts.keys()):
        pct = step_counts[steps] / len(data) * 100
        bar = "█" * int(pct / 2)
        print(f"  {steps} steps: {step_counts[steps]:3d} ({pct:4.1f}%) {bar}")

    # Analyze text signals
    print("\n" + "=" * 70)
    print("TEXT SIGNAL FREQUENCY")
    print("=" * 70)

    signal_counts = defaultdict(Counter)

    for item in data:
        signals = extract_text_signals(item["question"])
        for op, labels in signals.items():
            for label in labels:
                signal_counts[op][label] += 1

    for op in ["add", "subtract", "multiply", "divide", "compare", "fraction", "time", "rate"]:
        if signal_counts[op]:
            print(f"\n{op.upper()} signals:")
            for label, count in signal_counts[op].most_common(10):
                pct = count / len(data) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")

    # Find problems with patterns NOT in our training
    print("\n" + "=" * 70)
    print("GAPS IN CURRENT TEMPLATE LIBRARY")
    print("=" * 70)

    # Our current patterns
    current_patterns = {
        "add": ["gets", "finds", "receives", "earns", "buys", "collects", "more"],
        "subtract": ["eats", "gives away", "loses", "uses", "spends", "breaks", "drops"],
        "multiply": ["each", "per", "rows of", "groups of", "boxes with", "bags with"],
        "divide": ["among", "equally", "split", "divide"],
    }

    # Missing patterns
    print("\nPatterns in GSM8K but NOT in training:")

    missing = defaultdict(list)

    for op, counts in signal_counts.items():
        if op in current_patterns:
            for label, count in counts.most_common():
                # Check if any current pattern matches
                is_covered = any(cp in label.lower() for cp in current_patterns[op])
                if not is_covered and count >= 3:
                    missing[op].append((label, count))

    for op, items in missing.items():
        if items:
            print(f"\n  {op.upper()}:")
            for label, count in items[:5]:
                print(f"    - '{label}' ({count} occurrences)")

    # Complex patterns not handled
    print("\nComplex patterns (need new template families):")

    complex_count = 0
    for item in data:
        signals = extract_text_signals(item["question"])
        if signals["compare"] or signals["fraction"]:
            complex_count += 1

    print(f"  - Comparisons (X more/less than Y): {sum(signal_counts['compare'].values())} signals")
    print(f"  - Fractions/percentages: {sum(signal_counts['fraction'].values())} signals")
    print(f"  - Time calculations: {sum(signal_counts['time'].values())} signals")
    print(f"  - Rate problems: {sum(signal_counts['rate'].values())} signals")

    # Long chains
    long_chains = sum(1 for item in data if count_steps(item["solution"]) > 3)
    print(f"  - Long chains (4+ steps): {long_chains} problems ({long_chains/len(data)*100:.0f}%)")

    # Sample problems by category
    print("\n" + "=" * 70)
    print("SAMPLE PROBLEMS BY GAP CATEGORY")
    print("=" * 70)

    # Comparison problems
    print("\n--- COMPARISON PROBLEMS ---")
    shown = 0
    for item in data:
        signals = extract_text_signals(item["question"])
        if signals["compare"] and shown < 3:
            print(f"\nQ: {item['question'][:150]}...")
            print(f"Signals: {signals['compare']}")
            print(f"Answer: {item['answer']}")
            shown += 1

    # Fraction problems
    print("\n--- FRACTION/PERCENTAGE PROBLEMS ---")
    shown = 0
    for item in data:
        signals = extract_text_signals(item["question"])
        if signals["fraction"] and shown < 3:
            print(f"\nQ: {item['question'][:150]}...")
            print(f"Signals: {signals['fraction']}")
            print(f"Answer: {item['answer']}")
            shown += 1

    # Long chain problems
    print("\n--- LONG CHAIN PROBLEMS (5+ steps) ---")
    shown = 0
    for item in data:
        if count_steps(item["solution"]) >= 5 and shown < 3:
            print(f"\nQ: {item['question'][:150]}...")
            print(f"Steps: {count_steps(item['solution'])}")
            print(f"Answer: {item['answer']}")
            shown += 1

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS - New templates to add")
    print("=" * 70)

    print("""
1. COMPARISON TEMPLATES
   - "X has Y more than Z" → need to track multiple entities
   - "twice as many" → multiply by 2
   - "half as many" → divide by 2

2. FRACTION/PERCENTAGE TEMPLATES
   - "X% of Y" → Y * X / 100
   - "half of X" → X / 2
   - "a third of X" → X / 3

3. TIME TEMPLATES
   - "X per hour for Y hours" → X * Y
   - "every day for X days" → daily_amount * X

4. RATE TEMPLATES
   - "$X per item, Y items" → X * Y
   - "earns $X per hour, works Y hours" → X * Y

5. MULTI-ENTITY TEMPLATES
   - Track multiple people/things
   - "A has X, B has Y, together..." → X + Y

6. LONGER CHAINS
   - Current: up to 3 steps
   - GSM8K: 43% have 4+ steps
   - Need: 4-6 step templates
""")


if __name__ == "__main__":
    main()
