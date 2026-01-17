"""
Chain CoT Generator - Synthetic Training Data for Verifiable Math.

Format:
  16 - 3 = 13
  _ - 4 = 9
  _ * 2 = 18

Where _ = previous result. Each line verifiable independently.

Pattern Library:
  Text Pattern → Operation → Chain Template
"""

import random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# CHAIN FORMAT
# =============================================================================

@dataclass
class ChainStep:
    """Single step in computation chain."""
    op: str          # +, -, *, /
    operand: int     # right operand
    result: int      # computed result
    is_first: bool   # True if first step (no _ prefix)
    first_operand: int = None  # Only for first step


def format_chain(steps: list[ChainStep], add_end: bool = True) -> str:
    """Format chain as verifiable text with END marker."""
    lines = []
    for step in steps:
        if step.is_first:
            lines.append(f"{step.first_operand} {step.op} {step.operand} = {step.result}")
        else:
            lines.append(f"_ {step.op} {step.operand} = {step.result}")
    result = "\n".join(lines)
    if add_end:
        result += "\n[END]"
    return result


def compute_chain(initial: int, operations: list[tuple[str, int]]) -> list[ChainStep]:
    """Compute chain from initial value and operations."""
    steps = []
    current = initial

    for i, (op, operand) in enumerate(operations):
        if op == '+':
            result = current + operand
        elif op == '-':
            result = current - operand
        elif op == '*':
            result = current * operand
        elif op == '/':
            result = current // operand
        else:
            raise ValueError(f"Unknown op: {op}")

        if i == 0:
            steps.append(ChainStep(op=op, operand=operand, result=result,
                                   is_first=True, first_operand=current))
        else:
            steps.append(ChainStep(op=op, operand=operand, result=result, is_first=False))

        current = result

    return steps


# =============================================================================
# TEMPLATE LIBRARY
# =============================================================================

# Names for variety
NAMES = ["Tom", "Sarah", "Mike", "Lisa", "Emma", "Jake", "Anna", "Sam",
         "Maria", "Alex", "Jenny", "Chris", "Kate", "Ben", "Zoe", "Dan"]

# Nouns
ITEMS = ["apple", "cookie", "book", "pencil", "sticker", "marble", "candy",
         "flower", "toy", "card", "egg", "orange", "dollar", "point"]

CONTAINERS = ["box", "bag", "basket", "jar", "pack", "shelf", "row", "tray"]

LOSS_VERBS = ["eats", "gives away", "loses", "uses", "spends", "breaks"]
GAIN_VERBS = ["gets", "finds", "receives", "earns", "buys", "collects"]


# =============================================================================
# SINGLE OPERATION TEMPLATES
# =============================================================================

def gen_multiply_rate():
    """X items at $Y each → X * Y"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    qty = random.randint(2, 12)
    price = random.randint(2, 15)

    templates = [
        f"{person} buys {qty} {item}s at ${price} each. Total cost?",
        f"Each {item} costs ${price}. {person} buys {qty}. How much total?",
        f"{qty} {item}s at ${price} each. Total?",
    ]

    question = random.choice(templates)
    steps = compute_chain(qty, [('*', price)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_multiply_groups():
    """X containers with Y items each → X * Y"""
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)
    groups = random.randint(2, 10)
    per_group = random.randint(3, 15)

    templates = [
        f"There are {groups} {container}s with {per_group} {item}s each. Total {item}s?",
        f"A store has {groups} {container}s. Each has {per_group} {item}s. How many {item}s?",
        f"{groups} {container}s with {per_group} {item}s in each. Total?",
    ]

    question = random.choice(templates)
    steps = compute_chain(groups, [('*', per_group)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_subtract_loss():
    """Has X, loses Y → X - Y"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(15, 100)
    lose = random.randint(3, have - 5)
    verb = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb} {lose}. How many left?",
        f"{person} has {have} {item}s and {verb} {lose}. How many remain?",
        f"Start with {have} {item}s. {lose} are gone. How many left?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('-', lose)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_add_gain():
    """Has X, gets Y → X + Y"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(5, 50)
    gain = random.randint(3, 30)
    verb = random.choice(GAIN_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb} {gain} more. Total?",
        f"{person} has {have} {item}s and {verb} {gain} more. How many now?",
        f"Start with {have}. Add {gain}. Total?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('+', gain)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_divide_split():
    """Split X among Y → X / Y"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    divisor = random.randint(2, 10)
    total = divisor * random.randint(3, 15)  # ensure clean division

    templates = [
        f"Split {total} {item}s among {divisor} friends. Each gets?",
        f"{total} {item}s shared equally by {divisor} people. How many each?",
        f"Divide {total} {item}s into {divisor} equal groups. Per group?",
    ]

    question = random.choice(templates)
    steps = compute_chain(total, [('/', divisor)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


# =============================================================================
# TWO-STEP CHAIN TEMPLATES
# =============================================================================

def gen_subtract_subtract():
    """Has X, loses Y, then loses Z → chain of subtracts"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(30, 100)
    lose1 = random.randint(5, have // 3)
    lose2 = random.randint(3, (have - lose1) // 2)
    verb1 = random.choice(LOSS_VERBS)
    verb2 = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb1} {lose1} and then {verb2} {lose2}. How many left?",
        f"{person} has {have} {item}s. First {verb1} {lose1}. Then {verb2} {lose2}. Remaining?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('-', lose1), ('-', lose2)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_add_subtract():
    """Has X, gains Y, loses Z → add then subtract"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(10, 50)
    gain = random.randint(5, 30)
    lose = random.randint(3, have + gain - 5)
    gain_verb = random.choice(GAIN_VERBS)
    loss_verb = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {gain_verb} {gain} more, then {loss_verb} {lose}. How many now?",
        f"{person} starts with {have} {item}s. Gets {gain} more. Then {loss_verb} {lose}. Total?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('+', gain), ('-', lose)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_multiply_subtract():
    """Groups × per_group, then lose some → multiply then subtract"""
    person = random.choice(NAMES)
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)
    groups = random.randint(3, 10)
    per_group = random.randint(4, 12)
    total = groups * per_group
    lose = random.randint(3, total // 2)

    templates = [
        f"{person} has {groups} {container}s with {per_group} {item}s each. {person} uses {lose}. How many left?",
        f"There are {groups} {container}s of {per_group} {item}s. {lose} are sold. Remaining?",
    ]

    question = random.choice(templates)
    steps = compute_chain(groups, [('*', per_group), ('-', lose)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_subtract_multiply():
    """Has X, loses Y, sells rest at $Z each → subtract then multiply"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(15, 50)
    lose = random.randint(3, have // 2)
    price = random.randint(2, 10)
    verb = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb} {lose}. Sells the rest for ${price} each. Total money?",
        f"Start with {have} {item}s. Use {lose}. Sell remaining at ${price} each. Revenue?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('-', lose), ('*', price)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


# =============================================================================
# THREE-STEP CHAIN TEMPLATES
# =============================================================================

def gen_subtract_subtract_multiply():
    """Janet's ducks pattern: has X, loses Y, loses Z, sells at $W"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(15, 40)
    lose1 = random.randint(2, have // 4)
    lose2 = random.randint(2, have // 4)
    remaining = have - lose1 - lose2
    price = random.randint(2, 8)
    verb1 = random.choice(LOSS_VERBS)
    verb2 = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb1} {lose1} and {verb2} {lose2}. Sells the rest for ${price} each. Total money?",
        f"{person}'s farm produces {have} {item}s daily. {person} {verb1} {lose1} for breakfast and {verb2} {lose2} for baking. Sells the rest at ${price} each. Revenue?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('-', lose1), ('-', lose2), ('*', price)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_multiply_subtract_divide():
    """Groups of items, use some, split rest"""
    person = random.choice(NAMES)
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)
    groups = random.randint(4, 10)
    per_group = random.randint(5, 12)
    total = groups * per_group
    use = random.randint(5, total // 3)
    remaining = total - use
    # ensure clean division
    split = random.choice([d for d in [2, 3, 4, 5, 6] if remaining % d == 0] or [1])

    templates = [
        f"{person} has {groups} {container}s with {per_group} {item}s each. Uses {use}, then splits the rest among {split} friends. Each friend gets?",
    ]

    question = random.choice(templates)
    steps = compute_chain(groups, [('*', per_group), ('-', use), ('/', split)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


def gen_add_add_subtract():
    """Multiple gains then a loss"""
    person = random.choice(NAMES)
    item = random.choice(ITEMS)
    have = random.randint(10, 30)
    gain1 = random.randint(5, 20)
    gain2 = random.randint(5, 20)
    total = have + gain1 + gain2
    lose = random.randint(5, total // 2)
    verb1 = random.choice(GAIN_VERBS)
    verb2 = random.choice(GAIN_VERBS)
    verb3 = random.choice(LOSS_VERBS)

    templates = [
        f"{person} has {have} {item}s. {person} {verb1} {gain1} more, then {verb2} {gain2} more, then {verb3} {lose}. How many now?",
    ]

    question = random.choice(templates)
    steps = compute_chain(have, [('+', gain1), ('+', gain2), ('-', lose)])

    return {"q": question, "chain": format_chain(steps), "ans": steps[-1].result}


# =============================================================================
# GENERATOR REGISTRY
# =============================================================================

GENERATORS = {
    # Single ops (40%)
    "multiply_rate": gen_multiply_rate,
    "multiply_groups": gen_multiply_groups,
    "subtract_loss": gen_subtract_loss,
    "add_gain": gen_add_gain,
    "divide_split": gen_divide_split,

    # Two-step chains (40%)
    "subtract_subtract": gen_subtract_subtract,
    "add_subtract": gen_add_subtract,
    "multiply_subtract": gen_multiply_subtract,
    "subtract_multiply": gen_subtract_multiply,

    # Three-step chains (20%)
    "subtract_subtract_multiply": gen_subtract_subtract_multiply,
    "multiply_subtract_divide": gen_multiply_subtract_divide,
    "add_add_subtract": gen_add_add_subtract,
}

WEIGHTS = {
    "multiply_rate": 8,
    "multiply_groups": 8,
    "subtract_loss": 8,
    "add_gain": 8,
    "divide_split": 8,
    "subtract_subtract": 10,
    "add_subtract": 10,
    "multiply_subtract": 10,
    "subtract_multiply": 10,
    "subtract_subtract_multiply": 7,
    "multiply_subtract_divide": 7,
    "add_add_subtract": 6,
}


def generate_problem() -> dict:
    """Generate a single problem with chain format answer."""
    pattern = random.choices(
        list(GENERATORS.keys()),
        weights=[WEIGHTS[k] for k in GENERATORS.keys()]
    )[0]

    gen = GENERATORS[pattern]
    problem = gen()
    problem["pattern"] = pattern

    return problem


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    """Generate n problems."""
    random.seed(seed)
    return [generate_problem() for _ in range(n)]


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_chain(chain_text: str) -> tuple[bool, list[dict]]:
    """
    Verify each step in a chain.

    Returns: (all_correct, step_details)
    """
    lines = chain_text.strip().split('\n')
    steps = []
    prev_result = None
    all_correct = True

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip END marker
        if line == '[END]' or line.startswith('[END]'):
            continue

        # Parse: "X op Y = Z" or "_ op Y = Z"
        parts = line.replace('=', ' ').split()

        if len(parts) < 4:
            steps.append({"line": line, "valid": False, "error": "parse_fail"})
            all_correct = False
            continue

        try:
            if parts[0] == '_':
                if prev_result is None:
                    steps.append({"line": line, "valid": False, "error": "no_prev"})
                    all_correct = False
                    continue
                left = prev_result
            else:
                left = int(parts[0])

            op = parts[1]
            right = int(parts[2])
            claimed = int(parts[3])

            if op == '+':
                actual = left + right
            elif op == '-':
                actual = left - right
            elif op == '*':
                actual = left * right
            elif op == '/':
                if right == 0:
                    steps.append({"line": line, "valid": False, "error": "div_zero"})
                    all_correct = False
                    continue
                actual = left // right
            else:
                steps.append({"line": line, "valid": False, "error": f"unknown_op:{op}"})
                all_correct = False
                continue

            correct = actual == claimed
            steps.append({
                "line": line,
                "valid": True,
                "left": left,
                "op": op,
                "right": right,
                "claimed": claimed,
                "actual": actual,
                "correct": correct
            })

            if not correct:
                all_correct = False

            prev_result = claimed  # Use claimed for next step (matches training format)

        except (ValueError, IndexError) as e:
            steps.append({"line": line, "valid": False, "error": str(e)})
            all_correct = False

    return all_correct, steps


def compute_reward(chain_text: str, expected_answer: int) -> tuple[float, str]:
    """
    Compute reward for chain output.

    1.0 = all steps correct, final answer correct
    0.7 = all steps correct, wrong final
    0.5 = some steps correct, right final (lucky)
    0.3 = some steps correct, wrong final
    0.0 = parse failure or all wrong
    """
    all_correct, steps = verify_chain(chain_text)

    if not steps:
        return 0.0, "empty"

    valid_steps = [s for s in steps if s.get("valid", False)]
    correct_steps = [s for s in valid_steps if s.get("correct", False)]

    if not valid_steps:
        return 0.0, "parse_fail"

    # Get final result
    final_result = steps[-1].get("claimed") if steps[-1].get("valid") else None
    final_correct = final_result == expected_answer

    if all_correct and final_correct:
        return 1.0, "perfect"

    if all_correct and not final_correct:
        return 0.7, f"steps_ok_wrong_final:{final_result}"

    if final_correct:
        return 0.5, f"partial:{len(correct_steps)}/{len(valid_steps)}"

    if correct_steps:
        return 0.3, f"partial:{len(correct_steps)}/{len(valid_steps)}"

    return 0.1, "all_wrong"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  CHAIN COT GENERATOR")
    print("  Format: X op Y = Z  or  _ op Y = Z")
    print("=" * 70)

    # Generate samples
    print("\nGenerating 20 sample problems...\n")
    problems = generate_dataset(20, seed=42)

    # Show by pattern
    by_pattern = {}
    for p in problems:
        pattern = p["pattern"]
        if pattern not in by_pattern:
            by_pattern[pattern] = []
        by_pattern[pattern].append(p)

    for pattern, examples in by_pattern.items():
        print(f"\n{'='*60}")
        print(f"Pattern: {pattern}")
        print('='*60)

        for ex in examples[:2]:  # Show 2 per pattern
            print(f"\nQ: {ex['q']}")
            print(f"A:\n{ex['chain']}")
            print(f"Answer: {ex['ans']}")

            # Verify
            all_ok, details = verify_chain(ex['chain'])
            reward, reason = compute_reward(ex['chain'], ex['ans'])
            print(f"Verified: {all_ok}, Reward: {reward} ({reason})")

    # Save dataset
    output_dir = Path(__file__).parent / "chain_data"
    output_dir.mkdir(exist_ok=True)

    train = generate_dataset(1000, seed=42)
    val = generate_dataset(100, seed=123)
    test = generate_dataset(100, seed=456)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\nSaved {path}: {len(data)} examples")

    # Stats
    print("\n" + "=" * 70)
    print("  PATTERN DISTRIBUTION (train)")
    print("=" * 70)

    pattern_counts = {}
    for p in train:
        pattern = p["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:30} {count:4} ({count/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()
