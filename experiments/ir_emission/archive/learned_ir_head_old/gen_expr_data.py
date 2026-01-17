"""
Generate diverse multi-step expression-only training data.

Focus on patterns that require chains:
- "loses X, loses Y" → subtract-subtract
- "X rows of Y, sells Z" → multiply-subtract
- "has X, uses Y, sells rest at $Z" → subtract-multiply
"""

import random
import json
from pathlib import Path

# Names and items for variety
NAMES = ["Tom", "Sam", "Lisa", "Emma", "Jake", "Anna", "Mike", "Kate", "Ben", "Zoe",
         "Maria", "Alex", "Chris", "Dan", "Amy", "Nick", "Sara", "Ryan", "Mia", "Evan"]

ITEMS = ["apple", "cookie", "book", "pencil", "sticker", "marble", "candy",
         "flower", "toy", "card", "egg", "orange", "coin", "dollar", "point",
         "ball", "rock", "stamp", "bead", "shell"]

CONTAINERS = ["box", "bag", "basket", "jar", "pack", "shelf", "row", "tray",
              "crate", "pile", "stack", "bunch", "set", "group"]

LOSS_VERBS = ["eats", "gives away", "loses", "uses", "spends", "drops", "breaks"]
GAIN_VERBS = ["gets", "finds", "receives", "earns", "buys", "collects", "picks up"]


def gen_single_subtract():
    """Single subtraction: has X, loses Y"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    verb = random.choice(LOSS_VERBS)

    a = random.randint(20, 100)
    b = random.randint(5, a - 5)

    templates = [
        f"{name} has {a} {item}s. {name} {verb} {b}. How many left?",
        f"{name} has {a} {item}s and {verb} {b}. Left?",
        f"Has {a} {item}s. {verb.capitalize()} {b}. How many remain?",
        f"Start with {a}. Lose {b}. Left?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n[END]",
        "ans": a - b
    }


def gen_single_multiply():
    """Single multiplication: X containers with Y each"""
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)

    a = random.randint(3, 12)
    b = random.randint(4, 15)

    templates = [
        f"{a} {container}s with {b} {item}s each. Total {item}s?",
        f"Has {a} {container}s. Each has {b} {item}s. Total?",
        f"{a} rows of {b}. Total?",
        f"{a} groups with {b} each. How many total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n[END]",
        "ans": a * b
    }


def gen_single_add():
    """Single addition: has X, gets Y more"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    verb = random.choice(GAIN_VERBS)

    a = random.randint(10, 50)
    b = random.randint(5, 40)

    templates = [
        f"{name} has {a} {item}s. {name} {verb} {b} more. Total?",
        f"Has {a}. Gets {b} more. Total?",
        f"Start with {a} {item}s and {verb} {b}. How many now?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} + {b} =\n[END]",
        "ans": a + b
    }


def gen_single_divide():
    """Single division: divide X among Y"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    b = random.randint(3, 10)
    result = random.randint(3, 15)
    a = b * result  # Ensure clean division

    templates = [
        f"Divide {a} {item}s among {b} people. Each gets?",
        f"{name} splits {a} {item}s equally among {b} friends. Each gets?",
        f"Share {a} items with {b} people. How many each?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} / {b} =\n[END]",
        "ans": a // b
    }


# =============================================================================
# MULTI-STEP PATTERNS (key focus)
# =============================================================================

def gen_subtract_subtract():
    """Two subtractions: has X, loses Y, loses Z"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    verb1 = random.choice(LOSS_VERBS)
    verb2 = random.choice(LOSS_VERBS)

    a = random.randint(50, 100)
    b = random.randint(5, 20)
    c = random.randint(5, min(20, a - b - 5))

    templates = [
        f"{name} has {a} {item}s. {verb1.capitalize()} {b}, then {verb2} {c}. Left?",
        f"Has {a}. Loses {b}, loses {c}. How many left?",
        f"Start with {a}. {verb1.capitalize()} {b}. Then {verb2} {c} more. Remaining?",
        f"{name} has {a} {item}s. {verb1.capitalize()} {b}. Later {verb2} {c}. Left?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ - {c} =\n[END]",
        "ans": a - b - c
    }


def gen_add_subtract():
    """Add then subtract: has X, gets Y, loses Z"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(20, 50)
    b = random.randint(10, 30)
    c = random.randint(5, a + b - 5)

    templates = [
        f"{name} has {a} {item}s. Gets {b} more. Then loses {c}. How many now?",
        f"Has {a}. Gains {b}. Spends {c}. Left?",
        f"Start with {a}. Add {b}. Remove {c}. Remaining?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} + {b} =\n_ - {c} =\n[END]",
        "ans": a + b - c
    }


def gen_multiply_subtract():
    """Multiply then subtract: X rows of Y, sells Z"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)

    a = random.randint(4, 10)
    b = random.randint(5, 12)
    total = a * b
    c = random.randint(5, total - 5)

    templates = [
        f"{a} {container}s with {b} {item}s each. Sells {c}. Left?",
        f"{name} has {a} rows of {b}. Removes {c}. Remaining?",
        f"{a} groups of {b}. Takes away {c}. How many left?",
        f"Has {a} {container}s with {b} each. Uses {c}. Left?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ - {c} =\n[END]",
        "ans": total - c
    }


def gen_subtract_multiply():
    """Subtract then multiply: has X, uses Y, sells rest at $Z each"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(30, 60)
    b = random.randint(5, 15)
    c = random.randint(2, 5)

    templates = [
        f"{name} has {a} {item}s. Uses {b}. Sells rest at ${c} each. Revenue?",
        f"Has {a}. Uses {b}. Sells remaining at ${c} each. Total earned?",
        f"Start with {a}. Remove {b}. Sell rest for ${c} each. How much?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ * {c} =\n[END]",
        "ans": (a - b) * c
    }


def gen_multiply_divide():
    """Multiply then divide: X packs of Y, split among Z people"""
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)

    a = random.randint(4, 8)
    b = random.randint(6, 12)
    total = a * b
    # Find divisor that works
    c = random.choice([d for d in range(2, 10) if total % d == 0])

    templates = [
        f"{a} {container}s with {b} {item}s each. Split among {c} people. Each gets?",
        f"Has {a} groups of {b}. Divides equally among {c}. How many each?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ / {c} =\n[END]",
        "ans": total // c
    }


def gen_subtract_subtract_multiply():
    """Three steps: has X, loses Y, loses Z, sells rest at $W"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(60, 100)
    b = random.randint(10, 20)
    c = random.randint(5, 15)
    d = random.randint(2, 4)

    templates = [
        f"{name} has {a} {item}s. Loses {b}. Then loses {c} more. Sells rest at ${d} each. Revenue?",
        f"Has {a}. Loses {b}, loses {c}. Sells remaining at ${d} each. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ - {c} =\n_ * {d} =\n[END]",
        "ans": (a - b - c) * d
    }


def gen_multiply_subtract_divide():
    """Three steps: X rows of Y, removes Z, splits among W"""
    item = random.choice(ITEMS)

    a = random.randint(5, 8)
    b = random.randint(8, 12)
    total = a * b
    c = random.randint(5, 15)
    remaining = total - c
    # Find divisor
    w = random.choice([d for d in range(2, 8) if remaining % d == 0] or [1])

    templates = [
        f"{a} rows of {b} {item}s. Removes {c}. Splits rest among {w} people. Each gets?",
        f"Has {a} groups of {b}. Takes {c} away. Divides remaining among {w}. How many each?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ - {c} =\n_ / {w} =\n[END]",
        "ans": remaining // w
    }


# =============================================================================
# GENERATOR
# =============================================================================

GENERATORS = {
    # Single step (40%)
    "single_subtract": (gen_single_subtract, 0.10),
    "single_multiply": (gen_single_multiply, 0.10),
    "single_add": (gen_single_add, 0.10),
    "single_divide": (gen_single_divide, 0.10),

    # Two-step (45%)
    "subtract_subtract": (gen_subtract_subtract, 0.12),
    "add_subtract": (gen_add_subtract, 0.08),
    "multiply_subtract": (gen_multiply_subtract, 0.10),
    "subtract_multiply": (gen_subtract_multiply, 0.10),
    "multiply_divide": (gen_multiply_divide, 0.05),

    # Three-step (15%)
    "subtract_subtract_multiply": (gen_subtract_subtract_multiply, 0.08),
    "multiply_subtract_divide": (gen_multiply_subtract_divide, 0.07),
}


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    """Generate n examples with weighted sampling."""
    random.seed(seed)

    generators = list(GENERATORS.keys())
    weights = [GENERATORS[g][1] for g in generators]

    data = []
    for _ in range(n):
        gen_name = random.choices(generators, weights=weights)[0]
        gen_fn = GENERATORS[gen_name][0]

        try:
            example = gen_fn()
            example["pattern"] = gen_name
            data.append(example)
        except Exception as e:
            print(f"Error in {gen_name}: {e}")
            continue

    return data


def save_jsonl(data: list[dict], path: str):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    # Generate datasets
    output_dir = Path(__file__).parent / "expr_data"
    output_dir.mkdir(exist_ok=True)

    train = generate_dataset(500, seed=42)
    val = generate_dataset(50, seed=123)
    test = generate_dataset(100, seed=456)

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    # Stats
    print(f"Generated:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

    # Pattern distribution
    from collections import Counter
    patterns = Counter(d["pattern"] for d in train)
    print(f"\nPattern distribution (train):")
    for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
        pct = c / len(train) * 100
        print(f"  {p}: {c} ({pct:.0f}%)")

    # Sample examples
    print("\nSample examples:")
    for item in random.sample(train, 5):
        print(f"\n  Q: {item['q']}")
        print(f"  Expr: {item['expr'].replace(chr(10), ' | ')}")
        print(f"  Ans: {item['ans']} ({item['pattern']})")
