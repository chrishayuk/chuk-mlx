"""Comparison problem generator."""

import random

NAMES = ["Alice", "Bob", "Carol", "Dan", "Emma", "Frank"]
ITEMS = ["stickers", "cards", "marbles", "coins", "books"]


def generate_times_more():
    """A has X times as many as B. How many more?"""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)
    base = random.randint(5, 20)
    multiplier = random.randint(2, 5)

    larger = base * multiplier
    difference = larger - base

    question = f"{name1} has {multiplier} times as many {item} as {name2}. {name2} has {base}. How many more does {name1} have than {name2}?"

    trace = [
        {"init": f"{name2.lower()}.{item}", "value": base},
        {"compute": {"op": "mul", "args": [base, multiplier], "var": f"{name1.lower()}.{item}", "result": larger}},
        {"compare": {"op": "sub", "args": [larger, base], "var": "difference", "result": difference}},
        {"query": "difference"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": difference,
    }


def generate_sum_and_difference():
    """Together they have X. A has Y more than B. How many does each have?"""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)

    # Work backwards: a = b + diff, a + b = total
    b = random.randint(10, 30)
    diff = random.randint(5, 15)
    a = b + diff
    total = a + b

    question = f"{name1} and {name2} have {total} {item} together. {name1} has {diff} more than {name2}. How many does {name1} have?"

    trace = [
        {"given": {"total": total, "difference": diff}},
        {"formula": f"{name1.lower()} = {name2.lower()} + {diff}"},
        {"formula": f"2*{name2.lower()} + {diff} = {total}"},
        {"compute": {"op": "sub", "args": [total, diff], "var": "twice_b", "result": total - diff}},
        {"compute": {"op": "div", "args": ["twice_b", 2], "var": f"{name2.lower()}.{item}", "result": b}},
        {"compute": {"op": "add", "args": [b, diff], "var": f"{name1.lower()}.{item}", "result": a}},
        {"query": f"{name1.lower()}.{item}"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": a,
    }


def generate_ratio_comparison():
    """Ratio comparison."""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)

    base = random.randint(3, 10)
    ratio1 = random.randint(2, 4)
    ratio2 = random.randint(1, 3)

    amount1 = base * ratio1
    amount2 = base * ratio2
    difference = abs(amount1 - amount2)

    question = f"The ratio of {name1}'s {item} to {name2}'s is {ratio1}:{ratio2}. If the smaller amount is {min(amount1, amount2)}, what's the difference?"

    trace = [
        {"given": {"ratio1": ratio1, "ratio2": ratio2}},
        {"init": "smaller", "value": min(amount1, amount2)},
        {"compute": {"op": "div", "args": ["smaller", min(ratio1, ratio2)], "var": "unit", "result": base}},
        {"compute": {"op": "mul", "args": ["unit", max(ratio1, ratio2)], "var": "larger", "result": max(amount1, amount2)}},
        {"compare": {"op": "sub", "args": [max(amount1, amount2), min(amount1, amount2)], "var": "difference", "result": difference}},
        {"query": "difference"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": difference,
    }


def generate_more_less():
    """A has X more than B. B has Y. Total?"""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)

    base = random.randint(10, 30)
    more = random.randint(5, 15)
    amount1 = base + more
    total = amount1 + base

    question = f"{name1} has {more} more {item} than {name2}. {name2} has {base}. How many do they have together?"

    trace = [
        {"init": f"{name2.lower()}.{item}", "value": base},
        {"compute": {"op": "add", "args": [base, more], "var": f"{name1.lower()}.{item}", "result": amount1}},
        {"compute": {"op": "add", "args": [amount1, base], "var": "total", "result": total}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": total,
    }


GENERATORS = [
    generate_times_more,
    generate_sum_and_difference,
    generate_ratio_comparison,
    generate_more_less,
]


def generate(n: int = 40) -> list[dict]:
    """Generate n comparison examples."""
    examples = []
    for _ in range(n):
        gen = random.choice(GENERATORS)
        examples.append(gen())
    return examples


if __name__ == "__main__":
    for ex in generate(3):
        print(f"Q: {ex['question']}")
        print(f"Answer: {ex['answer']}")
        print(f"Trace: {ex['trace']}")
        print()
