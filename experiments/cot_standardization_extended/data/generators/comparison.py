"""Comparison problem generator - symbolic traces (no results)."""

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
        {"init": "multiplier", "value": multiplier},
        {"compute": {"op": "mul", "args": [f"{name2.lower()}.{item}", "multiplier"], "var": f"{name1.lower()}.{item}"}},
        {"compute": {"op": "sub", "args": [f"{name1.lower()}.{item}", f"{name2.lower()}.{item}"], "var": "difference"}},
        {"query": "difference"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": difference,
    }


def generate_sum_and_difference():
    """Together they have X. A has Y more than B. How many does A have?"""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)

    b = random.randint(10, 30)
    diff = random.randint(5, 15)
    a = b + diff
    total = a + b

    question = f"{name1} and {name2} have {total} {item} together. {name1} has {diff} more than {name2}. How many does {name1} have?"

    trace = [
        {"init": "total", "value": total},
        {"init": "difference", "value": diff},
        {"formula": f"{name1.lower()} + {name2.lower()} = total"},
        {"formula": f"{name1.lower()} = {name2.lower()} + difference"},
        {"compute": {"op": "sub", "args": ["total", "difference"], "var": "twice_b"}},
        {"compute": {"op": "div", "args": ["twice_b", 2], "var": f"{name2.lower()}.{item}"}},
        {"compute": {"op": "add", "args": [f"{name2.lower()}.{item}", "difference"], "var": f"{name1.lower()}.{item}"}},
        {"query": f"{name1.lower()}.{item}"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": a,
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
        {"init": "more", "value": more},
        {"compute": {"op": "add", "args": [f"{name2.lower()}.{item}", "more"], "var": f"{name1.lower()}.{item}"}},
        {"compute": {"op": "add", "args": [f"{name1.lower()}.{item}", f"{name2.lower()}.{item}"], "var": "total"}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": total,
    }


def generate_half_as_many():
    """A has half as many as B. B has X. How many does A have?"""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)

    base = random.randint(10, 40) * 2  # Ensure even
    half = base // 2

    question = f"{name1} has half as many {item} as {name2}. {name2} has {base}. How many does {name1} have?"

    trace = [
        {"init": f"{name2.lower()}.{item}", "value": base},
        {"compute": {"op": "div", "args": [f"{name2.lower()}.{item}", 2], "var": f"{name1.lower()}.{item}"}},
        {"query": f"{name1.lower()}.{item}"},
    ]

    return {
        "question": question,
        "expert": "comparison",
        "trace": trace,
        "answer": half,
    }


GENERATORS = [
    generate_times_more,
    generate_sum_and_difference,
    generate_more_less,
    generate_half_as_many,
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
