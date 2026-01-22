"""Percentage problem generator."""

import random


def generate_percent_off():
    """X% off a price."""
    price = random.randint(20, 200)
    percent = random.choice([10, 15, 20, 25, 30, 40, 50])
    discount = price * percent / 100
    final = price - discount

    item = random.choice(["shirt", "book", "toy", "jacket", "bag"])

    question = f"A {item} costs ${price}. It's {percent}% off. What's the sale price?"

    trace = [
        {"init": "price", "value": price},
        {"percent_off": {"base": "price", "rate": percent, "var": "sale_price", "result": final}},
        {"query": "sale_price"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": final,
    }


def generate_percent_increase():
    """X% increase."""
    base = random.randint(50, 200)
    percent = random.choice([10, 15, 20, 25, 50])
    increase = base * percent / 100
    final = base + increase

    scenarios = [
        f"A stock worth ${base} increases by {percent}%. What's the new value?",
        f"Rent of ${base} goes up {percent}%. What's the new rent?",
        f"A salary of ${base} gets a {percent}% raise. What's the new salary?",
    ]

    question = random.choice(scenarios)

    trace = [
        {"init": "base", "value": base},
        {"compute": {"op": "mul", "args": ["base", percent / 100], "var": "increase", "result": increase}},
        {"compute": {"op": "add", "args": ["base", "increase"], "var": "final", "result": final}},
        {"query": "final"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": final,
    }


def generate_find_percent():
    """What percent is X of Y?"""
    whole = random.randint(50, 200)
    percent = random.choice([10, 20, 25, 40, 50, 75])
    part = whole * percent / 100

    question = f"What percent of {whole} is {int(part)}?"

    trace = [
        {"init": "part", "value": part},
        {"init": "whole", "value": whole},
        {"compute": {"op": "div", "args": ["part", "whole"], "var": "fraction", "result": part / whole}},
        {"compute": {"op": "mul", "args": ["fraction", 100], "var": "percent", "result": percent}},
        {"query": "percent"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": percent,
    }


def generate_tip_calculation():
    """Calculate tip on a bill."""
    bill = random.randint(20, 100)
    tip_percent = random.choice([15, 18, 20, 25])
    tip = bill * tip_percent / 100
    total = bill + tip

    question = f"Your bill is ${bill}. You want to leave a {tip_percent}% tip. What's the total including tip?"

    trace = [
        {"init": "bill", "value": bill},
        {"compute": {"op": "mul", "args": ["bill", tip_percent / 100], "var": "tip", "result": tip}},
        {"compute": {"op": "add", "args": ["bill", "tip"], "var": "total", "result": total}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": total,
    }


GENERATORS = [
    generate_percent_off,
    generate_percent_increase,
    generate_find_percent,
    generate_tip_calculation,
]


def generate(n: int = 15) -> list[dict]:
    """Generate n percentage examples."""
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
