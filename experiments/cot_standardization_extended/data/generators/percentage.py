"""Percentage problem generator - symbolic traces (no results)."""

import random


def generate_percent_off():
    """X% off a price."""
    price = random.randint(20, 200)
    percent = random.choice([10, 15, 20, 25, 30, 40, 50])
    final = price * (100 - percent) / 100

    item = random.choice(["shirt", "book", "toy", "jacket", "bag"])

    question = f"A {item} costs ${price}. It's {percent}% off. What's the sale price?"

    # Symbolic: model outputs structure, solver computes
    trace = [
        {"init": "price", "value": price},
        {"init": "discount_rate", "value": percent},
        {"percent_off": {"base": "price", "rate": "discount_rate", "var": "sale_price"}},
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
    final = base * (100 + percent) / 100

    scenarios = [
        f"A stock worth ${base} increases by {percent}%. What's the new value?",
        f"Rent of ${base} goes up {percent}%. What's the new rent?",
        f"A salary of ${base} gets a {percent}% raise. What's the new salary?",
    ]

    question = random.choice(scenarios)

    trace = [
        {"init": "base", "value": base},
        {"init": "increase_rate", "value": percent},
        {"percent_increase": {"base": "base", "rate": "increase_rate", "var": "final"}},
        {"query": "final"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": final,
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
        {"init": "tip_rate", "value": tip_percent},
        {"compute": {"op": "mul", "args": ["bill", "tip_rate"], "var": "tip_times_100"}},
        {"compute": {"op": "div", "args": ["tip_times_100", 100], "var": "tip"}},
        {"compute": {"op": "add", "args": ["bill", "tip"], "var": "total"}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": total,
    }


def generate_simple_percent():
    """What is X% of Y?"""
    whole = random.randint(50, 200)
    percent = random.choice([10, 20, 25, 50, 75])
    part = whole * percent / 100

    question = f"What is {percent}% of {whole}?"

    trace = [
        {"init": "whole", "value": whole},
        {"init": "percent", "value": percent},
        {"compute": {"op": "mul", "args": ["whole", "percent"], "var": "times_100"}},
        {"compute": {"op": "div", "args": ["times_100", 100], "var": "result"}},
        {"query": "result"},
    ]

    return {
        "question": question,
        "expert": "percentage",
        "trace": trace,
        "answer": part,
    }


GENERATORS = [
    generate_percent_off,
    generate_percent_increase,
    generate_tip_calculation,
    generate_simple_percent,
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
