"""Arithmetic chain problem generator - symbolic traces (no results)."""

import random


def generate_price_chain():
    """Price + tax + shipping pattern."""
    base = random.randint(10, 100)
    tax = round(random.uniform(1, 10), 2)
    shipping = random.randint(2, 10)

    total = round(base + tax + shipping, 2)
    item = random.choice(["toy", "book", "shirt", "gadget", "tool"])

    question = f"A {item} costs ${base}. Tax adds ${tax}. Shipping is ${shipping}. What's the total?"

    # Symbolic: no results, solver computes
    trace = [
        {"init": "price", "value": base},
        {"init": "tax", "value": tax},
        {"init": "shipping", "value": shipping},
        {"compute": {"op": "add", "args": ["price", "tax"], "var": "with_tax"}},
        {"compute": {"op": "add", "args": ["with_tax", "shipping"], "var": "total"}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "arithmetic",
        "trace": trace,
        "answer": total,
    }


def generate_subtract_chain():
    """Start with amount, subtract multiple times."""
    start = random.randint(50, 200)
    sub1 = random.randint(5, 30)
    sub2 = random.randint(5, 30)
    sub3 = random.randint(5, 20)

    final = start - sub1 - sub2 - sub3

    question = f"You have ${start}. You spend ${sub1} on lunch, ${sub2} on a ticket, and ${sub3} on snacks. How much do you have left?"

    trace = [
        {"init": "money", "value": start},
        {"init": "lunch", "value": sub1},
        {"init": "ticket", "value": sub2},
        {"init": "snacks", "value": sub3},
        {"compute": {"op": "sub", "args": ["money", "lunch"], "var": "after_lunch"}},
        {"compute": {"op": "sub", "args": ["after_lunch", "ticket"], "var": "after_ticket"}},
        {"compute": {"op": "sub", "args": ["after_ticket", "snacks"], "var": "remaining"}},
        {"query": "remaining"},
    ]

    return {
        "question": question,
        "expert": "arithmetic",
        "trace": trace,
        "answer": final,
    }


def generate_multiply_add():
    """Multiply then add."""
    count = random.randint(3, 10)
    price = random.randint(5, 20)
    extra = random.randint(5, 20)

    total = count * price + extra
    item = random.choice(["pens", "notebooks", "bottles", "bags"])

    question = f"You buy {count} {item} at ${price} each and pay ${extra} for gift wrapping. What's the total?"

    trace = [
        {"init": "count", "value": count},
        {"init": "price", "value": price},
        {"init": "wrapping", "value": extra},
        {"compute": {"op": "mul", "args": ["count", "price"], "var": "subtotal"}},
        {"compute": {"op": "add", "args": ["subtotal", "wrapping"], "var": "total"}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "arithmetic",
        "trace": trace,
        "answer": total,
    }


def generate_divide_multiply():
    """Divide then multiply."""
    total = random.randint(20, 100)
    divisor = random.choice([2, 4, 5, 10])
    multiplier = random.randint(2, 5)

    per_item = total // divisor
    final = per_item * multiplier

    if multiplier == 3:
        mult_text = "triples"
    elif multiplier == 2:
        mult_text = "doubles"
    else:
        mult_text = f"multiplies by {multiplier}"

    question = f"You split ${total} equally among {divisor} people. Each person {mult_text} their share. How much does each have?"

    trace = [
        {"init": "total", "value": total},
        {"init": "people", "value": divisor},
        {"init": "multiplier", "value": multiplier},
        {"compute": {"op": "div", "args": ["total", "people"], "var": "per_person"}},
        {"compute": {"op": "mul", "args": ["per_person", "multiplier"], "var": "final"}},
        {"query": "final"},
    ]

    return {
        "question": question,
        "expert": "arithmetic",
        "trace": trace,
        "answer": final,
    }


GENERATORS = [
    generate_price_chain,
    generate_subtract_chain,
    generate_multiply_add,
    generate_divide_multiply,
]


def generate(n: int = 40) -> list[dict]:
    """Generate n arithmetic examples."""
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
