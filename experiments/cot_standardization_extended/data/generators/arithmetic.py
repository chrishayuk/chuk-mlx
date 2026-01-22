"""Arithmetic chain problem generator."""

import random


def generate_price_chain():
    """Price + tax + shipping pattern."""
    base = random.randint(10, 100)
    tax = round(random.uniform(1, 10), 2)
    shipping = random.randint(2, 10)

    with_tax = round(base + tax, 2)
    total = round(with_tax + shipping, 2)

    item = random.choice(["toy", "book", "shirt", "gadget", "tool"])

    question = f"A {item} costs ${base}. Tax adds ${tax}. Shipping is ${shipping}. What's the total?"

    trace = [
        {"init": "price", "value": base},
        {"compute": {"op": "add", "args": ["price", tax], "var": "with_tax", "result": with_tax}},
        {"compute": {"op": "add", "args": ["with_tax", shipping], "var": "total", "result": total}},
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

    step1 = start - sub1
    step2 = step1 - sub2
    final = step2 - sub3

    question = f"You have ${start}. You spend ${sub1} on lunch, ${sub2} on a ticket, and ${sub3} on snacks. How much do you have left?"

    trace = [
        {"init": "money", "value": start},
        {"compute": {"op": "sub", "args": ["money", sub1], "var": "after_lunch", "result": step1}},
        {"compute": {"op": "sub", "args": ["after_lunch", sub2], "var": "after_ticket", "result": step2}},
        {"compute": {"op": "sub", "args": ["after_ticket", sub3], "var": "remaining", "result": final}},
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

    subtotal = count * price
    total = subtotal + extra

    item = random.choice(["pens", "notebooks", "bottles", "bags"])

    question = f"You buy {count} {item} at ${price} each and pay ${extra} for gift wrapping. What's the total?"

    trace = [
        {"init": "count", "value": count},
        {"init": "price", "value": price},
        {"compute": {"op": "mul", "args": ["count", "price"], "var": "subtotal", "result": subtotal}},
        {"compute": {"op": "add", "args": ["subtotal", extra], "var": "total", "result": total}},
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

    question = f"You split ${total} equally among {divisor} people. Each person then triples their share. How much does each have?"

    if multiplier == 3:
        mult_text = "triples"
    elif multiplier == 2:
        mult_text = "doubles"
    else:
        mult_text = f"multiplies by {multiplier}"

    question = f"You split ${total} equally among {divisor} people. Each person {mult_text} their share. How much does each have?"

    trace = [
        {"init": "total", "value": total},
        {"compute": {"op": "div", "args": ["total", divisor], "var": "per_person", "result": per_item}},
        {"compute": {"op": "mul", "args": ["per_person", multiplier], "var": "final", "result": final}},
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
