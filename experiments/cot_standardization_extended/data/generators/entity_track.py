"""Entity tracking problem generator."""

import random

NAMES = ["Alice", "Bob", "Carol", "Dan", "Emma", "Frank", "Grace", "Henry"]
ITEMS = ["apples", "marbles", "books", "coins", "cards", "stickers", "pencils", "cookies"]
GIVE_VERBS = ["gives", "hands", "passes", "transfers"]
LOSE_VERBS = ["loses", "drops", "misplaces"]
FIND_VERBS = ["finds", "discovers", "picks up"]
EAT_VERBS = ["eats", "consumes", "uses"]


def generate_simple_transfer():
    """A gives B some items."""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)
    initial = random.randint(10, 50)
    transfer = random.randint(1, initial - 1)
    verb = random.choice(GIVE_VERBS)

    question = f"{name1} has {initial} {item}. {name1} {verb} {transfer} to {name2}. How many {item} does {name1} have?"

    trace = [
        {"init": f"{name1.lower()}.{item}", "value": initial},
        {"init": f"{name2.lower()}.{item}", "value": 0},
        {"transfer": {"from": f"{name1.lower()}.{item}", "to": f"{name2.lower()}.{item}", "amount": transfer}},
        {"state": {f"{name1.lower()}.{item}": initial - transfer}},
        {"query": f"{name1.lower()}.{item}"},
    ]

    return {
        "question": question,
        "expert": "entity_track",
        "trace": trace,
        "answer": initial - transfer,
    }


def generate_consume_sequence():
    """Entity consumes items multiple times."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    initial = random.randint(15, 40)
    consume1 = random.randint(1, initial // 3)
    consume2 = random.randint(1, initial // 3)
    verb1 = random.choice(EAT_VERBS)
    verb2 = random.choice(EAT_VERBS)

    remaining = initial - consume1 - consume2

    question = f"{name} has {initial} {item}. {name} {verb1} {consume1} and then {verb2} {consume2}. How many {item} does {name} have left?"

    trace = [
        {"init": item, "value": initial},
        {"consume": {"entity": item, "amount": consume1}},
        {"state": {item: initial - consume1}},
        {"consume": {"entity": item, "amount": consume2}},
        {"state": {item: remaining}},
        {"query": item},
    ]

    return {
        "question": question,
        "expert": "entity_track",
        "trace": trace,
        "answer": remaining,
    }


def generate_consume_then_multiply():
    """Classic GSM-8K pattern: consume then multiply remaining."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    initial = random.randint(10, 30)
    consume1 = random.randint(1, initial // 4)
    consume2 = random.randint(1, initial // 4)
    multiplier = random.randint(2, 5)

    remaining = initial - consume1 - consume2
    final = remaining * multiplier

    verb1 = random.choice(EAT_VERBS)
    verb2 = random.choice(EAT_VERBS)

    question = f"{name} has {initial} {item}. {name} {verb1} {consume1} and {verb2} {consume2}. {name} sells the rest for ${multiplier} each. How much money does {name} make?"

    trace = [
        {"init": item, "value": initial},
        {"consume": {"entity": item, "amount": consume1}},
        {"consume": {"entity": item, "amount": consume2}},
        {"state": {item: remaining}},
        {"compute": {"op": "mul", "args": [remaining, multiplier], "var": "revenue", "result": final}},
        {"query": "revenue"},
    ]

    return {
        "question": question,
        "expert": "entity_track",
        "trace": trace,
        "answer": final,
    }


def generate_bidirectional_transfer():
    """A gives to B, B gives back some."""
    name1, name2 = random.sample(NAMES, 2)
    item = random.choice(ITEMS)
    initial1 = random.randint(20, 40)
    initial2 = random.randint(5, 15)
    transfer1 = random.randint(5, 15)
    transfer2 = random.randint(1, 5)

    final1 = initial1 - transfer1 + transfer2

    question = f"{name1} has {initial1} {item} and {name2} has {initial2}. {name1} gives {transfer1} to {name2}. Then {name2} gives {transfer2} back. How many does {name1} have?"

    trace = [
        {"init": f"{name1.lower()}.{item}", "value": initial1},
        {"init": f"{name2.lower()}.{item}", "value": initial2},
        {"transfer": {"from": f"{name1.lower()}.{item}", "to": f"{name2.lower()}.{item}", "amount": transfer1}},
        {"transfer": {"from": f"{name2.lower()}.{item}", "to": f"{name1.lower()}.{item}", "amount": transfer2}},
        {"state": {f"{name1.lower()}.{item}": final1}},
        {"query": f"{name1.lower()}.{item}"},
    ]

    return {
        "question": question,
        "expert": "entity_track",
        "trace": trace,
        "answer": final1,
    }


def generate_find_and_lose():
    """Entity finds and loses items."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    initial = random.randint(10, 30)
    found = random.randint(2, 10)
    lost = random.randint(1, 5)

    final = initial + found - lost

    question = f"{name} has {initial} {item}. {name} {random.choice(FIND_VERBS)} {found} more, then {random.choice(LOSE_VERBS)} {lost}. How many does {name} have now?"

    trace = [
        {"init": item, "value": initial},
        {"compute": {"op": "add", "args": [initial, found], "var": item, "result": initial + found}},
        {"consume": {"entity": item, "amount": lost}},
        {"state": {item: final}},
        {"query": item},
    ]

    return {
        "question": question,
        "expert": "entity_track",
        "trace": trace,
        "answer": final,
    }


GENERATORS = [
    generate_simple_transfer,
    generate_consume_sequence,
    generate_consume_then_multiply,
    generate_bidirectional_transfer,
    generate_find_and_lose,
]


def generate(n: int = 100) -> list[dict]:
    """Generate n entity tracking examples."""
    examples = []
    for _ in range(n):
        gen = random.choice(GENERATORS)
        examples.append(gen())
    return examples


if __name__ == "__main__":
    for ex in generate(3):
        print(f"Q: {ex['question']}")
        print(f"Expert: {ex['expert']}")
        print(f"Answer: {ex['answer']}")
        print(f"Trace: {ex['trace']}")
        print()
