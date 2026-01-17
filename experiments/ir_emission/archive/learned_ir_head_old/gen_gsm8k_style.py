"""
Generate GSM8K-style training data.

Based on mined patterns:
- MULTIPLY: "each", "per", "every", "times", "twice"
- DIVIDE: "half", "third", "equally", "split", "among"
- SUBTRACT: "remaining", "takes", "spends", "eats", "uses", "gives"
- ADD: "more", "gets", "additional", "finds", "earns"

Key insight: Full phrase patterns matter, not individual words.
"""

import random
import json
from pathlib import Path

import functools
print = functools.partial(print, flush=True)


# Names (GSM8K style)
NAMES = ["Janet", "Mark", "Julie", "James", "Natalia", "Betty", "Randy",
         "Weng", "Albert", "Josh", "Emma", "John", "Sarah", "Mike", "Lisa",
         "Tom", "Mary", "David", "Amy", "Chris"]

ITEMS = ["apples", "oranges", "books", "pencils", "cookies", "eggs", "pages",
         "stickers", "marbles", "candies", "flowers", "cards", "toys", "balls",
         "clips", "pens", "shirts", "bottles", "boxes", "bags"]

CONTAINERS = ["boxes", "bags", "packs", "cartons", "crates", "baskets", "cases"]

ACTIVITIES = ["reading", "walking", "working", "studying", "playing", "running"]

MEALS = ["breakfast", "lunch", "dinner", "snacks"]

PURPOSES = ["baking", "crafts", "gifts", "the party", "decorations", "school"]


def gen_multiply_each():
    """Pattern: "{person} has {n} {containers}. Each {container} has {m} {items}."
    Answer: n * m
    """
    person = random.choice(NAMES)
    n = random.randint(3, 15)
    m = random.randint(2, 12)
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)

    q = f"{person} has {n} {container}. Each {container[:-1] if container.endswith('s') else container} has {m} {item}. How many {item} does {person} have in total?"
    ir = f"step1 = {n}*{m}\n[END]"
    ans = n * m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "multiply_each"}


def gen_multiply_per():
    """Pattern: "{person} earns ${rate} per hour and works {hours} hours."
    Answer: rate * hours
    """
    person = random.choice(NAMES)
    rate = random.randint(8, 25)
    hours = random.randint(3, 10)

    q = f"{person} earns ${rate} per hour. If {person.lower()} works {hours} hours, how much does {person.lower()} earn?"
    ir = f"step1 = {rate}*{hours}\n[END]"
    ans = rate * hours

    return {"q": q, "ir": ir, "ans": ans, "pattern": "multiply_per"}


def gen_multiply_times():
    """Pattern: "{person} does something {n} times a {period}. Each time uses {m}."
    Answer: n * m
    """
    person = random.choice(NAMES)
    n = random.randint(2, 7)
    m = random.randint(2, 10)
    period = random.choice(["day", "week", "month"])
    item = random.choice(ITEMS)

    q = f"{person} buys {m} {item} {n} times a {period}. How many {item} does {person} buy in a {period}?"
    ir = f"step1 = {m}*{n}\n[END]"
    ans = m * n

    return {"q": q, "ir": ir, "ans": ans, "pattern": "multiply_times"}


def gen_multiply_twice():
    """Pattern: "{person} has {n}. {other_person} has twice as many."
    Answer: n * 2
    """
    person = random.choice(NAMES)
    other = random.choice([n for n in NAMES if n != person])
    n = random.randint(5, 30)
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {other} has twice as many {item}. How many {item} does {other} have?"
    ir = f"step1 = {n}*2\n[END]"
    ans = n * 2

    return {"q": q, "ir": ir, "ans": ans, "pattern": "multiply_twice"}


def gen_divide_half():
    """Pattern: "{person} has {n}. Gives half to {other}."
    Answer: n / 2
    """
    person = random.choice(NAMES)
    other = random.choice([n for n in NAMES if n != person])
    n = random.randint(10, 50) * 2  # ensure even
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {person} gives half of them to {other}. How many {item} does {other} receive?"
    ir = f"step1 = {n}/2\n[END]"
    ans = n // 2

    return {"q": q, "ir": ir, "ans": ans, "pattern": "divide_half"}


def gen_divide_equally():
    """Pattern: "{person} has {n} {items} to share equally among {m} friends."
    Answer: n / m
    """
    person = random.choice(NAMES)
    m = random.randint(2, 8)
    n = random.randint(2, 10) * m  # ensure divisible
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item} to share equally among {m} friends. How many {item} does each friend get?"
    ir = f"step1 = {n}/{m}\n[END]"
    ans = n // m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "divide_equally"}


def gen_divide_split():
    """Pattern: "{n} {items} are split into {m} groups."
    Answer: n / m
    """
    m = random.randint(2, 6)
    n = random.randint(2, 8) * m  # ensure divisible
    item = random.choice(ITEMS)

    q = f"There are {n} {item}. They are split into {m} equal groups. How many {item} are in each group?"
    ir = f"step1 = {n}/{m}\n[END]"
    ans = n // m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "divide_split"}


def gen_subtract_eats():
    """Pattern: "{person} has {n} {items}. Eats {m} for {meal}."
    Answer: n - m
    """
    person = random.choice(NAMES)
    n = random.randint(10, 40)
    m = random.randint(2, min(8, n-1))
    item = random.choice(["cookies", "apples", "candies", "eggs", "oranges"])
    meal = random.choice(MEALS)

    q = f"{person} has {n} {item}. {person} eats {m} for {meal}. How many {item} does {person} have left?"
    ir = f"step1 = {n}-{m}\n[END]"
    ans = n - m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "subtract_eats"}


def gen_subtract_spends():
    """Pattern: "{person} has ${n}. Spends ${m} on {thing}."
    Answer: n - m
    """
    person = random.choice(NAMES)
    n = random.randint(50, 200)
    m = random.randint(10, min(80, n-10))
    thing = random.choice(["a book", "lunch", "groceries", "a toy", "supplies"])

    q = f"{person} has ${n}. {person} spends ${m} on {thing}. How much money does {person} have left?"
    ir = f"step1 = {n}-{m}\n[END]"
    ans = n - m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "subtract_spends"}


def gen_subtract_gives():
    """Pattern: "{person} has {n}. Gives {m} to {other}."
    Answer: n - m
    """
    person = random.choice(NAMES)
    other = random.choice([n for n in NAMES if n != person])
    n = random.randint(15, 50)
    m = random.randint(2, min(12, n-1))
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {person} gives {m} {item} to {other}. How many {item} does {person} have now?"
    ir = f"step1 = {n}-{m}\n[END]"
    ans = n - m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "subtract_gives"}


def gen_subtract_uses():
    """Pattern: "{person} has {n}. Uses {m} for {purpose}."
    Answer: n - m
    """
    person = random.choice(NAMES)
    n = random.randint(20, 60)
    m = random.randint(5, min(20, n-5))
    item = random.choice(ITEMS)
    purpose = random.choice(PURPOSES)

    q = f"{person} has {n} {item}. {person} uses {m} for {purpose}. How many {item} does {person} have remaining?"
    ir = f"step1 = {n}-{m}\n[END]"
    ans = n - m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "subtract_uses"}


def gen_add_gets():
    """Pattern: "{person} has {n}. Gets {m} more."
    Answer: n + m
    """
    person = random.choice(NAMES)
    n = random.randint(10, 40)
    m = random.randint(5, 25)
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {person} gets {m} more. How many {item} does {person} have now?"
    ir = f"step1 = {n}+{m}\n[END]"
    ans = n + m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "add_gets"}


def gen_add_finds():
    """Pattern: "{person} has {n}. Finds {m} more."
    Answer: n + m
    """
    person = random.choice(NAMES)
    n = random.randint(5, 30)
    m = random.randint(3, 20)
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {person} finds {m} additional {item}. How many {item} does {person} have in total?"
    ir = f"step1 = {n}+{m}\n[END]"
    ans = n + m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "add_finds"}


def gen_add_earns():
    """Pattern: "{person} has ${n}. Earns ${m} more."
    Answer: n + m
    """
    person = random.choice(NAMES)
    n = random.randint(20, 100)
    m = random.randint(10, 50)

    q = f"{person} has ${n} saved. {person} earns ${m} more from working. How much money does {person} have now?"
    ir = f"step1 = {n}+{m}\n[END]"
    ans = n + m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "add_earns"}


# Multi-step patterns (GSM8K style)

def gen_two_step_buy_sell():
    """Pattern: Has {n}, sells {a} at ${p} each.
    Answer: a * p
    """
    person = random.choice(NAMES)
    n = random.randint(20, 50)
    a = random.randint(5, min(15, n-1))
    p = random.randint(2, 8)
    item = random.choice(ITEMS)

    q = f"{person} has {n} {item}. {person} sells {a} of them at ${p} each. How much money does {person} make?"
    ir = f"step1 = {a}*{p}\n[END]"
    ans = a * p

    return {"q": q, "ir": ir, "ans": ans, "pattern": "two_step_sell"}


def gen_two_step_consume_remainder():
    """Pattern: Has {n}, eats {a} for breakfast, uses {b} for baking. Remainder?
    Answer: n - a - b
    """
    person = random.choice(NAMES)
    n = random.randint(20, 50)
    a = random.randint(2, 6)
    b = random.randint(3, 8)
    item = random.choice(["eggs", "apples", "cookies", "oranges"])

    if a + b >= n:
        a = random.randint(2, 4)
        b = random.randint(2, 4)

    q = f"{person} has {n} {item}. {person} eats {a} for breakfast and uses {b} for baking. How many {item} does {person} have left?"
    ir = f"step1 = {n}-{a}\nstep2 = step1-{b}\n[END]"
    ans = n - a - b

    return {"q": q, "ir": ir, "ans": ans, "pattern": "two_step_consume"}


def gen_two_step_earn_spend():
    """Pattern: Earns ${rate} per hour for {hours} hours. Spends ${m}.
    Answer: rate * hours - m
    """
    person = random.choice(NAMES)
    rate = random.randint(10, 20)
    hours = random.randint(4, 8)
    m = random.randint(10, 40)

    total = rate * hours
    if m >= total:
        m = random.randint(10, total - 10)

    q = f"{person} earns ${rate} per hour. After working {hours} hours, {person} spends ${m} on supplies. How much money does {person} have left?"
    ir = f"step1 = {rate}*{hours}\nstep2 = step1-{m}\n[END]"
    ans = total - m

    return {"q": q, "ir": ir, "ans": ans, "pattern": "two_step_earn_spend"}


def gen_two_step_share_after():
    """Pattern: {n} items in {c} containers. Share equally among {p} people.
    Answer: (n * c) / p
    """
    c = random.randint(3, 8)
    n = random.randint(4, 12)
    p = random.randint(2, 6)
    total = n * c
    # Adjust to be divisible
    while total % p != 0:
        n += 1
        total = n * c

    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)

    q = f"There are {c} {container}, each containing {n} {item}. The {item} are shared equally among {p} friends. How many {item} does each friend get?"
    ir = f"step1 = {c}*{n}\nstep2 = step1/{p}\n[END]"
    ans = total // p

    return {"q": q, "ir": ir, "ans": ans, "pattern": "two_step_share"}


def gen_three_step_production():
    """Pattern: Makes {n} per day for {d} days. Sells {a}, gives away {b}.
    Answer: n * d - a - b
    """
    person = random.choice(NAMES)
    n = random.randint(5, 15)
    d = random.randint(3, 7)
    total = n * d
    a = random.randint(5, min(20, total // 2))
    b = random.randint(3, min(10, (total - a) // 2))
    item = random.choice(ITEMS)

    q = f"{person} makes {n} {item} per day for {d} days. {person} sells {a} of them and gives {b} to friends. How many {item} does {person} have left?"
    ir = f"step1 = {n}*{d}\nstep2 = step1-{a}\nstep3 = step2-{b}\n[END]"
    ans = total - a - b

    return {"q": q, "ir": ir, "ans": ans, "pattern": "three_step_production"}


def gen_three_step_buy_use_sell():
    """Pattern: Buys {n} for ${cost}. Uses {a}. Sells rest at ${price} each.
    Answer: (n - a) * price
    """
    person = random.choice(NAMES)
    n = random.randint(15, 40)
    cost = random.randint(20, 60)  # not used in IR since question asks revenue
    a = random.randint(3, min(10, n-5))
    price = random.randint(2, 6)
    item = random.choice(ITEMS)

    q = f"{person} buys {n} {item}. {person} uses {a} of them. {person} sells the rest at ${price} each. How much money does {person} make from selling?"
    ir = f"step1 = {n}-{a}\nstep2 = step1*{price}\n[END]"
    ans = (n - a) * price

    return {"q": q, "ir": ir, "ans": ans, "pattern": "three_step_buy_sell"}


def gen_complex_janet_ducks():
    """Janet's ducks pattern: Produces {n}, eats {a}, uses {b} for purpose, sells remainder at ${p}.
    Answer: (n - a - b) * p
    """
    person = random.choice(NAMES)
    animal = random.choice(["ducks", "chickens", "hens"])
    produce_item = "eggs"
    n = random.randint(12, 25)
    a = random.randint(2, 5)
    b = random.randint(2, 6)
    p = random.randint(2, 5)

    if a + b >= n:
        a = 2
        b = 3

    q = f"{person}'s {animal} lay {n} {produce_item} per day. {person} eats {a} for breakfast every morning and uses {b} for baking. {person} sells the remaining {produce_item} at ${p} each. How much money does {person} make each day?"
    ir = f"step1 = {n}-{a}\nstep2 = step1-{b}\nstep3 = step2*{p}\n[END]"
    ans = (n - a - b) * p

    return {"q": q, "ir": ir, "ans": ans, "pattern": "janet_ducks"}


def gen_complex_work_hours():
    """Work pattern: Earns ${rate}/hour, works {h} hours/day for {d} days. Spends ${s}.
    Answer: rate * h * d - s
    """
    person = random.choice(NAMES)
    rate = random.randint(10, 20)
    h = random.randint(4, 8)
    d = random.randint(3, 6)
    total = rate * h * d
    s = random.randint(20, min(100, total - 50))

    q = f"{person} earns ${rate} per hour. {person} works {h} hours a day for {d} days. After that, {person} spends ${s} on a new phone. How much money does {person} have left?"
    ir = f"step1 = {rate}*{h}\nstep2 = step1*{d}\nstep3 = step2-{s}\n[END]"
    ans = total - s

    return {"q": q, "ir": ir, "ans": ans, "pattern": "complex_work"}


# Pattern registry
PATTERNS = [
    # Single step - MULTIPLY
    (gen_multiply_each, 15),
    (gen_multiply_per, 15),
    (gen_multiply_times, 10),
    (gen_multiply_twice, 10),
    # Single step - DIVIDE
    (gen_divide_half, 10),
    (gen_divide_equally, 10),
    (gen_divide_split, 8),
    # Single step - SUBTRACT
    (gen_subtract_eats, 10),
    (gen_subtract_spends, 10),
    (gen_subtract_gives, 8),
    (gen_subtract_uses, 8),
    # Single step - ADD
    (gen_add_gets, 10),
    (gen_add_finds, 8),
    (gen_add_earns, 8),
    # Multi-step
    (gen_two_step_buy_sell, 15),
    (gen_two_step_consume_remainder, 15),
    (gen_two_step_earn_spend, 15),
    (gen_two_step_share_after, 12),
    (gen_three_step_production, 12),
    (gen_three_step_buy_use_sell, 12),
    # GSM8K-specific complex patterns
    (gen_complex_janet_ducks, 20),
    (gen_complex_work_hours, 15),
]


def generate_dataset(n: int) -> list[dict]:
    """Generate n examples using weighted pattern selection."""
    data = []
    total_weight = sum(w for _, w in PATTERNS)

    for _ in range(n):
        # Weighted random selection
        r = random.random() * total_weight
        cumulative = 0
        for gen_fn, weight in PATTERNS:
            cumulative += weight
            if r <= cumulative:
                data.append(gen_fn())
                break

    return data


def main():
    random.seed(42)

    print("=" * 70)
    print("  GSM8K-STYLE DATA GENERATOR")
    print("  Patterns based on mined GSM8K semantic structures")
    print("=" * 70)

    # Generate datasets
    train_data = generate_dataset(2000)
    val_data = generate_dataset(200)
    test_data = generate_dataset(400)

    # Statistics
    print(f"\nGenerated: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Pattern distribution
    pattern_counts = {}
    for item in train_data:
        p = item["pattern"]
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    print("\nPattern distribution (train):")
    for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")

    # Samples
    print("\n" + "=" * 70)
    print("SAMPLE PROBLEMS")
    print("=" * 70)

    for item in train_data[:8]:
        print(f"\nQ: {item['q']}")
        print(f"IR: {item['ir'].replace(chr(10), ' | ')}")
        print(f"Answer: {item['ans']}")

    # Save
    output_dir = Path(__file__).parent / "gsm8k_style_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\nWrote {len(data)} examples to {path}")


if __name__ == "__main__":
    main()
