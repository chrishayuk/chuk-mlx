"""
Extended Expression Data Generator v2.

Adds GSM8K-style patterns:
1. Fractions/percentages (13.5% of GSM8K)
2. "X times"/"twice" multipliers
3. Time calculations
4. Rate problems
5. Longer chains (4-6 steps)
6. Comparison patterns
"""

import random
import json
from pathlib import Path

# =============================================================================
# VOCABULARY
# =============================================================================

NAMES = ["Tom", "Sam", "Lisa", "Emma", "Jake", "Anna", "Mike", "Kate", "Ben", "Zoe",
         "Maria", "Alex", "Chris", "Dan", "Amy", "Nick", "Sara", "Ryan", "Mia", "Evan",
         "John", "Mary", "James", "Linda", "David", "Susan", "Robert", "Karen"]

ITEMS = ["apple", "cookie", "book", "pencil", "sticker", "marble", "candy",
         "flower", "toy", "card", "egg", "orange", "coin", "dollar", "point",
         "ball", "rock", "stamp", "bead", "shell", "ticket", "shirt", "bottle"]

CONTAINERS = ["box", "bag", "basket", "jar", "pack", "shelf", "row", "tray",
              "crate", "pile", "stack", "bunch", "set", "group", "carton", "case"]

# Extended verbs (from GSM8K analysis)
LOSS_VERBS = ["eats", "gives away", "loses", "uses", "spends", "drops", "breaks",
              "sells", "donates", "throws away"]
GAIN_VERBS = ["gets", "finds", "receives", "earns", "buys", "collects", "picks up",
              "wins", "saves"]

# Time units
TIME_UNITS = ["hour", "day", "week", "minute"]

# =============================================================================
# SINGLE STEP PATTERNS (existing + enhanced)
# =============================================================================

def gen_single_subtract():
    """Single subtraction with varied signals."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(20, 100)
    b = random.randint(5, a - 5)

    templates = [
        f"{name} has {a} {item}s. {name} {random.choice(LOSS_VERBS)} {b}. How many left?",
        f"{name} has {a} {item}s and loses {b}. How many remaining?",
        f"Start with {a}. Lose {b}. What remains?",
        f"Has {a} {item}s. {b} fewer now. How many left?",  # "fewer" signal
        f"{name} had {a} {item}s but now has {b} fewer. Remaining?",
    ]

    return {"q": random.choice(templates), "expr": f"{a} - {b} =\n[END]", "ans": a - b}


def gen_single_multiply():
    """Single multiplication with varied signals."""
    container = random.choice(CONTAINERS)
    item = random.choice(ITEMS)

    a = random.randint(3, 12)
    b = random.randint(4, 15)

    templates = [
        f"{a} {container}s with {b} {item}s each. Total {item}s?",
        f"Has {a} {container}s. Each has {b} {item}s. How many in all?",  # "in all"
        f"{a} rows of {b}. What is the total?",  # "total"
        f"{a} groups with {b} each. Combined?",  # "combined"
        f"There are {a} {container}s. Each contains {b} {item}s. Together?",  # "together"
    ]

    return {"q": random.choice(templates), "expr": f"{a} * {b} =\n[END]", "ans": a * b}


def gen_single_add():
    """Single addition with varied signals."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(10, 50)
    b = random.randint(5, 40)

    templates = [
        f"{name} has {a} {item}s. {name} {random.choice(GAIN_VERBS)} {b} more. Total?",
        f"Has {a}. Gets {b} more. How many in all?",
        f"{name} has {a} {item}s and {b} more. Combined total?",
        f"Start with {a}. Add {b}. What is the total?",
    ]

    return {"q": random.choice(templates), "expr": f"{a} + {b} =\n[END]", "ans": a + b}


def gen_single_divide():
    """Single division."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    divisor = random.randint(3, 10)
    result = random.randint(3, 15)
    total = divisor * result

    templates = [
        f"Divide {total} {item}s among {divisor} people. Each gets?",
        f"{name} splits {total} {item}s equally among {divisor} friends. Each gets?",
        f"Share {total} items with {divisor} people equally. How many each?",
    ]

    return {"q": random.choice(templates), "expr": f"{total} / {divisor} =\n[END]", "ans": result}


# =============================================================================
# NEW: "X TIMES" / "TWICE" PATTERNS (30 signals in GSM8K)
# =============================================================================

def gen_times_multiply():
    """X times pattern: "twice as many", "3 times"."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    base = random.randint(5, 30)
    multiplier = random.choice([2, 2, 2, 3, 3, 4, 5])  # twice is most common

    if multiplier == 2:
        mult_text = "twice"
    else:
        mult_text = f"{multiplier} times"

    templates = [
        f"{name} has {base} {item}s. Gets {mult_text} as many more. Total now?",
        f"Has {base}. Earns {mult_text} that amount. New total?",
        f"{name} had {base} {item}s. Now has {mult_text} as many. How many now?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{base} * {multiplier} =\n[END]",
        "ans": base * multiplier
    }


def gen_twice_compare():
    """Comparison: A has X, B has twice as many."""
    name1 = random.choice(NAMES)
    name2 = random.choice([n for n in NAMES if n != name1])
    item = random.choice(ITEMS)

    a = random.randint(10, 40)
    multiplier = random.choice([2, 2, 3])

    if multiplier == 2:
        mult_text = "twice"
    else:
        mult_text = f"{multiplier} times"

    templates = [
        f"{name1} has {a} {item}s. {name2} has {mult_text} as many. How many does {name2} have?",
        f"{name1} collected {a} {item}s. {name2} collected {mult_text} that. {name2}'s total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {multiplier} =\n[END]",
        "ans": a * multiplier
    }


# =============================================================================
# NEW: FRACTION/PERCENTAGE PATTERNS (67 signals in GSM8K - 13.5%)
# =============================================================================

def gen_half_of():
    """Half of X pattern."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    # Ensure even number
    total = random.randint(10, 50) * 2

    templates = [
        f"{name} has {total} {item}s. Uses half. How many left?",
        f"Start with {total}. Take half. Remaining?",
        f"Has {total} {item}s. Gives away half. How many remain?",
        f"{name} had {total} {item}s and lost half of them. Left?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{total} / 2 =\n[END]",
        "ans": total // 2
    }


def gen_third_of():
    """Third of X pattern."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    # Ensure divisible by 3
    total = random.randint(5, 20) * 3

    templates = [
        f"{name} has {total} {item}s. Uses a third. How many used?",
        f"Has {total}. Sells a third. How many sold?",
        f"{name} gives away a third of {total} {item}s. How many given?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{total} / 3 =\n[END]",
        "ans": total // 3
    }


def gen_percent_of():
    """Percentage pattern: X% of Y."""
    name = random.choice(NAMES)

    # Use nice percentages that give whole numbers
    total = random.choice([100, 200, 50, 80, 120, 150, 200])
    percent = random.choice([10, 20, 25, 50, 75])

    result = total * percent // 100

    templates = [
        f"{name} has ${total}. Spends {percent}%. How much spent?",
        f"A store has {total} items. {percent}% are sold. How many sold?",
        f"{name} earns ${total}. Saves {percent}%. How much saved?",
        f"Of {total} students, {percent}% passed. How many passed?",
    ]

    # Express as: total * percent / 100
    # Simplify: if percent is 50, that's / 2; if 25, that's / 4, etc.
    if percent == 50:
        expr = f"{total} / 2 =\n[END]"
    elif percent == 25:
        expr = f"{total} / 4 =\n[END]"
    elif percent == 75:
        expr = f"{total} * 3 =\n_ / 4 =\n[END]"
    elif percent == 10:
        expr = f"{total} / 10 =\n[END]"
    elif percent == 20:
        expr = f"{total} / 5 =\n[END]"
    else:
        expr = f"{total} * {percent} =\n_ / 100 =\n[END]"

    return {
        "q": random.choice(templates),
        "expr": expr,
        "ans": result
    }


# =============================================================================
# NEW: TIME PATTERNS (51 signals in GSM8K)
# =============================================================================

def gen_rate_time():
    """Rate × Time: X per hour for Y hours."""
    name = random.choice(NAMES)

    rate = random.randint(5, 20)
    time = random.randint(2, 8)
    unit = random.choice(TIME_UNITS)

    templates = [
        f"{name} earns ${rate} per {unit}. Works {time} {unit}s. Total earned?",
        f"A machine makes {rate} items per {unit}. Runs for {time} {unit}s. Total items?",
        f"{name} reads {rate} pages per {unit}. Reads for {time} {unit}s. Pages read?",
        f"Travels {rate} miles per {unit}. Goes for {time} {unit}s. Distance?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{rate} * {time} =\n[END]",
        "ans": rate * time
    }


def gen_daily_total():
    """Daily rate over multiple days."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    daily = random.randint(5, 20)
    days = random.randint(3, 10)

    templates = [
        f"{name} collects {daily} {item}s every day for {days} days. Total collected?",
        f"Earns ${daily} per day for {days} days. Total earnings?",
        f"Uses {daily} items daily for {days} days. Total used?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{daily} * {days} =\n[END]",
        "ans": daily * days
    }


# =============================================================================
# NEW: RATE/PRICE PATTERNS (25 signals in GSM8K)
# =============================================================================

def gen_price_quantity():
    """Price × Quantity: $X each, buy Y."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    price = random.randint(2, 15)
    qty = random.randint(3, 12)

    templates = [
        f"{name} buys {qty} {item}s at ${price} each. Total cost?",
        f"{item.capitalize()}s cost ${price} each. Buys {qty}. How much spent?",
        f"Each {item} is ${price}. {name} gets {qty}. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{price} * {qty} =\n[END]",
        "ans": price * qty
    }


def gen_price_change():
    """Buy at X, sell at Y: profit calculation."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    buy_price = random.randint(5, 20)
    sell_price = buy_price + random.randint(2, 10)
    qty = random.randint(3, 10)

    profit_per = sell_price - buy_price
    total_profit = profit_per * qty

    templates = [
        f"{name} buys {qty} {item}s at ${buy_price} each. Sells at ${sell_price} each. Total profit?",
        f"Buys items for ${buy_price}, sells for ${sell_price}. With {qty} items, profit?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{sell_price} - {buy_price} =\n_ * {qty} =\n[END]",
        "ans": total_profit
    }


# =============================================================================
# MULTI-STEP PATTERNS (existing + enhanced)
# =============================================================================

def gen_subtract_subtract():
    """Two subtractions."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(50, 100)
    b = random.randint(5, 20)
    c = random.randint(5, min(20, a - b - 5))

    templates = [
        f"{name} has {a} {item}s. Loses {b}, then loses {c} more. How many remaining?",
        f"Has {a}. Loses {b}, loses {c}. Left?",
        f"Start with {a}. Use {b}. Then use {c} more. Remaining?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ - {c} =\n[END]",
        "ans": a - b - c
    }


def gen_multiply_subtract():
    """Multiply then subtract."""
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)

    a = random.randint(4, 10)
    b = random.randint(5, 12)
    total = a * b
    c = random.randint(5, total - 5)

    templates = [
        f"{a} {container}s with {b} {item}s each. Uses {c}. How many remaining?",
        f"Has {a} groups of {b}. Removes {c}. Left?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ - {c} =\n[END]",
        "ans": total - c
    }


def gen_subtract_multiply():
    """Subtract then multiply."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(30, 60)
    b = random.randint(5, 15)
    c = random.randint(2, 5)

    templates = [
        f"{name} has {a} {item}s. Uses {b}. Sells rest at ${c} each. Revenue?",
        f"Has {a}. Uses {b}. Sells remaining for ${c} each. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ * {c} =\n[END]",
        "ans": (a - b) * c
    }


def gen_add_subtract():
    """Add then subtract."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(20, 50)
    b = random.randint(10, 30)
    c = random.randint(5, a + b - 5)

    templates = [
        f"{name} has {a} {item}s. Gets {b} more. Then loses {c}. How many now?",
        f"Has {a}. Gains {b}. Spends {c}. Remaining?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} + {b} =\n_ - {c} =\n[END]",
        "ans": a + b - c
    }


def gen_multiply_divide():
    """Multiply then divide."""
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)

    a = random.randint(4, 8)
    b = random.randint(6, 12)
    total = a * b
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


# =============================================================================
# NEW: LONGER CHAINS (4-6 steps) - 42% of GSM8K
# =============================================================================

def gen_four_step_chain():
    """Four-step problem."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(80, 120)
    b = random.randint(10, 20)
    c = random.randint(5, 15)
    d = random.randint(5, 10)
    e = random.randint(2, 4)

    result1 = a - b
    result2 = result1 - c
    result3 = result2 - d
    result4 = result3 * e

    templates = [
        f"{name} has {a} {item}s. Loses {b}. Then loses {c}. Then loses {d} more. Sells rest at ${e} each. Revenue?",
        f"Start with {a}. Remove {b}, then {c}, then {d}. Sell remaining for ${e} each. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ - {c} =\n_ - {d} =\n_ * {e} =\n[END]",
        "ans": result4
    }


def gen_five_step_chain():
    """Five-step problem."""
    name = random.choice(NAMES)

    a = random.randint(5, 8)
    b = random.randint(8, 12)
    c = random.randint(10, 20)
    d = random.randint(5, 10)
    e = random.randint(2, 5)

    result1 = a * b
    result2 = result1 + c
    result3 = result2 - d
    # Find divisor for result3
    divisors = [x for x in range(2, 8) if result3 % x == 0]
    f = random.choice(divisors) if divisors else 1
    result4 = result3 // f
    result5 = result4 * e

    templates = [
        f"{name} has {a} boxes with {b} items each. Gets {c} more items. Uses {d}. Splits rest among {f} people. Each sells their share at ${e} each. Each person's revenue?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ + {c} =\n_ - {d} =\n_ / {f} =\n_ * {e} =\n[END]",
        "ans": result5
    }


def gen_rate_time_chain():
    """Rate × Time with additional steps."""
    name = random.choice(NAMES)

    rate = random.randint(8, 15)
    hours = random.randint(4, 8)
    expense = random.randint(10, 30)
    bonus = random.randint(10, 20)

    earned = rate * hours
    after_expense = earned - expense
    final = after_expense + bonus

    templates = [
        f"{name} earns ${rate} per hour. Works {hours} hours. Spends ${expense} on lunch. Gets ${bonus} bonus. Final amount?",
        f"Earns ${rate}/hour for {hours} hours. Subtracts ${expense} expense. Adds ${bonus} tip. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{rate} * {hours} =\n_ - {expense} =\n_ + {bonus} =\n[END]",
        "ans": final
    }


def gen_buy_sell_chain():
    """Buy, modify, sell chain."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    qty = random.randint(10, 20)
    buy_price = random.randint(3, 8)
    broken = random.randint(2, 5)
    sell_price = buy_price + random.randint(2, 5)

    cost = qty * buy_price
    remaining = qty - broken
    revenue = remaining * sell_price
    profit = revenue - cost

    templates = [
        f"{name} buys {qty} {item}s at ${buy_price} each. {broken} break. Sells rest at ${sell_price} each. Profit?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{qty} * {buy_price} =\n{qty} - {broken} =\n_ * {sell_price} =\n_ - {cost} =\n[END]",
        "ans": profit
    }


def gen_three_step_standard():
    """Standard three-step chain."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    a = random.randint(60, 100)
    b = random.randint(10, 20)
    c = random.randint(5, 15)
    d = random.randint(2, 4)

    templates = [
        f"{name} has {a} {item}s. Loses {b}. Loses {c} more. Sells rest at ${d} each. Revenue?",
        f"Has {a}. Uses {b}, then {c}. Sells remaining for ${d} each. Total?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} - {b} =\n_ - {c} =\n_ * {d} =\n[END]",
        "ans": (a - b - c) * d
    }


def gen_multiply_subtract_divide():
    """Multiply, subtract, divide chain."""
    item = random.choice(ITEMS)

    a = random.randint(5, 8)
    b = random.randint(8, 12)
    total = a * b
    c = random.randint(5, 15)
    remaining = total - c
    divisors = [x for x in range(2, 8) if remaining % x == 0]
    d = random.choice(divisors) if divisors else 1

    templates = [
        f"{a} rows of {b} {item}s. Removes {c}. Splits rest among {d} people. Each gets?",
    ]

    return {
        "q": random.choice(templates),
        "expr": f"{a} * {b} =\n_ - {c} =\n_ / {d} =\n[END]",
        "ans": remaining // d
    }


# =============================================================================
# GENERATOR CONFIG
# =============================================================================

GENERATORS = {
    # Single step (25%)
    "single_subtract": (gen_single_subtract, 0.06),
    "single_multiply": (gen_single_multiply, 0.06),
    "single_add": (gen_single_add, 0.06),
    "single_divide": (gen_single_divide, 0.04),

    # NEW: Times/twice patterns (8%)
    "times_multiply": (gen_times_multiply, 0.04),
    "twice_compare": (gen_twice_compare, 0.04),

    # NEW: Fraction/percentage (10%)
    "half_of": (gen_half_of, 0.04),
    "third_of": (gen_third_of, 0.03),
    "percent_of": (gen_percent_of, 0.03),

    # NEW: Time/rate patterns (8%)
    "rate_time": (gen_rate_time, 0.04),
    "daily_total": (gen_daily_total, 0.02),
    "price_quantity": (gen_price_quantity, 0.02),

    # Two-step (24%)
    "subtract_subtract": (gen_subtract_subtract, 0.06),
    "multiply_subtract": (gen_multiply_subtract, 0.05),
    "subtract_multiply": (gen_subtract_multiply, 0.05),
    "add_subtract": (gen_add_subtract, 0.04),
    "multiply_divide": (gen_multiply_divide, 0.04),

    # NEW: Price chain (3%)
    "price_change": (gen_price_change, 0.03),

    # Three-step (12%)
    "three_step_standard": (gen_three_step_standard, 0.06),
    "multiply_subtract_divide": (gen_multiply_subtract_divide, 0.03),
    "rate_time_chain": (gen_rate_time_chain, 0.03),

    # NEW: Four+ step chains (10%)
    "four_step_chain": (gen_four_step_chain, 0.05),
    "five_step_chain": (gen_five_step_chain, 0.03),
    "buy_sell_chain": (gen_buy_sell_chain, 0.02),
}


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    """Generate n examples with weighted sampling."""
    random.seed(seed)

    generators = list(GENERATORS.keys())
    weights = [GENERATORS[g][1] for g in generators]

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    data = []
    for _ in range(n):
        gen_name = random.choices(generators, weights=weights)[0]
        gen_fn = GENERATORS[gen_name][0]

        try:
            example = gen_fn()
            example["pattern"] = gen_name
            data.append(example)
        except Exception as e:
            # Retry with different generator
            continue

    return data


def save_jsonl(data: list[dict], path: str):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    from collections import Counter

    output_dir = Path(__file__).parent / "expr_data_v2"
    output_dir.mkdir(exist_ok=True)

    # Generate larger datasets
    train = generate_dataset(1000, seed=42)
    val = generate_dataset(100, seed=123)
    test = generate_dataset(200, seed=456)

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    print(f"Generated:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

    # Pattern distribution
    patterns = Counter(d["pattern"] for d in train)

    print(f"\nPattern distribution (train):")

    # Group by category
    categories = {
        "Single step": ["single_subtract", "single_multiply", "single_add", "single_divide"],
        "Times/twice": ["times_multiply", "twice_compare"],
        "Fractions": ["half_of", "third_of", "percent_of"],
        "Time/rate": ["rate_time", "daily_total", "price_quantity", "price_change"],
        "Two-step": ["subtract_subtract", "multiply_subtract", "subtract_multiply", "add_subtract", "multiply_divide"],
        "Three-step": ["three_step_standard", "multiply_subtract_divide", "rate_time_chain"],
        "Four+ step": ["four_step_chain", "five_step_chain", "buy_sell_chain"],
    }

    for cat, pats in categories.items():
        cat_count = sum(patterns.get(p, 0) for p in pats)
        cat_pct = cat_count / len(train) * 100
        print(f"\n  {cat}: {cat_count} ({cat_pct:.0f}%)")
        for p in pats:
            if patterns.get(p, 0) > 0:
                print(f"    {p}: {patterns[p]}")

    # Sample by category
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES BY CATEGORY")
    print("=" * 60)

    for cat, pats in categories.items():
        samples = [d for d in train if d["pattern"] in pats]
        if samples:
            sample = random.choice(samples)
            print(f"\n[{cat}] {sample['pattern']}")
            print(f"  Q: {sample['q']}")
            print(f"  Expr: {sample['expr'].replace(chr(10), ' | ')}")
            print(f"  Ans: {sample['ans']}")
