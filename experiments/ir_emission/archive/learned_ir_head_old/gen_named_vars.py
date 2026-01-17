"""
Named Variable Expression Generator.

Format:
  cost = 10 * 8
  remaining = 10 - 4
  revenue = remaining * 11
  profit = revenue - cost
  [END]

Maps directly to IR/SSA form.
"""

import random
import json
import re
from pathlib import Path

# =============================================================================
# VOCABULARY
# =============================================================================

NAMES = ["Tom", "Sam", "Lisa", "Emma", "Jake", "Anna", "Mike", "Kate", "Ben", "Zoe",
         "Maria", "Alex", "Chris", "Dan", "Amy", "Nick", "Sara", "Ryan", "Mia", "Evan"]

ITEMS = ["apple", "cookie", "book", "pencil", "sticker", "marble", "candy",
         "flower", "toy", "card", "egg", "orange", "coin", "ball", "shirt"]

CONTAINERS = ["box", "bag", "basket", "jar", "pack", "shelf", "row", "tray", "crate"]

TIME_UNITS = ["hour", "day", "week"]


# =============================================================================
# NAMED VARIABLE EXECUTOR
# =============================================================================

def execute_named(code: str) -> tuple[int | None, str, dict]:
    """
    Execute named variable expression chain.
    Returns (final_result, reason, all_vars)
    """
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    lines = [l for l in lines if l != '[END]']

    if not lines:
        return None, "empty", {}

    variables = {}
    last_result = None

    for line in lines:
        # Parse: varname = expr
        match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if not match:
            return None, f"parse_fail:{line[:20]}", variables

        var_name, expr = match.groups()
        expr = expr.strip()

        # Parse expression: operand op operand
        # Operands can be numbers or variable names
        expr_match = re.match(r'(\w+)\s*([+\-*/])\s*(\w+)', expr)
        if not expr_match:
            # Maybe just a number assignment
            if expr.isdigit():
                variables[var_name] = int(expr)
                last_result = int(expr)
                continue
            return None, f"expr_fail:{expr[:20]}", variables

        left, op, right = expr_match.groups()

        # Resolve operands
        try:
            left_val = int(left) if left.isdigit() else variables[left]
            right_val = int(right) if right.isdigit() else variables[right]
        except KeyError as e:
            return None, f"undefined:{e}", variables

        # Compute
        if op == '+':
            result = left_val + right_val
        elif op == '-':
            result = left_val - right_val
        elif op == '*':
            result = left_val * right_val
        elif op == '/':
            if right_val == 0:
                return None, "div_zero", variables
            result = left_val // right_val
        else:
            return None, f"bad_op:{op}", variables

        variables[var_name] = result
        last_result = result

    return last_result, "ok", variables


def compute_reward(code: str, expected: int) -> tuple[float, str]:
    result, reason, _ = execute_named(code)

    if result is None:
        return 0.0, reason

    if result == expected:
        return 1.0, "correct"

    return 0.3, f"wrong:{result}"


# =============================================================================
# SINGLE VARIABLE PATTERNS (simple, use last var as answer)
# =============================================================================

def gen_single_op():
    """Single operation with named result."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    op_type = random.choice(['add', 'subtract', 'multiply', 'divide'])

    if op_type == 'add':
        a, b = random.randint(10, 50), random.randint(5, 30)
        q = f"{name} has {a} {item}s. Gets {b} more. Total?"
        expr = f"total = {a} + {b}\n[END]"
        ans = a + b
    elif op_type == 'subtract':
        a = random.randint(30, 80)
        b = random.randint(5, a - 5)
        q = f"{name} has {a} {item}s. Uses {b}. How many left?"
        expr = f"remaining = {a} - {b}\n[END]"
        ans = a - b
    elif op_type == 'multiply':
        a, b = random.randint(3, 12), random.randint(4, 15)
        container = random.choice(CONTAINERS)
        q = f"{a} {container}s with {b} {item}s each. Total?"
        expr = f"total = {a} * {b}\n[END]"
        ans = a * b
    else:  # divide
        divisor = random.randint(3, 10)
        result = random.randint(3, 12)
        total = divisor * result
        q = f"Split {total} {item}s among {divisor} people. Each gets?"
        expr = f"each = {total} / {divisor}\n[END]"
        ans = result

    return {"q": q, "expr": expr, "ans": ans}


# =============================================================================
# TWO-STEP CHAINS (sequential, can use _ style or named)
# =============================================================================

def gen_two_step_sequential():
    """Two sequential operations."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    pattern = random.choice(['sub_sub', 'mul_sub', 'sub_mul', 'add_sub'])

    if pattern == 'sub_sub':
        a = random.randint(50, 100)
        b = random.randint(5, 20)
        c = random.randint(5, min(20, a - b - 5))
        q = f"{name} has {a} {item}s. Loses {b}, then loses {c}. Remaining?"
        expr = f"after_first = {a} - {b}\nremaining = after_first - {c}\n[END]"
        ans = a - b - c

    elif pattern == 'mul_sub':
        a, b = random.randint(4, 10), random.randint(5, 12)
        c = random.randint(5, a * b - 5)
        container = random.choice(CONTAINERS)
        q = f"{a} {container}s with {b} each. Uses {c}. Left?"
        expr = f"total = {a} * {b}\nremaining = total - {c}\n[END]"
        ans = a * b - c

    elif pattern == 'sub_mul':
        a = random.randint(30, 60)
        b = random.randint(5, 15)
        c = random.randint(2, 5)
        q = f"{name} has {a} {item}s. Uses {b}. Sells rest at ${c} each. Revenue?"
        expr = f"remaining = {a} - {b}\nrevenue = remaining * {c}\n[END]"
        ans = (a - b) * c

    else:  # add_sub
        a = random.randint(20, 50)
        b = random.randint(10, 30)
        c = random.randint(5, a + b - 5)
        q = f"{name} has {a}. Gets {b} more. Spends {c}. Left?"
        expr = f"after_gain = {a} + {b}\nremaining = after_gain - {c}\n[END]"
        ans = a + b - c

    return {"q": q, "expr": expr, "ans": ans}


# =============================================================================
# MULTI-VARIABLE PATTERNS (the key new capability!)
# =============================================================================

def gen_buy_sell_profit():
    """Buy X at price A, some break, sell rest at price B. Profit?"""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    qty = random.randint(10, 25)
    buy_price = random.randint(3, 10)
    broken = random.randint(2, 6)
    sell_price = buy_price + random.randint(2, 8)

    cost = qty * buy_price
    remaining = qty - broken
    revenue = remaining * sell_price
    profit = revenue - cost

    q = f"{name} buys {qty} {item}s at ${buy_price} each. {broken} break. Sells rest at ${sell_price} each. Profit?"

    expr = f"""cost = {qty} * {buy_price}
remaining = {qty} - {broken}
revenue = remaining * {sell_price}
profit = revenue - cost
[END]"""

    return {"q": q, "expr": expr, "ans": profit}


def gen_two_people_compare():
    """A has X, B has Y times as many. Combined?"""
    name1 = random.choice(NAMES)
    name2 = random.choice([n for n in NAMES if n != name1])
    item = random.choice(ITEMS)

    a_count = random.randint(10, 30)
    multiplier = random.choice([2, 2, 3, 3, 4])

    if multiplier == 2:
        mult_text = "twice"
    else:
        mult_text = f"{multiplier} times"

    b_count = a_count * multiplier
    total = a_count + b_count

    q = f"{name1} has {a_count} {item}s. {name2} has {mult_text} as many. Combined total?"

    expr = f"""a_count = {a_count}
b_count = a_count * {multiplier}
total = a_count + b_count
[END]"""

    return {"q": q, "expr": expr, "ans": total}


def gen_rate_expense_bonus():
    """Earns rate*time, minus expense, plus bonus."""
    name = random.choice(NAMES)

    rate = random.randint(8, 20)
    hours = random.randint(4, 10)
    expense = random.randint(10, 40)
    bonus = random.randint(10, 30)

    earned = rate * hours
    after_expense = earned - expense
    final = after_expense + bonus

    q = f"{name} earns ${rate}/hour for {hours} hours. Spends ${expense} on supplies. Gets ${bonus} tip. Final amount?"

    expr = f"""earned = {rate} * {hours}
after_expense = earned - {expense}
final = after_expense + {bonus}
[END]"""

    return {"q": q, "expr": expr, "ans": final}


def gen_discount_calculation():
    """Original price, discount %, final price."""
    item = random.choice(ITEMS)

    # Use prices that work out to nice percentages
    original = random.choice([40, 50, 60, 80, 100, 120])
    discount_pct = random.choice([10, 20, 25, 50])

    discount_amt = original * discount_pct // 100
    final_price = original - discount_amt

    q = f"A {item} costs ${original}. It's {discount_pct}% off. Final price?"

    # Express discount as division for clean math
    if discount_pct == 50:
        expr = f"""original = {original}
discount = original / 2
final = original - discount
[END]"""
    elif discount_pct == 25:
        expr = f"""original = {original}
discount = original / 4
final = original - discount
[END]"""
    elif discount_pct == 10:
        expr = f"""original = {original}
discount = original / 10
final = original - discount
[END]"""
    else:  # 20%
        expr = f"""original = {original}
discount = original / 5
final = original - discount
[END]"""

    return {"q": q, "expr": expr, "ans": final_price}


def gen_split_and_share():
    """Total items, split into groups, each group shares among people."""
    item = random.choice(ITEMS)

    total = random.randint(60, 120)
    num_groups = random.choice([2, 3, 4, 5, 6])
    # Ensure divisible
    total = (total // num_groups) * num_groups
    per_group = total // num_groups

    people_per_group = random.choice([d for d in [2, 3, 4, 5, 6] if per_group % d == 0])
    each_gets = per_group // people_per_group

    q = f"{total} {item}s split into {num_groups} groups. Each group shared among {people_per_group} people. Each person gets?"

    expr = f"""total = {total}
per_group = total / {num_groups}
each_gets = per_group / {people_per_group}
[END]"""

    return {"q": q, "expr": expr, "ans": each_gets}


def gen_profit_margin():
    """Cost to make, sell price, quantity sold, total profit."""
    item = random.choice(ITEMS)
    name = random.choice(NAMES)

    cost_each = random.randint(3, 10)
    sell_each = cost_each + random.randint(2, 8)
    qty = random.randint(5, 15)

    profit_each = sell_each - cost_each
    total_profit = profit_each * qty

    q = f"{name} makes {item}s for ${cost_each} each, sells for ${sell_each} each. Sells {qty}. Total profit?"

    expr = f"""profit_each = {sell_each} - {cost_each}
total_profit = profit_each * {qty}
[END]"""

    return {"q": q, "expr": expr, "ans": total_profit}


# =============================================================================
# LONGER CHAINS (4-5 steps)
# =============================================================================

def gen_four_step_business():
    """Buy inventory, some damaged, sell rest, pay tax."""
    name = random.choice(NAMES)
    item = random.choice(ITEMS)

    qty = random.randint(20, 40)
    cost_each = random.randint(5, 12)
    damaged = random.randint(3, 8)
    sell_each = cost_each + random.randint(3, 10)

    total_cost = qty * cost_each
    good_items = qty - damaged
    revenue = good_items * sell_each
    profit = revenue - total_cost

    q = f"{name} buys {qty} {item}s at ${cost_each} each. {damaged} are damaged. Sells good ones at ${sell_each} each. Profit?"

    expr = f"""total_cost = {qty} * {cost_each}
good_items = {qty} - {damaged}
revenue = good_items * {sell_each}
profit = revenue - total_cost
[END]"""

    return {"q": q, "expr": expr, "ans": profit}


def gen_five_step_weekly():
    """Daily earnings over week, expenses, savings."""
    name = random.choice(NAMES)

    weekday_rate = random.randint(50, 100)
    weekend_rate = weekday_rate + random.randint(20, 50)
    weekdays = 5
    weekend_days = 2
    weekly_expense = random.randint(100, 200)

    weekday_total = weekday_rate * weekdays
    weekend_total = weekend_rate * weekend_days
    gross = weekday_total + weekend_total
    net = gross - weekly_expense

    q = f"{name} earns ${weekday_rate}/day on weekdays, ${weekend_rate}/day on weekends. Weekly expenses are ${weekly_expense}. Weekly savings?"

    expr = f"""weekday_total = {weekday_rate} * {weekdays}
weekend_total = {weekend_rate} * {weekend_days}
gross = weekday_total + weekend_total
savings = gross - {weekly_expense}
[END]"""

    return {"q": q, "expr": expr, "ans": net}


# =============================================================================
# GENERATOR CONFIG
# =============================================================================

GENERATORS = {
    # Single step (20%)
    "single_op": (gen_single_op, 0.20),

    # Two-step sequential (25%)
    "two_step_sequential": (gen_two_step_sequential, 0.25),

    # Multi-variable (40%) - the key patterns
    "buy_sell_profit": (gen_buy_sell_profit, 0.10),
    "two_people_compare": (gen_two_people_compare, 0.08),
    "rate_expense_bonus": (gen_rate_expense_bonus, 0.06),
    "discount_calculation": (gen_discount_calculation, 0.06),
    "split_and_share": (gen_split_and_share, 0.05),
    "profit_margin": (gen_profit_margin, 0.05),

    # Longer chains (15%)
    "four_step_business": (gen_four_step_business, 0.08),
    "five_step_weekly": (gen_five_step_weekly, 0.07),
}


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    random.seed(seed)

    generators = list(GENERATORS.keys())
    weights = [GENERATORS[g][1] for g in generators]
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
        except:
            continue

    return data


def save_jsonl(data: list[dict], path: str):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from collections import Counter

    # Test executor
    print("=" * 60)
    print("TESTING NAMED VARIABLE EXECUTOR")
    print("=" * 60)

    test_cases = [
        ("total = 10 + 5\n[END]", 15),
        ("cost = 10 * 8\nremaining = 10 - 4\nrevenue = remaining * 11\nprofit = revenue - cost\n[END]", -14),
        ("a = 20\nb = a * 2\ntotal = a + b\n[END]", 60),
    ]

    for code, expected in test_cases:
        result, reason, vars = execute_named(code)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} Code: {code.replace(chr(10), ' | ')}")
        print(f"  Result: {result}, Expected: {expected}, Vars: {vars}")

    # Generate datasets
    print("\n" + "=" * 60)
    print("GENERATING DATASETS")
    print("=" * 60)

    output_dir = Path(__file__).parent / "named_var_data"
    output_dir.mkdir(exist_ok=True)

    train = generate_dataset(1000, seed=42)
    val = generate_dataset(100, seed=123)
    test = generate_dataset(200, seed=456)

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    print(f"\nGenerated:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")

    # Distribution
    patterns = Counter(d["pattern"] for d in train)
    print(f"\nPattern distribution:")
    for p, c in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c} ({c/len(train)*100:.0f}%)")

    # Samples
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)

    for pattern in GENERATORS.keys():
        samples = [d for d in train if d["pattern"] == pattern]
        if samples:
            s = random.choice(samples)
            print(f"\n[{pattern}]")
            print(f"  Q: {s['q']}")
            print(f"  Expr:\n    " + s['expr'].replace('\n', '\n    '))
            print(f"  Ans: {s['ans']}")
