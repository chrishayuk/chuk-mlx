"""
Generate SFT Training Data for Word Problem Normalizer.

Uses the prompt-engineered pipeline to generate (problem, expression) pairs,
then VERIFIES each pair by executing through WASM.

Only pairs that:
1. Parse successfully
2. Execute to the expected answer

are included in the training set. This gives us verified ground truth.
"""

import sys
from pathlib import Path
import random
import json
import re

import mlx.core as mx

# Add project root for chuk_lazarus imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from chuk_lazarus.models_v2.loader import load_model
from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime


# =============================================================================
# Execution Verification
# =============================================================================

def parse_expression(text: str) -> list[dict] | None:
    """Parse expression to IR using shunting yard."""
    text = text.strip().rstrip("=").strip()
    tokens = re.findall(r'\d+|[+\-*/()]', text)

    if not tokens:
        return None

    output = []
    op_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    op_map = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

    try:
        for token in tokens:
            if token.isdigit():
                output.append({'type': 'push', 'value': int(token)})
            elif token in precedence:
                while (op_stack and op_stack[-1] != '(' and
                       op_stack[-1] in precedence and
                       precedence[op_stack[-1]] >= precedence[token]):
                    output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
                op_stack.append(token)
            elif token == '(':
                op_stack.append(token)
            elif token == ')':
                while op_stack and op_stack[-1] != '(':
                    output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
                if op_stack:
                    op_stack.pop()

        while op_stack:
            if op_stack[-1] != '(':
                output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
            else:
                op_stack.pop()

        return output if output else None
    except:
        return None


def execute_ir(ir_seq: list[dict], runtime: WASMRuntime) -> int | None:
    """Execute IR sequence via WASM."""
    OP_TO_WASM = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    try:
        body = bytearray()
        for instr in ir_seq:
            if instr['type'] == 'push':
                body.extend(encode_i32_const(instr['value']))
            elif instr['type'] == 'op':
                body.extend(OPCODE_TO_WASM[OP_TO_WASM[instr['op']]])

        result = runtime.execute(bytes(body))
        return result.result if result.success else None
    except:
        return None


def verify_expression(expression: str, expected: int, runtime: WASMRuntime) -> bool:
    """Verify expression parses and executes to expected value."""
    ir = parse_expression(expression)
    if ir is None:
        return False

    result = execute_ir(ir, runtime)
    return result == expected


# =============================================================================
# Problem Generation (with known answers)
# =============================================================================

def generate_problems_with_answers(n: int) -> list[dict]:
    """Generate word problems with computed answers."""
    random.seed(42)
    problems = []

    names = ["Tom", "Sarah", "John", "Lisa", "Mike", "Emma", "Alex", "Maria",
             "James", "Anna", "David", "Kate", "Chris", "Amy", "Ben", "Zoe"]

    # Type 1: Simple operations
    templates_simple = [
        # Multiply
        (lambda a, b: f"{random.choice(names)} buys {a} items at ${b} each. How much total?",
         lambda a, b: a * b, lambda a, b: f"{a} * {b} ="),
        (lambda a, b: f"There are {a} boxes with {b} apples each. How many apples?",
         lambda a, b: a * b, lambda a, b: f"{a} * {b} ="),
        (lambda a, b: f"{random.choice(names)} earns ${a} per hour for {b} hours. Total?",
         lambda a, b: a * b, lambda a, b: f"{a} * {b} ="),

        # Divide
        (lambda a, b: f"{a} cookies shared equally among {b} friends. Each gets?",
         lambda a, b: a // b, lambda a, b: f"{a} / {b} ="),
        (lambda a, b: f"${a} split equally between {b} people. Each gets?",
         lambda a, b: a // b, lambda a, b: f"{a} / {b} ="),

        # Add
        (lambda a, b: f"{random.choice(names)} has {a} stickers and gets {b} more. Total?",
         lambda a, b: a + b, lambda a, b: f"{a} + {b} ="),
        (lambda a, b: f"A store has {a} red apples and {b} green apples. Total?",
         lambda a, b: a + b, lambda a, b: f"{a} + {b} ="),

        # Subtract
        (lambda a, b: f"{random.choice(names)} has {a} candies and eats {b}. How many left?",
         lambda a, b: a - b, lambda a, b: f"{a} - {b} ="),
        (lambda a, b: f"{random.choice(names)} has ${a} and spends ${b}. How much left?",
         lambda a, b: a - b, lambda a, b: f"{a} - {b} ="),
    ]

    # Type 2: Two operations
    templates_two = [
        # has X, buys Y at Z each
        (lambda a, b, c: f"{random.choice(names)} has ${a}. Buys {b} toys at ${c} each. Left?",
         lambda a, b, c: a - b * c, lambda a, b, c: f"{a} - {b} * {c} ="),

        # makes X per batch, Y batches, sells Z
        (lambda a, b, c: f"{random.choice(names)} bakes {a} cookies per batch. Makes {b} batches, sells {c}. Left?",
         lambda a, b, c: a * b - c, lambda a, b, c: f"{a} * {b} - {c} ="),

        # has X, gets Y, gives Z
        (lambda a, b, c: f"{random.choice(names)} has {a} marbles. Gets {b} more, gives away {c}. How many?",
         lambda a, b, c: a + b - c, lambda a, b, c: f"{a} + {b} - {c} ="),

        # split into groups, each group gets items
        (lambda a, b, c: f"{a} students split into groups of {b}. Each group gets {c} books. Total books?",
         lambda a, b, c: (a // b) * c, lambda a, b, c: f"{a} / {b} * {c} ="),
    ]

    # Type 3: Three operations / nested
    templates_three = [
        # X containers, Y items each, Z items each
        (lambda a, b, c: f"{a} boxes. Each has {b} bags. Each bag has {c} marbles. Total marbles?",
         lambda a, b, c: a * b * c, lambda a, b, c: f"{a} * {b} * {c} ="),

        # earn rate, hours day1, hours day2
        (lambda a, b, c: f"{random.choice(names)} earns ${a}/hour. Works {b} hours Monday, {c} hours Tuesday. Total?",
         lambda a, b, c: a * b + a * c, lambda a, b, c: f"{a} * {b} + {a} * {c} ="),

        # start, subtract, subtract, add
        (lambda a, b, c, d: f"{random.choice(names)} has ${a}. Spends ${b}, then ${c}, earns ${d}. How much?",
         lambda a, b, c, d: a - b - c + d, lambda a, b, c, d: f"{a} - {b} - {c} + {d} ="),
    ]

    # Type 4: Complex (GSM8K style)
    templates_complex = [
        # boxes with items, give some to each friend
        (lambda a, b, c, d: f"{random.choice(names)} has {a} boxes with {b} pencils each. Gives {c} pencils to each of {d} friends. Left?",
         lambda a, b, c, d: a * b - c * d, lambda a, b, c, d: f"{a} * {b} - {c} * {d} ="),

        # rows × seats - parked
        (lambda a, b, c: f"Parking lot: {a} rows, {b} spaces each. {c} cars parked. Empty spaces?",
         lambda a, b, c: a * b - c, lambda a, b, c: f"{a} * {b} - {c} ="),

        # items in boxes, sell boxes at price
        (lambda a, b, c: f"{a} cookies in boxes of {b}. Each box sells for ${c}. Total money if all sold?",
         lambda a, b, c: (a // b) * c, lambda a, b, c: f"{a} / {b} * {c} ="),

        # cars × seats - passengers
        (lambda a, b, c: f"A train: {a} cars, {b} seats each. {c} passengers board. Empty seats?",
         lambda a, b, c: a * b - c, lambda a, b, c: f"{a} * {b} - {c} ="),
    ]

    # Generate simple (40%)
    for _ in range(int(n * 0.4)):
        template_q, compute, template_a = random.choice(templates_simple)
        if "split" in template_q(10, 5).lower() or "shared" in template_q(10, 5).lower():
            b = random.randint(2, 10)
            a = b * random.randint(2, 15)
        else:
            a = random.randint(5, 100)
            b = random.randint(2, 20)
        problems.append({
            "question": template_q(a, b),
            "answer": compute(a, b),
            "expression": template_a(a, b),
            "type": "simple"
        })

    # Generate two-op (30%)
    for _ in range(int(n * 0.3)):
        template_q, compute, template_a = random.choice(templates_two)
        a = random.randint(20, 100)
        b = random.randint(2, 15)
        c = random.randint(2, 20)
        if "/" in template_a(a, b, c):
            b = random.randint(2, 10)
            a = b * random.randint(3, 15)
        problems.append({
            "question": template_q(a, b, c),
            "answer": compute(a, b, c),
            "expression": template_a(a, b, c),
            "type": "two_op"
        })

    # Generate three-op (15%)
    for _ in range(int(n * 0.15)):
        template_q, compute, template_a = random.choice(templates_three)
        a = random.randint(2, 20)
        b = random.randint(2, 15)
        c = random.randint(2, 15)
        d = random.randint(5, 30)
        try:
            q = template_q(a, b, c, d)
            ans = compute(a, b, c, d)
            expr = template_a(a, b, c, d)
        except TypeError:  # 3-arg template
            q = template_q(a, b, c)
            ans = compute(a, b, c)
            expr = template_a(a, b, c)
        problems.append({
            "question": q,
            "answer": ans,
            "expression": expr,
            "type": "three_op"
        })

    # Generate complex (15%)
    for _ in range(int(n * 0.15)):
        template_q, compute, template_a = random.choice(templates_complex)
        a = random.randint(3, 20)
        b = random.randint(3, 15)
        c = random.randint(2, 50)
        d = random.randint(2, 10)

        # Check arity
        try:
            # Try 4-arg first
            q = template_q(a, b, c, d)
            ans = compute(a, b, c, d)
            expr = template_a(a, b, c, d)
        except TypeError:
            # Fall back to 3-arg
            if "/" in template_a(a, b, c):
                b = random.randint(2, 10)
                a = b * random.randint(2, 15)
            q = template_q(a, b, c)
            ans = compute(a, b, c)
            expr = template_a(a, b, c)
        problems.append({
            "question": q,
            "answer": ans,
            "expression": expr,
            "type": "complex"
        })

    random.shuffle(problems)
    return problems


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("  GENERATING VERIFIED SFT TRAINING DATA")
    print("=" * 70)

    # Initialize WASM runtime for verification
    runtime = WASMRuntime()

    # Generate problems with known answers
    print("\nGenerating problems with computed answers...")
    problems = generate_problems_with_answers(5000)
    print(f"Generated {len(problems)} problems")

    # Verify each problem's expression executes correctly
    print("\nVerifying expressions via WASM execution...")
    verified = []
    failed = []

    for p in problems:
        if verify_expression(p["expression"], p["answer"], runtime):
            verified.append(p)
        else:
            failed.append(p)

    print(f"  Verified: {len(verified)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\n  Sample failures:")
        for f in failed[:3]:
            print(f"    {f['expression']} expected {f['answer']}")

    # Split into train/val/test
    random.shuffle(verified)
    n = len(verified)
    train = verified[:int(n * 0.8)]
    val = verified[int(n * 0.8):int(n * 0.9)]
    test = verified[int(n * 0.9):]

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save datasets
    output_dir = Path(__file__).parent / "sft_data"
    output_dir.mkdir(exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps({
                    "question": item["question"],
                    "expression": item["expression"],
                    "answer": item["answer"],
                    "type": item["type"]
                }) + "\n")
        print(f"Saved {path}")

    # Show examples
    print("\n" + "=" * 70)
    print("  SAMPLE TRAINING DATA")
    print("=" * 70)

    for i, item in enumerate(train[:10]):
        print(f"\n[{item['type']}]")
        print(f"  Q: {item['question']}")
        print(f"  A: {item['expression']}")
        print(f"  → {item['answer']}")

    # Summary for training
    print("\n" + "=" * 70)
    print("  TRAINING DATA READY")
    print("=" * 70)
    print(f"""
    Files created in {output_dir}:
    - train.jsonl ({len(train)} examples)
    - val.jsonl ({len(val)} examples)
    - test.jsonl ({len(test)} examples)

    Format:
    {{
        "question": "Tom has 3 boxes with 12 pencils each...",
        "expression": "3 * 12 - 5 * 2 =",
        "answer": 26,
        "type": "complex"
    }}

    All expressions VERIFIED via WASM execution.

    Next steps:
    1. SFT: Train model on (question → expression) pairs
    2. RL: Fine-tune with reward = parse_success × execution_correct
    """)


if __name__ == "__main__":
    main()
