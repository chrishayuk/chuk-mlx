#!/usr/bin/env python3
"""
Generate training data for multi-op chains.

Examples:
  "16 - 3, then multiply by 5" →
    [i32.const 16, i32.const 3, i32.sub, i32.const 5, i32.mul]

  Result stored in local.0 between operations.
"""

import json
import random
from pathlib import Path

# Canonical multi-op format: "a op b = intermediate; intermediate op c = "
# We train to parse sequences

CHAIN_TEMPLATES = [
    # Two-op chains
    {
        "pattern": "{a} + {b}, then subtract {c}",
        "ops": ["add", "sub"],
        "compute": lambda a, b, c: (a + b) - c,
    },
    {
        "pattern": "{a} - {b}, then add {c}",
        "ops": ["sub", "add"],
        "compute": lambda a, b, c: (a - b) + c,
    },
    {
        "pattern": "{a} * {b}, then add {c}",
        "ops": ["mul", "add"],
        "compute": lambda a, b, c: (a * b) + c,
    },
    {
        "pattern": "{a} + {b}, then multiply by {c}",
        "ops": ["add", "mul"],
        "compute": lambda a, b, c: (a + b) * c,
    },
    {
        "pattern": "{a} - {b}, then multiply by {c}",
        "ops": ["sub", "mul"],
        "compute": lambda a, b, c: (a - b) * c,
    },
    {
        "pattern": "{a} * {b}, then subtract {c}",
        "ops": ["mul", "sub"],
        "compute": lambda a, b, c: (a * b) - c,
    },
    {
        "pattern": "({a} + {b}) * {c}",
        "ops": ["add", "mul"],
        "compute": lambda a, b, c: (a + b) * c,
    },
    {
        "pattern": "({a} - {b}) * {c}",
        "ops": ["sub", "mul"],
        "compute": lambda a, b, c: (a - b) * c,
    },
    {
        "pattern": "{a} * {b} + {c}",
        "ops": ["mul", "add"],
        "compute": lambda a, b, c: a * b + c,
    },
    {
        "pattern": "{a} * {b} - {c}",
        "ops": ["mul", "sub"],
        "compute": lambda a, b, c: a * b - c,
    },
]

# NL variations
NL_CHAIN_TEMPLATES = [
    {
        "pattern": "Start with {a}, add {b}, then subtract {c}",
        "ops": ["add", "sub"],
        "compute": lambda a, b, c: (a + b) - c,
    },
    {
        "pattern": "Take {a}, subtract {b}, then multiply by {c}",
        "ops": ["sub", "mul"],
        "compute": lambda a, b, c: (a - b) * c,
    },
    {
        "pattern": "Add {a} and {b}, then multiply the result by {c}",
        "ops": ["add", "mul"],
        "compute": lambda a, b, c: (a + b) * c,
    },
    {
        "pattern": "Multiply {a} by {b}, then add {c}",
        "ops": ["mul", "add"],
        "compute": lambda a, b, c: (a * b) + c,
    },
    {
        "pattern": "{a} eggs daily for {b} days, sell {c}",
        "ops": ["mul", "sub"],
        "compute": lambda a, b, c: (a * b) - c,
    },
    {
        "pattern": "Buy {a} items at ${b} each, with ${c} discount",
        "ops": ["mul", "sub"],
        "compute": lambda a, b, c: (a * b) - c,
    },
    {
        "pattern": "{a} boxes with {b} items each, plus {c} extra",
        "ops": ["mul", "add"],
        "compute": lambda a, b, c: (a * b) + c,
    },
]

OP_TO_IR = {
    "add": 16,  # I32_ADD
    "sub": 17,  # I32_SUB
    "mul": 18,  # I32_MUL
    "div": 19,  # I32_DIV_S
}

# IR opcodes
START = 1
END = 2
SLOT_0 = 3
SLOT_1 = 4
SLOT_2 = 5


def generate_ir_sequence(ops: list[str]) -> list[int]:
    """Generate IR sequence for multi-op chain.

    For two ops: [START, SLOT_0, SLOT_1, OP1, SLOT_2, OP2, END]
    """
    ir = [START, SLOT_0, SLOT_1, OP_TO_IR[ops[0]]]

    if len(ops) > 1:
        ir.extend([SLOT_2, OP_TO_IR[ops[1]]])

    ir.append(END)
    return ir


def generate_canonical(a: int, b: int, c: int, ops: list[str]) -> str:
    """Generate canonical form showing intermediate step."""
    op_symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

    op1 = op_symbols[ops[0]]
    op2 = op_symbols[ops[1]] if len(ops) > 1 else ""

    if len(ops) == 1:
        return f"{a} {op1} {b} = "
    else:
        return f"({a} {op1} {b}) {op2} {c} = "


def generate_sample(use_nl: bool = False) -> dict:
    """Generate a multi-op training sample."""
    # Pick template
    if use_nl:
        templates = NL_CHAIN_TEMPLATES
    else:
        templates = CHAIN_TEMPLATES

    template = random.choice(templates)

    # Generate operands (keep small to avoid overflow)
    a = random.randint(1, 50)
    b = random.randint(1, 30)
    c = random.randint(1, 20)

    # Compute result
    try:
        result = template["compute"](a, b, c)
    except:
        result = 0

    nl_input = template["pattern"].format(a=a, b=b, c=c)
    canonical = generate_canonical(a, b, c, template["ops"])
    ir_sequence = generate_ir_sequence(template["ops"])

    return {
        "nl_input": nl_input,
        "canonical_output": canonical,
        "operands": [a, b, c],
        "ops": template["ops"],
        "ir_sequence": ir_sequence,
        "expected_result": result,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--output-dir", default="experiments/ir_emission/data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for _ in range(args.num_samples):
        use_nl = random.random() < 0.4
        sample = generate_sample(use_nl)
        all_samples.append(sample)

    random.shuffle(all_samples)

    # Split
    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]

    # Write
    train_path = output_dir / "multiop_train.jsonl"
    val_path = output_dir / "multiop_val.jsonl"

    with open(train_path, "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")

    print(f"Train: {len(train)} → {train_path}")
    print(f"Val: {len(val)} → {val_path}")

    # Examples
    print("\nExamples:")
    for s in train[:8]:
        print(f"  {s['nl_input'][:45]:45} → {s['canonical_output']}")
        print(f"    IR: {s['ir_sequence']} = {s['expected_result']}")


if __name__ == "__main__":
    main()
