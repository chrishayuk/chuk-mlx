#!/usr/bin/env python3
"""
Generate training data for IR emission.

Creates datasets mapping NL prompts to IR sequences:
- Phase 1: Single-op arithmetic (3 + 4)
- Phase 2: Multi-op chains (3 + 4 - 2)
- Phase 3: Word problems (Janet's ducks...)
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from codebook import IROpcode


@dataclass
class IRSample:
    """A training sample for IR emission."""

    prompt: str                    # Natural language input
    ir_sequence: list[int]         # Target IR (codebook indices)
    operands: list[int]            # Numbers extracted from prompt
    expected_result: int           # Ground truth result
    phase: int                     # Training phase (1, 2, 3, 4)
    operation: Optional[str] = None  # For phase 1: add/sub/mul/div

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "ir_sequence": self.ir_sequence,
            "operands": self.operands,
            "expected_result": self.expected_result,
            "phase": self.phase,
            "operation": self.operation,
        }


# Templates for NL prompts
SINGLE_OP_TEMPLATES = {
    "add": [
        "{a} + {b} = ",
        "What is {a} plus {b}?",
        "Calculate {a} + {b}",
        "Add {a} and {b}",
        "{a} added to {b} equals",
        "The sum of {a} and {b} is",
    ],
    "sub": [
        "{a} - {b} = ",
        "What is {a} minus {b}?",
        "Calculate {a} - {b}",
        "Subtract {b} from {a}",
        "{a} take away {b} equals",
        "The difference of {a} and {b} is",
    ],
    "mul": [
        "{a} * {b} = ",
        "{a} x {b} = ",
        "What is {a} times {b}?",
        "Calculate {a} * {b}",
        "Multiply {a} by {b}",
        "{a} multiplied by {b} equals",
        "The product of {a} and {b} is",
    ],
    "div": [
        "{a} / {b} = ",
        "What is {a} divided by {b}?",
        "Calculate {a} / {b}",
        "Divide {a} by {b}",
        "{a} divided by {b} equals",
    ],
}

MULTI_OP_TEMPLATES = [
    "{a} + {b} - {c} = ",
    "{a} - {b} + {c} = ",
    "{a} * {b} + {c} = ",
    "{a} + {b} * {c} = ",  # Note: we do left-to-right, not PEMDAS
    "({a} + {b}) * {c} = ",
    "({a} - {b}) * {c} = ",
    "{a} * {b} - {c} = ",
]

WORD_PROBLEM_TEMPLATES = [
    # Addition
    (
        "{name} has {a} apples. {name2} gives {pronoun} {b} more. How many apples does {name} have?",
        "add",
        lambda a, b: a + b,
    ),
    (
        "There are {a} birds in a tree. {b} more birds land. How many birds are there now?",
        "add",
        lambda a, b: a + b,
    ),
    # Subtraction
    (
        "{name} has {a} cookies. {name} eats {b}. How many cookies are left?",
        "sub",
        lambda a, b: a - b,
    ),
    (
        "A store has {a} items. {b} are sold. How many items remain?",
        "sub",
        lambda a, b: a - b,
    ),
    # Multiplication
    (
        "{name} has {a} bags with {b} marbles each. How many marbles in total?",
        "mul",
        lambda a, b: a * b,
    ),
    (
        "There are {a} rows of {b} chairs. How many chairs are there?",
        "mul",
        lambda a, b: a * b,
    ),
    # Multi-step (Janet's eggs style)
    (
        "{name}'s ducks lay {a} eggs daily. {name} eats {b} for breakfast and bakes {c} into muffins. {name} sells the rest at ${d} each. How many eggs does {name} sell daily?",
        "multi_sub",
        lambda a, b, c, d: a - b - c,  # d is price, not used in egg count
    ),
]

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack"]
NAMES2 = ["Sam", "Pat", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Quinn"]
PRONOUNS = {"Alice": "her", "Bob": "him", "Carol": "her", "David": "him",
            "Emma": "her", "Frank": "him", "Grace": "her", "Henry": "him",
            "Ivy": "her", "Jack": "him"}


def generate_phase1_samples(n: int = 1000, seed: int = 42) -> list[IRSample]:
    """Generate single-operation arithmetic samples."""
    random.seed(seed)
    samples = []

    ops = ["add", "sub", "mul", "div"]
    op_to_ir = {
        "add": IROpcode.I32_ADD,
        "sub": IROpcode.I32_SUB,
        "mul": IROpcode.I32_MUL,
        "div": IROpcode.I32_DIV_S,
    }
    op_to_func = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a // b,
    }

    for _ in range(n):
        op = random.choice(ops)

        # Generate operands
        if op == "div":
            # Ensure clean division
            b = random.randint(1, 12)
            a = b * random.randint(1, 10)
        elif op == "mul":
            a = random.randint(1, 20)
            b = random.randint(1, 20)
        else:
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            if op == "sub" and a < b:
                a, b = b, a  # Keep positive

        # Generate prompt
        template = random.choice(SINGLE_OP_TEMPLATES[op])
        prompt = template.format(a=a, b=b)

        # Generate IR
        ir_sequence = [
            IROpcode.START,
            IROpcode.SLOT_0,
            IROpcode.SLOT_1,
            op_to_ir[op],
            IROpcode.END,
        ]

        result = op_to_func[op](a, b)

        samples.append(IRSample(
            prompt=prompt,
            ir_sequence=ir_sequence,
            operands=[a, b],
            expected_result=result,
            phase=1,
            operation=op,
        ))

    return samples


def generate_phase2_samples(n: int = 500, seed: int = 42) -> list[IRSample]:
    """Generate multi-operation chain samples."""
    random.seed(seed)
    samples = []

    for _ in range(n):
        template = random.choice(MULTI_OP_TEMPLATES)

        a = random.randint(1, 50)
        b = random.randint(1, 50)
        c = random.randint(1, 20)

        prompt = template.format(a=a, b=b, c=c)

        # Parse template to determine ops (simplified: look at operators)
        if "+ {b} -" in template or "+ {b} - " in template:
            # a + b - c
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_ADD,
                IROpcode.SLOT_2, IROpcode.I32_SUB,
                IROpcode.END,
            ]
            result = a + b - c
        elif "- {b} +" in template:
            # a - b + c
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_SUB,
                IROpcode.SLOT_2, IROpcode.I32_ADD,
                IROpcode.END,
            ]
            result = a - b + c
        elif "* {b} +" in template:
            # a * b + c
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_MUL,
                IROpcode.SLOT_2, IROpcode.I32_ADD,
                IROpcode.END,
            ]
            result = a * b + c
        elif "* {b} -" in template:
            # a * b - c
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_MUL,
                IROpcode.SLOT_2, IROpcode.I32_SUB,
                IROpcode.END,
            ]
            result = a * b - c
        elif "+ {b}) *" in template:
            # (a + b) * c
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_ADD,
                IROpcode.SLOT_2, IROpcode.I32_MUL,
                IROpcode.END,
            ]
            result = (a + b) * c
        elif "- {b}) *" in template:
            # (a - b) * c
            if a < b:
                a, b = b, a
            prompt = template.format(a=a, b=b, c=c)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_SUB,
                IROpcode.SLOT_2, IROpcode.I32_MUL,
                IROpcode.END,
            ]
            result = (a - b) * c
        else:
            # a + b * c (left to right, not PEMDAS)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_ADD,
                IROpcode.SLOT_2, IROpcode.I32_MUL,
                IROpcode.END,
            ]
            result = (a + b) * c

        samples.append(IRSample(
            prompt=prompt,
            ir_sequence=ir_sequence,
            operands=[a, b, c],
            expected_result=result,
            phase=2,
        ))

    return samples


def generate_phase3_samples(n: int = 500, seed: int = 42) -> list[IRSample]:
    """Generate word problem samples."""
    random.seed(seed)
    samples = []

    for _ in range(n):
        template, op_type, compute = random.choice(WORD_PROBLEM_TEMPLATES)
        name = random.choice(NAMES)
        name2 = random.choice(NAMES2)
        pronoun = PRONOUNS.get(name, "them")

        if op_type == "add":
            a = random.randint(5, 50)
            b = random.randint(1, 30)
            prompt = template.format(name=name, name2=name2, pronoun=pronoun, a=a, b=b)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_ADD,
                IROpcode.END,
            ]
            result = compute(a, b)
            operands = [a, b]

        elif op_type == "sub":
            a = random.randint(20, 100)
            b = random.randint(1, a - 1)
            prompt = template.format(name=name, name2=name2, pronoun=pronoun, a=a, b=b)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_SUB,
                IROpcode.END,
            ]
            result = compute(a, b)
            operands = [a, b]

        elif op_type == "mul":
            a = random.randint(2, 12)
            b = random.randint(2, 12)
            prompt = template.format(name=name, name2=name2, pronoun=pronoun, a=a, b=b)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_MUL,
                IROpcode.END,
            ]
            result = compute(a, b)
            operands = [a, b]

        elif op_type == "multi_sub":
            # Janet's eggs style
            a = random.randint(10, 30)  # eggs laid
            b = random.randint(1, 5)    # eaten
            c = random.randint(1, 5)    # baked
            d = random.randint(1, 5)    # price (not used in count)

            if a <= b + c:
                a = b + c + random.randint(5, 15)

            prompt = template.format(name=name, a=a, b=b, c=c, d=d)
            ir_sequence = [
                IROpcode.START,
                IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_SUB,
                IROpcode.SLOT_2, IROpcode.I32_SUB,
                IROpcode.END,
            ]
            result = compute(a, b, c, d)
            operands = [a, b, c]

        else:
            continue

        samples.append(IRSample(
            prompt=prompt,
            ir_sequence=ir_sequence,
            operands=operands,
            expected_result=result,
            phase=3,
            operation=op_type,
        ))

    return samples


def save_dataset(samples: list[IRSample], path: Path) -> None:
    """Save samples as JSONL."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate IR emission training data")
    parser.add_argument("--output-dir", "-o", default="experiments/ir_emission/data")
    parser.add_argument("--phase1-samples", type=int, default=2000)
    parser.add_argument("--phase2-samples", type=int, default=1000)
    parser.add_argument("--phase3-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Phase 1 (single-op) samples...")
    phase1 = generate_phase1_samples(args.phase1_samples, args.seed)
    save_dataset(phase1, output_dir / "phase1_single_op.jsonl")
    print(f"  Saved {len(phase1)} samples to phase1_single_op.jsonl")

    # Split for train/test
    random.seed(args.seed)
    random.shuffle(phase1)
    split = int(len(phase1) * 0.9)
    save_dataset(phase1[:split], output_dir / "phase1_train.jsonl")
    save_dataset(phase1[split:], output_dir / "phase1_test.jsonl")
    print(f"  Train: {split}, Test: {len(phase1) - split}")

    print("\nGenerating Phase 2 (multi-op) samples...")
    phase2 = generate_phase2_samples(args.phase2_samples, args.seed + 1)
    save_dataset(phase2, output_dir / "phase2_multi_op.jsonl")
    print(f"  Saved {len(phase2)} samples to phase2_multi_op.jsonl")

    print("\nGenerating Phase 3 (word problems) samples...")
    phase3 = generate_phase3_samples(args.phase3_samples, args.seed + 2)
    save_dataset(phase3, output_dir / "phase3_word_problems.jsonl")
    print(f"  Saved {len(phase3)} samples to phase3_word_problems.jsonl")

    # Combined dataset
    all_samples = phase1 + phase2 + phase3
    random.shuffle(all_samples)
    save_dataset(all_samples, output_dir / "all_phases.jsonl")
    print(f"\nTotal: {len(all_samples)} samples saved to all_phases.jsonl")

    # Print statistics
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for phase, samples in [(1, phase1), (2, phase2), (3, phase3)]:
        print(f"\nPhase {phase}:")
        if phase == 1:
            ops = {}
            for s in samples:
                ops[s.operation] = ops.get(s.operation, 0) + 1
            for op, count in sorted(ops.items()):
                print(f"  {op}: {count}")
        else:
            print(f"  Total: {len(samples)}")

    print("\nSample examples:")
    for phase, samples in [(1, phase1), (2, phase2), (3, phase3)]:
        s = samples[0]
        print(f"\nPhase {phase}:")
        print(f"  Prompt: {s.prompt}")
        print(f"  IR: {[IROpcode(i).name for i in s.ir_sequence]}")
        print(f"  Operands: {s.operands}")
        print(f"  Result: {s.expected_result}")


if __name__ == "__main__":
    main()
