#!/usr/bin/env python3
"""Generate arithmetic training data for classifier emergence experiments.

This creates SFT-format JSONL data with arithmetic problems labeled by operation type.

Usage:
    python experiments/cli_classifier_emergence/generate_data.py --output data/arithmetic_sft.jsonl --samples 1000
"""

import argparse
import json
import random
from pathlib import Path


def generate_arithmetic_sample():
    """Generate a single arithmetic sample with operation label."""
    ops = [
        ('*', 'multiply', lambda a, b: a * b),
        ('+', 'add', lambda a, b: a + b),
        ('-', 'subtract', lambda a, b: a - b),
        ('/', 'divide', lambda a, b: a // b if b != 0 else 0),
    ]

    op_sym, op_name, op_fn = random.choice(ops)

    if op_sym == '/':
        b = random.randint(1, 12)
        a = b * random.randint(1, 12)
    elif op_sym == '-':
        a = random.randint(10, 100)
        b = random.randint(1, a)
    else:
        a = random.randint(1, 50)
        b = random.randint(1, 50)

    result = op_fn(a, b)
    prompt = f"{a} {op_sym} {b} = "
    answer = str(result)

    return {
        "prompt": prompt,
        "response": answer,
        "operation": op_name,
        # For dual-reward training, we include the classification target
        "classification_target": op_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic training data")
    parser.add_argument("--output", "-o", default="data/arithmetic_sft.jsonl", help="Output JSONL file")
    parser.add_argument("--samples", "-n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [generate_arithmetic_sample() for _ in range(args.samples)]

    # Count by operation
    op_counts = {}
    for s in samples:
        op = s["operation"]
        op_counts[op] = op_counts.get(op, 0) + 1

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(samples)} samples to {output_path}")
    print(f"Distribution: {op_counts}")


if __name__ == "__main__":
    main()
