#!/usr/bin/env python3
"""Generate synthetic training data for extended CoT format."""

import json
import random
import sys
from pathlib import Path

# Add generators to path
sys.path.insert(0, str(Path(__file__).parent))

from generators import entity_track, arithmetic, rate_equation, comparison, percentage


def generate_all(
    n_entity: int = 100,
    n_arithmetic: int = 40,
    n_rate: int = 40,
    n_comparison: int = 40,
    n_percentage: int = 15,
) -> list[dict]:
    """Generate balanced training data."""
    data = []

    print(f"Generating synthetic training data...")
    print(f"  entity_track: {n_entity}")
    print(f"  arithmetic: {n_arithmetic}")
    print(f"  rate_equation: {n_rate}")
    print(f"  comparison: {n_comparison}")
    print(f"  percentage: {n_percentage}")

    data.extend(entity_track.generate(n_entity))
    data.extend(arithmetic.generate(n_arithmetic))
    data.extend(rate_equation.generate(n_rate))
    data.extend(comparison.generate(n_comparison))
    data.extend(percentage.generate(n_percentage))

    random.shuffle(data)

    print(f"  Total: {len(data)} examples")
    return data


def format_yaml(example: dict) -> str:
    """Format example as YAML string."""
    import yaml
    return yaml.dump({
        "expert": example["expert"],
        "trace": example["trace"],
        "answer": example["answer"],
    }, default_flow_style=None, sort_keys=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_train.json")
    parser.add_argument("--n-entity", type=int, default=100)
    parser.add_argument("--n-arithmetic", type=int, default=40)
    parser.add_argument("--n-rate", type=int, default=40)
    parser.add_argument("--n-comparison", type=int, default=40)
    parser.add_argument("--n-percentage", type=int, default=15)
    args = parser.parse_args()

    data = generate_all(
        n_entity=args.n_entity,
        n_arithmetic=args.n_arithmetic,
        n_rate=args.n_rate,
        n_comparison=args.n_comparison,
        n_percentage=args.n_percentage,
    )

    # Save
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show examples
    print("\n" + "=" * 60)
    print("Sample examples:")
    print("=" * 60)

    for expert_type in ["entity_track", "arithmetic", "rate_equation", "comparison", "percentage"]:
        examples = [e for e in data if e["expert"] == expert_type]
        if examples:
            ex = examples[0]
            print(f"\n[{expert_type}]")
            print(f"Q: {ex['question']}")
            print(f"YAML:\n{format_yaml(ex)}")


if __name__ == "__main__":
    main()
