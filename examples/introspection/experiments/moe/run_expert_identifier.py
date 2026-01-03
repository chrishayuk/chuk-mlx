#!/usr/bin/env python3
"""
Run the built-in ExpertIdentifier on GPT-OSS

Usage:
    uv run python examples/introspection/experiments/moe/run_expert_identifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from mlx_lm import load

from chuk_lazarus.introspection.moe import ExpertIdentifier


def main():
    print("=" * 70)
    print("Expert Identifier - Built-in Analysis Tool")
    print("=" * 70)

    # Load
    print("\nLoading GPT-OSS...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, tokenizer = load(str(model_path))

    # Create identifier
    print("\nCreating ExpertIdentifier...")
    identifier = ExpertIdentifier(model, tokenizer)

    # Analyze layer 12 (middle layer)
    print("\nAnalyzing layer 12...")
    result = identifier.identify_all_experts(layer_idx=12)

    # Print summary
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)

    # Show redundant pairs
    if result.redundant_pairs:
        print("\n\nREDUNDANT EXPERT PAIRS (High Similarity):")
        print("-" * 50)
        for e1, e2, sim in result.redundant_pairs[:10]:
            print(f"  Experts {e1:2d} & {e2:2d}: {sim:.1%} similar")
        print(f"\n  Total redundant pairs: {len(result.redundant_pairs)}")

    # Show categories
    print("\n\nEXPERT CATEGORIES:")
    print("-" * 50)
    for category, experts in sorted(result.category_experts.items()):
        if experts:
            print(f"  {category}: {len(experts)} experts -> {experts[:5]}{'...' if len(experts) > 5 else ''}")

    # Show specialists vs generalists
    print("\n\nSPECIALISTS vs GENERALISTS:")
    print("-" * 50)
    print(f"  Specialists (focused): {len(result.specialist_experts)} experts")
    print(f"  Generalists (diverse): {len(result.generalist_experts)} experts")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
