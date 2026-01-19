#!/usr/bin/env python3
"""
Test MoE Expert Compression

This script tests the expert compression functionality:
1. Analyze compression potential (similarity, utilization)
2. Plan compression strategies
3. Show memory savings estimates

Usage:
    uv run python examples/introspection/experiments/moe/test_compression.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))


# Import test model from routing analysis
from moe_routing_analysis import create_test_moe_model

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    detect_moe_architecture,
    get_moe_layer_info,
)


def main():
    print("=" * 70)
    print("MoE Expert Compression Test")
    print("=" * 70)

    # Create test MoE model
    print("\nCreating test MoE model...")
    model = create_test_moe_model(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_experts=8,
        num_experts_per_tok=2,
    )

    print("\nTest MoE Model created:")
    print("  - 8 experts, 2 active per token")
    print("  - 4 layers")
    print("  - hidden_size=128")

    # Check architecture detection
    arch = detect_moe_architecture(model)
    print(f"\nDetected architecture: {arch.value}")

    # Create simple tokenizer
    class SimpleTokenizer:
        def encode(self, text):
            return [ord(c) % 1000 for c in text]

        def decode(self, ids):
            return "".join(chr(i % 128 + 32) for i in ids)

    tokenizer = SimpleTokenizer()

    # Get MoE layer info using MoEHooks
    print("\n" + "-" * 70)
    print("MoE Layer Info")
    print("-" * 70)

    hooks = MoEHooks(model)
    moe_layers = hooks.moe_layer_indices

    if not moe_layers:
        print("  No MoE layers detected!")
        return

    for layer_idx in moe_layers:
        info = get_moe_layer_info(model, layer_idx)
        if info:
            print(
                f"  Layer {layer_idx}: {info.num_experts} experts, {info.num_experts_per_tok} active"
            )

    # Create compressor
    print("\n" + "-" * 70)
    print("Compression Analysis")
    print("-" * 70)

    compressor = ExpertCompressor(model, tokenizer)

    # Analyze first MoE layer
    layer_idx = moe_layers[0]
    print(f"\nAnalyzing layer {layer_idx}...")

    try:
        analysis = compressor.analyze_compression_potential(layer_idx)

        print("\n  Analysis Results:")
        print(f"    Number of experts: {analysis.get('num_experts', 'N/A')}")

        if "merge_candidates" in analysis and analysis["merge_candidates"]:
            print("\n  Merge Candidates (most similar pairs):")
            for e1, e2, sim in analysis["merge_candidates"][:5]:
                print(f"    Experts {e1} & {e2}: {sim:.2%} similar")
        else:
            print("\n  No merge candidates found (no experts >60% similar)")

        if "prune_candidates" in analysis:
            print(f"\n  Prune Candidates (low utilization): {analysis['prune_candidates']}")

        if "specialist_experts" in analysis:
            print(f"  Specialist Experts: {analysis['specialist_experts']}")

        if "generalist_experts" in analysis:
            print(f"  Generalist Experts: {analysis['generalist_experts']}")

        if "mergeable_groups" in analysis:
            print(f"  Mergeable Groups: {analysis['mergeable_groups']}")

        print("\n  Compression Potential:")
        print(f"    Potential reduction: {analysis.get('potential_reduction', 0)} experts")
        print(f"    Max compression ratio: {analysis.get('max_compression_ratio', 1):.1%}")
        print(f"    Recommended target: {analysis.get('recommended_target', 'N/A')} experts")

        # Plan compression
        print("\n" + "-" * 70)
        print("Compression Planning")
        print("-" * 70)

        # Try different strategies
        for strategy in ["balanced", "aggressive", "conservative"]:
            print(f"\n  Strategy: {strategy}")
            try:
                plan = compressor.plan_compression(layer_idx, strategy=strategy)
                print(f"    Original experts: {plan.original_num_experts}")
                print(f"    Target experts:   {plan.target_num_experts}")
                print(f"    Merges planned:   {len(plan.merges)}")
                print(f"    Experts pruned:   {len(plan.pruned_experts)}")
                print(f"    Memory reduction: {plan.estimated_memory_reduction:.1%}")
                print(f"    Quality impact:   {plan.estimated_quality_impact}")
            except Exception as e:
                print(f"    Error: {e}")

        # Target-based compression
        print("\n  Target-based compression (to 4 experts):")
        try:
            plan = compressor.plan_compression(layer_idx, target_experts=4)
            print(f"    Original: {plan.original_num_experts} -> Target: {plan.target_num_experts}")
            print(f"    Merges: {[(m.source_experts, m.target_expert) for m in plan.merges]}")
            print(f"    Pruned: {plan.pruned_experts}")
            print(f"    Kept: {plan.kept_experts}")
        except Exception as e:
            print(f"    Error: {e}")

    except Exception as e:
        print(f"  Error analyzing layer: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Compression test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
