#!/usr/bin/env python3
"""
Test MoE Expert Compression on GPT-OSS 20B

This script tests expert compression on a real MoE model.

Usage:
    uv run python examples/introspection/experiments/moe/test_gptoss_compression.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    detect_moe_architecture,
    get_moe_layer_info,
)


def main():
    print("=" * 70)
    print("MoE Expert Compression Test - GPT-OSS 20B")
    print("=" * 70)

    # Load GPT-OSS
    print("\nLoading GPT-OSS 20B (may take a minute)...")
    # Use local cached model path
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, tokenizer = load(str(model_path))

    # Check architecture detection
    arch = detect_moe_architecture(model)
    print(f"\nDetected architecture: {arch.value}")

    # Get MoE layer info using MoEHooks
    print("\n" + "-" * 70)
    print("MoE Layer Info")
    print("-" * 70)

    hooks = MoEHooks(model)
    moe_layers = hooks.moe_layer_indices

    if not moe_layers:
        print("  No MoE layers detected!")
        return

    print(f"  Found {len(moe_layers)} MoE layers")

    # Show info for a few layers
    for layer_idx in moe_layers[:3]:
        info = get_moe_layer_info(model, layer_idx)
        if info:
            print(f"  Layer {layer_idx}: {info.num_experts} experts, {info.num_experts_per_tok} active")

    # Create compressor
    print("\n" + "-" * 70)
    print("Compression Analysis")
    print("-" * 70)

    compressor = ExpertCompressor(model, tokenizer)

    # Analyze middle layer
    layer_idx = moe_layers[len(moe_layers) // 2]
    print(f"\nAnalyzing layer {layer_idx}...")

    # Use diverse test prompts
    test_prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "import numpy as np",
        "Once upon a time",
        "SELECT * FROM users WHERE",
        "Hello, how are you?",
    ]

    try:
        analysis = compressor.analyze_compression_potential(layer_idx, test_prompts)

        print(f"\n  Analysis Results:")
        print(f"    Number of experts: {analysis.get('num_experts', 'N/A')}")

        if "merge_candidates" in analysis and analysis["merge_candidates"]:
            print(f"\n  Merge Candidates (most similar pairs):")
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

        print(f"\n  Compression Potential:")
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

        # Target-based compression (halving experts)
        info = get_moe_layer_info(model, layer_idx)
        if info:
            target = info.num_experts // 2
            print(f"\n  Target-based compression (to {target} experts):")
            try:
                plan = compressor.plan_compression(layer_idx, target_experts=target)
                print(f"    Original: {plan.original_num_experts} -> Target: {plan.target_num_experts}")
                if plan.merges:
                    print(f"    Merges:")
                    for m in plan.merges[:5]:
                        print(f"      {m.source_experts} -> Expert {m.target_expert} (blend: {m.weight_blend})")
                print(f"    Pruned: {plan.pruned_experts}")
                print(f"    Kept (sample): {plan.kept_experts[:10]}...")
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
