#!/usr/bin/env python3
"""
Test Aggressive MoE Compression - Target 50% Reduction

This script explores how aggressively we can compress GPT-OSS
while maintaining quality.

Usage:
    uv run python examples/introspection/experiments/moe/test_aggressive_compression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    estimate_model_size,
    estimate_compressed_size,
    print_compression_summary,
    get_moe_layer_info,
)


def main():
    print("=" * 70)
    print("Aggressive MoE Compression Test - Target 50% Reduction")
    print("=" * 70)

    # Load GPT-OSS
    print("\nLoading GPT-OSS 20B...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, tokenizer = load(str(model_path))

    # Baseline
    print("\n" + "-" * 70)
    print("BASELINE")
    print("-" * 70)
    baseline = estimate_model_size(model)
    print(f"  Total: {baseline['total']/1e9:.2f}B params")
    print(f"  Expert: {baseline['expert']/1e9:.2f}B ({baseline['expert']/baseline['total']*100:.1f}%)")

    # Get MoE info
    hooks = MoEHooks(model)
    moe_layers = hooks.moe_layer_indices
    print(f"\n  {len(moe_layers)} MoE layers, 32 experts each")

    compressor = ExpertCompressor(model, tokenizer)

    # Test different target reductions
    print("\n" + "-" * 70)
    print("COMPRESSION STRATEGIES COMPARISON")
    print("-" * 70)

    strategies = {
        "conservative": "conservative",
        "balanced": "balanced",
        "aggressive": "aggressive",
        "half (16 experts)": 16,
        "quarter (8 experts)": 8,
        "minimal (4 experts)": 4,
    }

    for name, strategy in strategies.items():
        print(f"\n  Strategy: {name}")
        plans = []

        for layer_idx in moe_layers:
            try:
                if isinstance(strategy, int):
                    plan = compressor.plan_compression(layer_idx, target_experts=strategy)
                else:
                    plan = compressor.plan_compression(layer_idx, strategy=strategy)
                plans.append(plan)
            except Exception as e:
                print(f"    Layer {layer_idx} error: {e}")

        if plans:
            stats = estimate_compressed_size(model, plans)
            orig_b = stats['original_params'] / 1e9
            comp_b = stats['compressed_params'] / 1e9
            reduction = stats['reduction_ratio'] * 100

            avg_experts = sum(p.target_num_experts for p in plans) / len(plans)
            min_experts = min(p.target_num_experts for p in plans)
            max_experts = max(p.target_num_experts for p in plans)

            print(f"    {orig_b:.2f}B → {comp_b:.2f}B ({reduction:.1f}% reduction)")
            print(f"    Experts: avg={avg_experts:.1f}, min={min_experts}, max={max_experts}")

    # Detailed analysis of aggressive target
    print("\n" + "-" * 70)
    print("DETAILED: HALVING EXPERTS (32 → 16)")
    print("-" * 70)

    plans_half = []
    for layer_idx in moe_layers:
        plan = compressor.plan_compression(layer_idx, target_experts=16)
        plans_half.append(plan)
        removed = plan.original_num_experts - plan.target_num_experts
        print(f"  Layer {layer_idx}: {plan.original_num_experts} → {plan.target_num_experts} "
              f"(merge={len(plan.merges)}, prune={len(plan.pruned_experts)})")

    print("\n")
    print_compression_summary(model, plans_half, "GPT-OSS (16 experts/layer)")

    # Even more aggressive - 8 experts
    print("\n" + "-" * 70)
    print("DETAILED: QUARTER EXPERTS (32 → 8)")
    print("-" * 70)

    plans_quarter = []
    for layer_idx in moe_layers:
        plan = compressor.plan_compression(layer_idx, target_experts=8)
        plans_quarter.append(plan)

    print_compression_summary(model, plans_quarter, "GPT-OSS (8 experts/layer)")

    # Calculate what we need for 50% total reduction
    print("\n" + "-" * 70)
    print("WHAT'S NEEDED FOR 50% REDUCTION?")
    print("-" * 70)

    total = baseline['total']
    expert = baseline['expert']
    non_expert = total - expert

    print(f"\n  Total params: {total/1e9:.2f}B")
    print(f"  Expert params: {expert/1e9:.2f}B ({expert/total*100:.1f}%)")
    print(f"  Non-expert params: {non_expert/1e9:.2f}B ({non_expert/total*100:.1f}%)")

    # To get 50% total reduction:
    target_total = total * 0.5
    expert_reduction_needed = total - target_total  # All must come from experts
    target_expert = expert - expert_reduction_needed

    print(f"\n  Target 50%: {target_total/1e9:.2f}B")
    print(f"  Expert params needed: {target_expert/1e9:.2f}B")
    print(f"  Expert reduction needed: {expert_reduction_needed/1e9:.2f}B ({expert_reduction_needed/expert*100:.1f}% of experts)")

    # What does that mean for expert count?
    params_per_expert_per_layer = expert / (len(moe_layers) * 32)
    experts_to_remove = expert_reduction_needed / params_per_expert_per_layer
    avg_experts_remaining = 32 - (experts_to_remove / len(moe_layers))

    print(f"\n  Params per expert: {params_per_expert_per_layer/1e6:.1f}M")
    print(f"  Experts to remove total: {experts_to_remove:.0f}")
    print(f"  Average experts per layer: {avg_experts_remaining:.1f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n  To achieve 50% model reduction ({total/1e9:.1f}B → {target_total/1e9:.1f}B):")
    print(f"  → Need to reduce experts from 32 to ~{avg_experts_remaining:.0f} per layer on average")
    print(f"  → This requires {expert_reduction_needed/expert*100:.0f}% reduction in expert params")
    print()


if __name__ == "__main__":
    main()
