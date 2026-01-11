#!/usr/bin/env python3
"""
Test All Compression Levels - 8, 6, 4 experts

Fast test: apply each compression level to layer 12 only and compare quality.

Usage:
    uv run python examples/introspection/experiments/moe/test_all_compression_levels.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    estimate_model_size,
    estimate_compressed_size,
)


def test_compression_level(model, tokenizer, compressor, layer_idx, target_experts, baseline_outputs, prompts):
    """Test a specific compression level and return quality score."""
    # Create plan
    plan = compressor.plan_compression(layer_idx, target_experts=target_experts, strategy="aggressive")

    # Apply
    config = compressor.apply_compression(plan, layer_idx, inplace=True)
    mx.eval(model.parameters())

    # Test quality
    quality_scores = []
    for i, p in enumerate(prompts):
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_tokens = set(baseline_outputs[i].split())
        new_tokens = set(out.split())
        overlap = len(baseline_tokens & new_tokens) / max(len(baseline_tokens), 1)
        quality_scores.append(overlap)

    return sum(quality_scores) / len(quality_scores), config


def main():
    print("=" * 70)
    print("All Compression Levels Test (8, 6, 4 experts)")
    print("=" * 70)

    # Load
    print("\nLoading GPT-OSS...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

    # Test prompts
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
    ]

    # Test each compression level
    compression_levels = [8, 6, 4]
    results = {}

    for target in compression_levels:
        print(f"\n{'='*70}")
        print(f"TESTING {target} EXPERTS (32 -> {target})")
        print("="*70)

        # Reload model fresh for each test
        model, tokenizer = load(str(model_path))
        baseline = estimate_model_size(model)

        # Get baseline outputs
        print("Getting baseline outputs...")
        baseline_outputs = []
        for p in prompts:
            out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
            baseline_outputs.append(out)

        # Create compressor and get plans for size estimation
        hooks = MoEHooks(model)
        compressor = ExpertCompressor(model, tokenizer)

        plans = []
        for layer_idx in hooks.moe_layer_indices:
            plan = compressor.plan_compression(layer_idx, target_experts=target, strategy="aggressive")
            plans.append(plan)

        # Estimate compressed size
        stats = estimate_compressed_size(model, plans)

        # Apply to layer 12 and test quality
        print(f"Applying to layer 12...")
        quality, config = test_compression_level(
            model, tokenizer, compressor, 12, target, baseline_outputs, prompts
        )

        results[target] = {
            "original_b": baseline["total"] / 1e9,
            "compressed_b": stats["compressed_params"] / 1e9,
            "reduction_pct": stats["reduction_ratio"] * 100,
            "quality_pct": quality * 100,
        }

        print(f"\nResults for {target} experts:")
        print(f"  Size: {results[target]['original_b']:.2f}B -> {results[target]['compressed_b']:.2f}B")
        print(f"  Reduction: {results[target]['reduction_pct']:.1f}%")
        print(f"  Quality (layer 12): {results[target]['quality_pct']:.0f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Compression Options for GPT-OSS 20B")
    print("=" * 70)
    print(f"\n{'Experts':<10} {'Original':<12} {'Compressed':<12} {'Reduction':<12} {'Quality':<10}")
    print("-" * 56)

    for target in compression_levels:
        r = results[target]
        print(f"{target:<10} {r['original_b']:.2f}B{'':<6} {r['compressed_b']:.2f}B{'':<6} {r['reduction_pct']:.1f}%{'':<7} {r['quality_pct']:.0f}%")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)

    # Find best option for ~50% reduction
    best = None
    for target in compression_levels:
        r = results[target]
        if r["quality_pct"] >= 80:  # Acceptable quality threshold
            if best is None or abs(r["reduction_pct"] - 50) < abs(results[best]["reduction_pct"] - 50):
                best = target

    if best:
        r = results[best]
        print(f"\n  {best} experts: {r['original_b']:.2f}B -> {r['compressed_b']:.2f}B ({r['reduction_pct']:.1f}% reduction)")
        print(f"  Quality maintained: {r['quality_pct']:.0f}%")
    else:
        print("\n  No compression level maintains acceptable quality (>=80%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
