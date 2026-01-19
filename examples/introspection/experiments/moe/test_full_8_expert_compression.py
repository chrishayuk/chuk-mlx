#!/usr/bin/env python3
"""
Full 8-Expert Compression Test - Apply to ALL Layers

Compress GPT-OSS from 32 -> 8 experts on ALL 24 layers and verify quality.
Target: 4.79B -> ~2.55B (47% reduction)

Usage:
    uv run python examples/introspection/experiments/moe/test_full_8_expert_compression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import generate, load

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    estimate_model_size,
    print_compression_summary,
)


def main():
    print("=" * 60)
    print("FULL 8-Expert Compression (32 -> 8 on ALL layers)")
    print("Target: 4.79B -> 2.55B (47% reduction)")
    print("=" * 60)

    # Load
    print("\nLoading GPT-OSS...")
    model_path = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )
    model, tokenizer = load(str(model_path))

    # Baseline size
    baseline = estimate_model_size(model)
    print(f"\nBaseline: {baseline['total'] / 1e9:.2f}B params")

    # Test prompts - more diverse for quality check
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
        "SELECT * FROM users WHERE",
        "The meaning of life is",
    ]

    # Get baseline outputs
    print("\n--- BASELINE OUTPUTS ---")
    baseline_outputs = []
    for p in prompts:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_outputs.append(out)
        print(f"{p}")
        print(f"  -> {out[:60]}...")

    # Create compression plans
    print("\n--- CREATING 8-EXPERT PLANS ---")
    hooks = MoEHooks(model)
    compressor = ExpertCompressor(model, tokenizer)

    plans = []
    for layer_idx in hooks.moe_layer_indices:
        plan = compressor.plan_compression(layer_idx, target_experts=8, strategy="aggressive")
        plans.append(plan)

    print(f"  Created plans for {len(plans)} layers")

    # Show summary BEFORE applying
    print("\n")
    print_compression_summary(model, plans, "GPT-OSS (8 experts)")

    # Apply compression to ALL layers
    print("\n--- APPLYING TO ALL LAYERS ---")
    for i, layer_idx in enumerate(hooks.moe_layer_indices):
        plan = plans[i]
        config = compressor.apply_compression(plan, layer_idx, inplace=True)
        print(
            f"  Layer {layer_idx}: {config.original_num_experts} -> {config.compressed_num_experts}"
        )

    # Force evaluation
    mx.eval(model.parameters())
    print("\nCompression applied to all layers!")

    # Verify size reduction
    compressed = estimate_model_size(model)
    print("\n--- SIZE VERIFICATION ---")
    print(f"  Before: {baseline['total'] / 1e9:.2f}B")
    print(f"  After:  {compressed['total'] / 1e9:.2f}B")
    print(f"  Reduction: {(1 - compressed['total'] / baseline['total']) * 100:.1f}%")

    # Test outputs after FULL compression
    print("\n--- POST-COMPRESSION OUTPUTS (ALL LAYERS) ---")
    quality_scores = []
    for i, p in enumerate(prompts):
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_tokens = set(baseline_outputs[i].split())
        new_tokens = set(out.split())
        overlap = len(baseline_tokens & new_tokens) / max(len(baseline_tokens), 1)
        quality_scores.append(overlap)

        print(f"{p}")
        print(f"  Baseline: {baseline_outputs[i][:50]}...")
        print(f"  Compressed: {out[:50]}...")
        print(f"  Token overlap: {overlap:.0%}")

    avg_quality = sum(quality_scores) / len(quality_scores)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model size: {baseline['total'] / 1e9:.2f}B -> {compressed['total'] / 1e9:.2f}B")
    print(f"  Reduction: {(1 - compressed['total'] / baseline['total']) * 100:.1f}%")
    print(f"  Avg quality (token overlap): {avg_quality:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
