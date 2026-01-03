#!/usr/bin/env python3
"""
Test 8-Expert Compression with Quality Check

Compress GPT-OSS from 32 -> 8 experts and verify quality.

Usage:
    uv run python examples/introspection/experiments/moe/test_8_experts.py
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
    print_compression_summary,
)


def main():
    print("=" * 60)
    print("8-Expert Compression Test (32 -> 8 experts)")
    print("=" * 60)

    # Load
    print("\nLoading GPT-OSS...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, tokenizer = load(str(model_path))

    # Baseline size
    baseline = estimate_model_size(model)
    print(f"\nBaseline: {baseline['total']/1e9:.2f}B params")

    # Test prompts
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
    ]

    # Get baseline outputs
    print("\n--- BASELINE OUTPUTS ---")
    baseline_outputs = []
    for p in prompts:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_outputs.append(out)
        print(f"{p}")
        print(f"  -> {out[:60]}...")

    # Create compression plans for 8 experts using AGGRESSIVE strategy
    print("\n--- CREATING 8-EXPERT PLANS (AGGRESSIVE) ---")
    hooks = MoEHooks(model)
    compressor = ExpertCompressor(model, tokenizer)

    plans = []
    for layer_idx in hooks.moe_layer_indices:
        plan = compressor.plan_compression(layer_idx, target_experts=8, strategy="aggressive")
        plans.append(plan)
        print(f"  Layer {layer_idx}: 32 -> {plan.target_num_experts}")

    # Show summary
    print("\n")
    print_compression_summary(model, plans, "GPT-OSS (8 experts)")

    # Apply compression to middle layer only for quick test
    print("\n--- APPLYING TO LAYER 12 ---")
    layer_idx = 12
    plan = plans[12]
    print(f"Plan: {plan.original_num_experts} -> {plan.target_num_experts} experts")
    print(f"  Pruned: {len(plan.pruned_experts)} experts")
    config = compressor.apply_compression(plan, layer_idx, inplace=True)
    mx.eval(model.parameters())
    print(f"Applied: {config.original_num_experts} -> {config.compressed_num_experts}")

    # Test outputs
    print("\n--- POST-COMPRESSION OUTPUTS ---")
    for i, p in enumerate(prompts):
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_tokens = set(baseline_outputs[i].split())
        new_tokens = set(out.split())
        overlap = len(baseline_tokens & new_tokens) / max(len(baseline_tokens), 1)
        print(f"{p}")
        print(f"  -> {out[:60]}...")
        print(f"  Token overlap: {overlap:.0%}")

    print("\n" + "=" * 60)
    print("Done! 8-expert compression on layer 12 tested.")
    print("=" * 60)


if __name__ == "__main__":
    main()
