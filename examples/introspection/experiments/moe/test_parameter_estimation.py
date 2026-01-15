#!/usr/bin/env python3
"""
Test Parameter Estimation for MoE Compression

This script demonstrates the parameter count estimation functions
to show actual model size reduction after expert compression.

Usage:
    uv run python examples/introspection/experiments/moe/test_parameter_estimation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from mlx_lm import load

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    estimate_compressed_size,
    estimate_model_size,
    print_compression_summary,
)


def main():
    print("=" * 70)
    print("MoE Parameter Estimation Test - GPT-OSS 20B")
    print("=" * 70)

    # Load GPT-OSS
    print("\nLoading GPT-OSS 20B...")
    model_path = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )
    model, tokenizer = load(str(model_path))

    # Get baseline model size
    print("\n" + "-" * 70)
    print("BASELINE MODEL SIZE")
    print("-" * 70)

    size_breakdown = estimate_model_size(model)

    total_b = size_breakdown["total"] / 1e9
    expert_b = size_breakdown["expert"] / 1e9
    attention_b = size_breakdown["attention"] / 1e9
    embeddings_b = size_breakdown["embeddings"] / 1e9
    other_b = size_breakdown["other"] / 1e9

    print(f"\n  Total parameters:     {total_b:.2f}B")
    print(
        f"  Expert parameters:    {expert_b:.2f}B ({size_breakdown['expert'] / size_breakdown['total'] * 100:.1f}%)"
    )
    print(
        f"  Attention parameters: {attention_b:.2f}B ({size_breakdown['attention'] / size_breakdown['total'] * 100:.1f}%)"
    )
    print(
        f"  Embedding parameters: {embeddings_b:.2f}B ({size_breakdown['embeddings'] / size_breakdown['total'] * 100:.1f}%)"
    )
    print(
        f"  Other parameters:     {other_b:.2f}B ({size_breakdown['other'] / size_breakdown['total'] * 100:.1f}%)"
    )

    # Create compression plans for all MoE layers
    print("\n" + "-" * 70)
    print("CREATING COMPRESSION PLANS")
    print("-" * 70)

    hooks = MoEHooks(model)
    moe_layers = hooks.moe_layer_indices
    print(f"\n  Found {len(moe_layers)} MoE layers")

    compressor = ExpertCompressor(model, tokenizer)

    # Test prompts for analysis
    test_prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "SELECT * FROM users WHERE",
    ]

    compression_plans = []
    for layer_idx in moe_layers:
        try:
            plan = compressor.plan_compression(layer_idx, strategy="balanced")
            compression_plans.append(plan)
            reduction = plan.original_num_experts - plan.target_num_experts
            if reduction > 0:
                print(
                    f"  Layer {layer_idx}: {plan.original_num_experts} -> {plan.target_num_experts} experts (-{reduction})"
                )
        except Exception as e:
            print(f"  Layer {layer_idx}: Error - {e}")

    # Print compression summary
    print("\n" + "-" * 70)
    print("COMPRESSION SUMMARY")
    print("-" * 70)

    print_compression_summary(model, compression_plans, model_name="GPT-OSS 20B")

    # Estimate compressed size
    print("\n" + "-" * 70)
    print("COMPRESSED SIZE ESTIMATION")
    print("-" * 70)

    compressed_info = estimate_compressed_size(model, compression_plans)

    original_b = compressed_info["original_params"] / 1e9
    compressed_b = compressed_info["compressed_params"] / 1e9
    removed_b = compressed_info["params_removed"] / 1e9

    print(f"\n  Original model:   {original_b:.2f}B parameters")
    print(f"  After compression: {compressed_b:.2f}B parameters")
    print(
        f"  Parameters removed: {removed_b:.2f}B ({compressed_info['reduction_ratio'] * 100:.1f}%)"
    )
    print(f"\n  Expert params before: {compressed_info['expert_params_original'] / 1e9:.2f}B")
    print(f"  Expert params after:  {compressed_info['expert_params_compressed'] / 1e9:.2f}B")

    print("\n" + "=" * 70)
    print(f"GPT-OSS 20B -> GPT-OSS {compressed_b:.1f}B (with balanced compression)")
    print("=" * 70)


if __name__ == "__main__":
    main()
