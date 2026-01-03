#!/usr/bin/env python3
"""
Shared Expert Pool - Cross-Layer Expert Sharing

Instead of 24 layers × 32 experts = 768 expert sets,
use a single shared pool of N experts that all layers reference.

This exploits the finding that expert weights are ~68% redundant across layers.

Usage:
    uv run python examples/introspection/experiments/moe/shared_expert_pool.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate

from chuk_lazarus.introspection.moe import MoEHooks, estimate_model_size


def find_best_expert_match(source_weights: mx.array, pool_weights: mx.array) -> tuple[int, float]:
    """Find the best matching expert in the pool for a source expert."""
    best_idx = 0
    best_sim = -1

    source_flat = source_weights.flatten()
    source_norm = mx.sqrt(mx.sum(source_flat * source_flat))

    for i in range(pool_weights.shape[0]):
        pool_flat = pool_weights[i].flatten()
        pool_norm = mx.sqrt(mx.sum(pool_flat * pool_flat))
        sim = float(mx.sum(source_flat * pool_flat) / (source_norm * pool_norm + 1e-8))
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx, best_sim


def create_shared_pool_from_layer(model, source_layer_idx: int, pool_size: int = 32) -> dict:
    """
    Create a shared expert pool from a single layer's experts.

    Args:
        model: The MoE model
        source_layer_idx: Layer to use as the source for the pool
        pool_size: Number of experts in the shared pool

    Returns:
        Dict with pool weights for gate_proj, up_proj, down_proj (and scales for quantized)
    """
    layers = model.model.layers
    source = layers[source_layer_idx].mlp.experts

    # Take first pool_size experts from source layer
    pool = {
        "gate_proj": source.gate_proj.weight[:pool_size],
        "up_proj": source.up_proj.weight[:pool_size],
        "down_proj": source.down_proj.weight[:pool_size],
    }

    # Handle quantized models - also copy scales
    if hasattr(source.gate_proj, "scales") and source.gate_proj.scales is not None:
        pool["gate_proj_scales"] = source.gate_proj.scales[:pool_size]
        pool["up_proj_scales"] = source.up_proj.scales[:pool_size]
        pool["down_proj_scales"] = source.down_proj.scales[:pool_size]

    # Also get biases if they exist
    if hasattr(source.gate_proj, "bias") and source.gate_proj.bias is not None:
        pool["gate_proj_bias"] = source.gate_proj.bias[:pool_size]
        pool["up_proj_bias"] = source.up_proj.bias[:pool_size]
        pool["down_proj_bias"] = source.down_proj.bias[:pool_size]

    return pool


def create_merged_pool(model, pool_size: int = 32) -> dict:
    """
    Create a shared pool by averaging experts across all layers.

    For each expert position, average the weights from all layers.
    This creates a "consensus" expert that represents all layers.

    Note: For quantized models, averaging quantized weights is not mathematically
    correct but may still work as an approximation. Scales are copied from layer 12.
    """
    layers = model.model.layers
    moe_layers = [l for l in layers if hasattr(l, "mlp") and hasattr(l.mlp, "experts")]

    if not moe_layers:
        return None

    # Get reference shapes
    ref = moe_layers[0].mlp.experts
    num_experts = ref.gate_proj.weight.shape[0]
    pool_size = min(pool_size, num_experts)

    # Accumulate weights across layers
    gate_sum = mx.zeros_like(ref.gate_proj.weight[:pool_size])
    up_sum = mx.zeros_like(ref.up_proj.weight[:pool_size])
    down_sum = mx.zeros_like(ref.down_proj.weight[:pool_size])

    for layer in moe_layers:
        gate_sum = gate_sum + layer.mlp.experts.gate_proj.weight[:pool_size]
        up_sum = up_sum + layer.mlp.experts.up_proj.weight[:pool_size]
        down_sum = down_sum + layer.mlp.experts.down_proj.weight[:pool_size]

    n = len(moe_layers)
    pool = {
        "gate_proj": gate_sum / n,
        "up_proj": up_sum / n,
        "down_proj": down_sum / n,
    }

    # For quantized models, copy scales from a middle layer (layer 12)
    # Averaging scales doesn't make mathematical sense for quantization
    mid_layer_idx = len(moe_layers) // 2
    mid_layer = moe_layers[mid_layer_idx].mlp.experts
    if hasattr(mid_layer.gate_proj, "scales") and mid_layer.gate_proj.scales is not None:
        pool["gate_proj_scales"] = mid_layer.gate_proj.scales[:pool_size]
        pool["up_proj_scales"] = mid_layer.up_proj.scales[:pool_size]
        pool["down_proj_scales"] = mid_layer.down_proj.scales[:pool_size]

    return pool


def apply_shared_pool(model, pool: dict, layers_to_share: list[int] | None = None):
    """
    Replace expert weights in specified layers with shared pool.

    Args:
        model: The MoE model
        pool: Shared pool weights from create_shared_pool_from_layer
        layers_to_share: Which layers to update (None = all)
    """
    layers = model.model.layers

    if layers_to_share is None:
        layers_to_share = list(range(len(layers)))

    for layer_idx in layers_to_share:
        layer = layers[layer_idx]
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue

        experts = layer.mlp.experts
        pool_size = pool["gate_proj"].shape[0]

        # Replace weights with shared pool
        experts.gate_proj.weight = pool["gate_proj"]
        experts.up_proj.weight = pool["up_proj"]
        experts.down_proj.weight = pool["down_proj"]

        # Update scales for quantized models (CRITICAL for gather_qmm)
        if "gate_proj_scales" in pool and hasattr(experts.gate_proj, "scales"):
            experts.gate_proj.scales = pool["gate_proj_scales"]
            experts.up_proj.scales = pool["up_proj_scales"]
            experts.down_proj.scales = pool["down_proj_scales"]

        # Update biases if present
        if "gate_proj_bias" in pool and hasattr(experts.gate_proj, "bias"):
            experts.gate_proj.bias = pool["gate_proj_bias"]
            experts.up_proj.bias = pool["up_proj_bias"]
            experts.down_proj.bias = pool["down_proj_bias"]

        # Update router to match pool size (prune to first pool_size experts)
        if hasattr(layer.mlp, "router"):
            router = layer.mlp.router
            router.weight = router.weight[:pool_size]
            if hasattr(router, "bias") and router.bias is not None:
                router.bias = router.bias[:pool_size]


def test_pool(model, tokenizer, pool: dict, pool_name: str, baseline: dict,
               prompts: list, baseline_outputs: list) -> dict:
    """Test a pool configuration and return results."""
    apply_shared_pool(model, pool)
    mx.eval(model.parameters())

    new_size = estimate_model_size(model)
    reduction = (1 - new_size['total'] / baseline['total']) * 100

    quality_scores = []
    outputs = []
    for i, p in enumerate(prompts):
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        outputs.append(out)
        baseline_tokens = set(baseline_outputs[i].split())
        new_tokens = set(out.split())
        overlap = len(baseline_tokens & new_tokens) / max(len(baseline_tokens), 1)
        quality_scores.append(overlap)

    return {
        "name": pool_name,
        "size_before": baseline['total'],
        "size_after": new_size['total'],
        "reduction": reduction,
        "quality": sum(quality_scores) / len(quality_scores),
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("Shared Expert Pool - Cross-Layer Expert Sharing")
    print("=" * 70)

    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

    # Load and get baseline
    print("\nLoading GPT-OSS...")
    model, tokenizer = load(str(model_path))
    baseline = estimate_model_size(model)
    print(f"Baseline: {baseline['total']/1e9:.2f}B params")

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
    ]

    print("\n--- BASELINE OUTPUTS ---")
    baseline_outputs = []
    for p in prompts:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_outputs.append(out)
        print(f"{p}")
        print(f"  -> {out[:60]}...")

    results = []

    # Test 1: Shared pool from layer 12 (16 experts)
    print(f"\n{'='*70}")
    print("TEST 1: Copy layer 12's experts to all layers (16 experts)")
    print("=" * 70)
    model, tokenizer = load(str(model_path))
    pool = create_shared_pool_from_layer(model, source_layer_idx=12, pool_size=16)
    result = test_pool(model, tokenizer, pool, "Copy Layer 12 (16)", baseline, prompts, baseline_outputs)
    results.append(result)
    print(f"Size: {result['size_before']/1e9:.2f}B -> {result['size_after']/1e9:.2f}B ({result['reduction']:.1f}%)")
    print(f"Quality: {result['quality']:.0%}")
    for i, p in enumerate(prompts):
        print(f"  {p[:30]}... -> {result['outputs'][i][:40]}...")

    # Test 2: Copy layer 12's experts (32 experts - no size reduction)
    print(f"\n{'='*70}")
    print("TEST 2: Copy layer 12's experts to all layers (32 experts - full)")
    print("=" * 70)
    model, tokenizer = load(str(model_path))
    pool = create_shared_pool_from_layer(model, source_layer_idx=12, pool_size=32)
    result = test_pool(model, tokenizer, pool, "Copy Layer 12 (32)", baseline, prompts, baseline_outputs)
    results.append(result)
    print(f"Size: {result['size_before']/1e9:.2f}B -> {result['size_after']/1e9:.2f}B ({result['reduction']:.1f}%)")
    print(f"Quality: {result['quality']:.0%}")
    for i, p in enumerate(prompts):
        print(f"  {p[:30]}... -> {result['outputs'][i][:40]}...")

    # Test 3: Copy layer 0's experts (early layer, 16 experts)
    print(f"\n{'='*70}")
    print("TEST 3: Copy layer 0's experts to all layers (16 experts)")
    print("=" * 70)
    model, tokenizer = load(str(model_path))
    pool = create_shared_pool_from_layer(model, source_layer_idx=0, pool_size=16)
    result = test_pool(model, tokenizer, pool, "Copy Layer 0 (16)", baseline, prompts, baseline_outputs)
    results.append(result)
    print(f"Size: {result['size_before']/1e9:.2f}B -> {result['size_after']/1e9:.2f}B ({result['reduction']:.1f}%)")
    print(f"Quality: {result['quality']:.0%}")
    for i, p in enumerate(prompts):
        print(f"  {p[:30]}... -> {result['outputs'][i][:40]}...")

    # Test 4: Copy layer 23's experts (last layer, 16 experts)
    print(f"\n{'='*70}")
    print("TEST 4: Copy layer 23's experts to all layers (16 experts)")
    print("=" * 70)
    model, tokenizer = load(str(model_path))
    pool = create_shared_pool_from_layer(model, source_layer_idx=23, pool_size=16)
    result = test_pool(model, tokenizer, pool, "Copy Layer 23 (16)", baseline, prompts, baseline_outputs)
    results.append(result)
    print(f"Size: {result['size_before']/1e9:.2f}B -> {result['size_after']/1e9:.2f}B ({result['reduction']:.1f}%)")
    print(f"Quality: {result['quality']:.0%}")
    for i, p in enumerate(prompts):
        print(f"  {p[:30]}... -> {result['outputs'][i][:40]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Layer Expert Sharing")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Size':<15} {'Reduction':<12} {'Quality':<10}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:<25} {r['size_after']/1e9:.2f}B{'':>8} {r['reduction']:.1f}%{'':>6} {r['quality']:.0%}")
    print("-" * 62)
    print("\nConclusion: Tests whether expert weights can be shared across layers")
    print("High quality = weights are redundant, Low quality = layer-specific specialization needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
