#!/usr/bin/env python3
"""
Selective Layer Compression - Find Quality-Preserving Configuration

Instead of compressing all layers equally, test:
1. Compress only middle layers (keep first/last intact)
2. Compress fewer experts per layer
3. Progressive compression (more aggressive in middle)

Usage:
    uv run python examples/introspection/experiments/moe/test_selective_compression.py
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
)


def test_compression_config(
    model_path: Path,
    config_name: str,
    layers_to_compress: list[int],
    target_experts: int,
    prompts: list[str],
    baseline_outputs: list[str],
    baseline_size: float,
) -> dict:
    """Test a specific compression configuration."""
    model, tokenizer = load(str(model_path))

    hooks = MoEHooks(model)
    compressor = ExpertCompressor(model, tokenizer)

    # Only compress specified layers
    for layer_idx in layers_to_compress:
        if layer_idx in hooks.moe_layer_indices:
            plan = compressor.plan_compression(layer_idx, target_experts=target_experts, strategy="aggressive")
            compressor.apply_compression(plan, layer_idx, inplace=True)

    mx.eval(model.parameters())

    new_size = estimate_model_size(model)
    reduction = (1 - new_size['total'] / baseline_size) * 100

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
        "name": config_name,
        "layers_compressed": len(layers_to_compress),
        "target_experts": target_experts,
        "size_after": new_size['total'],
        "reduction": reduction,
        "quality": sum(quality_scores) / len(quality_scores),
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("Selective Layer Compression - Quality Preservation Test")
    print("=" * 70)

    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

    # Get baseline
    print("\nLoading GPT-OSS for baseline...")
    model, tokenizer = load(str(model_path))
    baseline = estimate_model_size(model)
    baseline_size = baseline['total']
    print(f"Baseline: {baseline_size/1e9:.2f}B params")

    # All MoE layers: 0-23
    all_layers = list(range(24))

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
        "SELECT * FROM users WHERE",
    ]

    print("\n--- BASELINE OUTPUTS ---")
    baseline_outputs = []
    for p in prompts:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_outputs.append(out)
        print(f"{p}")
        print(f"  -> {out[:60]}...")

    del model, tokenizer

    results = []

    # Test configurations
    configs = [
        # (name, layers_to_compress, target_experts)
        ("16 experts (all layers)", all_layers, 16),
        ("16 experts (middle 12 layers)", list(range(6, 18)), 16),
        ("16 experts (middle 8 layers)", list(range(8, 16)), 16),
        ("20 experts (all layers)", all_layers, 20),
        ("24 experts (all layers)", all_layers, 24),
        ("8 experts (middle 8 layers)", list(range(8, 16)), 8),
        ("12 experts (all layers)", all_layers, 12),
    ]

    for config_name, layers, target in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config_name}")
        print(f"  Layers: {len(layers)} ({min(layers) if layers else 'N/A'}-{max(layers) if layers else 'N/A'})")
        print(f"  Target experts: {target}")
        print("=" * 70)

        result = test_compression_config(
            model_path, config_name, layers, target,
            prompts, baseline_outputs, baseline_size
        )
        results.append(result)

        print(f"Size: {baseline_size/1e9:.2f}B -> {result['size_after']/1e9:.2f}B ({result['reduction']:.1f}% reduction)")
        print(f"Quality: {result['quality']:.0%}")
        for i, p in enumerate(prompts):
            print(f"  {p[:25]}... -> {result['outputs'][i][:35]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Selective Compression Results")
    print("=" * 70)
    print(f"\n{'Configuration':<30} {'Size':<10} {'Reduction':<12} {'Quality':<10}")
    print("-" * 65)

    # Sort by quality descending
    results.sort(key=lambda x: x['quality'], reverse=True)

    for r in results:
        print(f"{r['name']:<30} {r['size_after']/1e9:.2f}B{'':>4} {r['reduction']:.1f}%{'':>6} {r['quality']:.0%}")

    print("-" * 65)

    # Find best quality with >20% reduction
    good_results = [r for r in results if r['reduction'] > 20 and r['quality'] >= 0.5]
    if good_results:
        best = max(good_results, key=lambda x: x['quality'])
        print(f"\nBest config (>20% reduction, >50% quality): {best['name']}")
        print(f"  -> {best['reduction']:.1f}% reduction, {best['quality']:.0%} quality")
    else:
        print("\nNo config achieved >20% reduction with >50% quality")
        best = max(results, key=lambda x: x['quality'])
        print(f"Highest quality: {best['name']} ({best['quality']:.0%} quality, {best['reduction']:.1f}% reduction)")

    print("=" * 70)


if __name__ == "__main__":
    main()
