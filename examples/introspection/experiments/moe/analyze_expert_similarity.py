#!/usr/bin/env python3
"""
Analyze Expert Similarity/Duplication in GPT-OSS

Quantify how much redundancy exists across experts within and between layers.

Usage:
    uv run python examples/introspection/experiments/moe/analyze_expert_similarity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
import numpy as np
from mlx_lm import load


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between flattened tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    return float(dot / (norm_a * norm_b + 1e-8))


def analyze_layer_experts(layer, layer_idx: int) -> dict:
    """Analyze expert similarity within a single layer."""
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
        return None

    experts = layer.mlp.experts
    if not hasattr(experts, "gate_proj"):
        return None

    # Get gate_proj weights: [num_experts, hidden_dims, input_dims]
    weights = experts.gate_proj.weight
    num_experts = weights.shape[0]

    # Compute pairwise similarities
    similarities = []
    high_sim_pairs = []

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = cosine_similarity(weights[i], weights[j])
            similarities.append(sim)
            if sim > 0.8:
                high_sim_pairs.append((i, j, sim))

    similarities = np.array(similarities)

    return {
        "layer_idx": layer_idx,
        "num_experts": num_experts,
        "mean_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities)),
        "high_sim_pairs": len(high_sim_pairs),  # pairs with sim > 0.8
        "pct_redundant": len([s for s in similarities if s > 0.7]) / len(similarities) * 100,
    }


def analyze_cross_layer_experts(layers, layer_a: int, layer_b: int) -> dict:
    """Analyze expert similarity between two layers."""
    mlp_a = layers[layer_a].mlp.experts.gate_proj.weight
    mlp_b = layers[layer_b].mlp.experts.gate_proj.weight

    num_experts = mlp_a.shape[0]

    # Find best matching experts between layers
    best_matches = []
    cross_sims = []

    for i in range(num_experts):
        best_sim = -1
        best_j = -1
        for j in range(num_experts):
            sim = cosine_similarity(mlp_a[i], mlp_b[j])
            cross_sims.append(sim)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        best_matches.append((i, best_j, best_sim))

    cross_sims = np.array(cross_sims)

    return {
        "layer_a": layer_a,
        "layer_b": layer_b,
        "mean_cross_similarity": float(np.mean(cross_sims)),
        "max_cross_similarity": float(np.max(cross_sims)),
        "avg_best_match": float(np.mean([m[2] for m in best_matches])),
        "shareable_experts": len([m for m in best_matches if m[2] > 0.9]),
    }


def main():
    print("=" * 70)
    print("Expert Similarity Analysis - Quantifying Redundancy")
    print("=" * 70)

    # Load
    print("\nLoading GPT-OSS...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, _ = load(str(model_path))

    layers = model.model.layers

    # Analyze within-layer similarity
    print("\n" + "=" * 70)
    print("WITHIN-LAYER EXPERT SIMILARITY")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'Mean':<8} {'Max':<8} {'Redundant%':<12} {'High Pairs':<10}")
    print("-" * 50)

    layer_stats = []
    for i, layer in enumerate(layers):
        stats = analyze_layer_experts(layer, i)
        if stats:
            layer_stats.append(stats)
            print(f"{i:<8} {stats['mean_similarity']:.3f}    {stats['max_similarity']:.3f}    "
                  f"{stats['pct_redundant']:.1f}%{'':<7} {stats['high_sim_pairs']}")

    # Summary
    if layer_stats:
        avg_mean = np.mean([s['mean_similarity'] for s in layer_stats])
        avg_redundant = np.mean([s['pct_redundant'] for s in layer_stats])
        print("-" * 50)
        print(f"{'AVG':<8} {avg_mean:.3f}    {'-':<8} {avg_redundant:.1f}%")

    # Analyze cross-layer similarity (sample adjacent and distant layers)
    print("\n" + "=" * 70)
    print("CROSS-LAYER EXPERT SIMILARITY")
    print("=" * 70)
    print(f"\n{'Layers':<12} {'Mean':<8} {'Max':<8} {'Avg Best':<10} {'Shareable':<10}")
    print("-" * 50)

    cross_pairs = [
        (0, 1), (0, 2), (0, 12), (0, 23),  # Layer 0 vs others
        (11, 12), (11, 13),  # Middle layers
        (22, 23),  # Late layers
    ]

    for layer_a, layer_b in cross_pairs:
        if layer_a < len(layers) and layer_b < len(layers):
            stats = analyze_cross_layer_experts(layers, layer_a, layer_b)
            print(f"{layer_a}->{layer_b:<6} {stats['mean_cross_similarity']:.3f}    "
                  f"{stats['max_cross_similarity']:.3f}    {stats['avg_best_match']:.3f}      "
                  f"{stats['shareable_experts']}/32")

    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if layer_stats:
        total_redundant = avg_redundant
        print(f"\n  Within-layer redundancy: {total_redundant:.1f}% of expert pairs are highly similar")
        print(f"  Average pairwise similarity: {avg_mean:.3f}")

        if total_redundant > 50:
            print(f"\n  -> HIGH REDUNDANCY: Could potentially share 50%+ of experts")
        elif total_redundant > 30:
            print(f"\n  -> MODERATE REDUNDANCY: Could potentially share 30-50% of experts")
        else:
            print(f"\n  -> LOW REDUNDANCY: Experts are relatively diverse")

    print("=" * 70)


if __name__ == "__main__":
    main()
