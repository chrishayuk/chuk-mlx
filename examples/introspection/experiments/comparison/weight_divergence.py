#!/usr/bin/env python3
"""
Weight Divergence Analysis - Where Does Tool-Calling Live?

Compare weight matrices between a base model and fine-tuned model to identify
which layers and components were most affected by fine-tuning.

This is the cheapest analysis (no inference needed) and gives immediate signal
about where tool-calling capabilities might be encoded.

Run: uv run python examples/introspection/weight_divergence.py
     uv run python examples/introspection/weight_divergence.py --base model1 --ft model2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from _loader import load_model

# Default model pairs to compare
DEFAULT_PAIRS = {
    "gemma": {
        "base": "mlx-community/gemma-3-270m-it-bf16",
        "ft": "mlx-community/functiongemma-270m-it-bf16",
    },
}


@dataclass
class WeightDivergence:
    """Weight divergence between two models for a specific component."""

    layer: int
    component: str
    frobenius_norm_diff: float
    cosine_similarity: float


def get_layer_weights(model, layer_idx: int) -> dict[str, mx.array]:
    """Extract weight tensors for a specific layer."""
    layer = model.model.layers[layer_idx]
    weights = {}

    # Attention weights
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, name):
                weights[f"attn_{name[0]}"] = getattr(attn, name).weight

    # MLP weights
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(mlp, name):
                short = name.replace("_proj", "")
                weights[f"mlp_{short}"] = getattr(mlp, name).weight

    return weights


def compute_divergence(base_model, ft_model, num_layers: int) -> list[WeightDivergence]:
    """Compute per-layer, per-component weight divergence."""
    divergences = []

    for layer_idx in range(num_layers):
        base_weights = get_layer_weights(base_model, layer_idx)
        ft_weights = get_layer_weights(ft_model, layer_idx)

        for component in base_weights:
            if component not in ft_weights:
                continue

            base_w = base_weights[component]
            ft_w = ft_weights[component]

            if base_w.shape != ft_w.shape:
                continue

            diff = ft_w - base_w

            # Frobenius norm (normalized)
            base_norm = float(mx.sqrt(mx.sum(base_w * base_w)))
            diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
            normalized_diff = diff_norm / (base_norm + 1e-8)

            # Cosine similarity
            base_flat = base_w.reshape(-1)
            ft_flat = ft_w.reshape(-1)
            dot = float(mx.sum(base_flat * ft_flat))
            norm_base = float(mx.sqrt(mx.sum(base_flat * base_flat)))
            norm_ft = float(mx.sqrt(mx.sum(ft_flat * ft_flat)))
            cos_sim = dot / (norm_base * norm_ft + 1e-8)

            divergences.append(
                WeightDivergence(
                    layer=layer_idx,
                    component=component,
                    frobenius_norm_diff=normalized_diff,
                    cosine_similarity=cos_sim,
                )
            )

    return divergences


def print_heatmap(divergences: list[WeightDivergence], num_layers: int):
    """Print ASCII heatmap of weight divergence."""
    print("\n" + "=" * 80)
    print("WEIGHT DIVERGENCE HEATMAP (Frobenius norm, normalized)")
    print("=" * 80)

    components = sorted(set(d.component for d in divergences))
    matrix = {(d.layer, d.component): d.frobenius_norm_diff for d in divergences}
    max_val = max(d.frobenius_norm_diff for d in divergences) if divergences else 1.0

    print(f"\n{'Layer':<6}", end="")
    for comp in components:
        print(f"{comp[:8]:>9}", end="")
    print("   | Interpretation")
    print("-" * 80)

    for layer_idx in range(num_layers):
        print(f"{layer_idx:<6}", end="")
        layer_max = 0.0

        for comp in components:
            val = matrix.get((layer_idx, comp), 0.0)
            layer_max = max(layer_max, val)
            intensity = int(val / max_val * 8) if max_val > 0 else 0
            bars = "█" * intensity + "░" * (8 - intensity)
            print(f" {bars}", end="")

        pct = (layer_idx + 1) / num_layers * 100
        marker = ""
        if layer_max > max_val * 0.7:
            marker = " <<< HIGH"
        elif layer_max > max_val * 0.4:
            marker = " << mid"

        print(f"   | {pct:5.1f}% depth{marker}")


def print_summary(divergences: list[WeightDivergence], num_layers: int):
    """Print summary by layer."""
    print("\n" + "=" * 80)
    print("LAYER SUMMARY")
    print("=" * 80)

    layer_totals = {}
    for d in divergences:
        layer_totals[d.layer] = layer_totals.get(d.layer, 0) + d.frobenius_norm_diff

    max_total = max(layer_totals.values()) if layer_totals else 1.0

    print(f"\n{'Layer':<6} {'Total Div':>10} {'Depth %':>8}  Visual")
    print("-" * 60)

    for layer_idx in range(num_layers):
        total = layer_totals.get(layer_idx, 0)
        pct = (layer_idx + 1) / num_layers * 100
        bar_len = int(total / max_total * 40) if max_total > 0 else 0
        bar = "█" * bar_len

        marker = ""
        if total > max_total * 0.8:
            marker = " *** PEAK"
        elif total > max_total * 0.6:
            marker = " ** high"

        print(f"{layer_idx:<6} {total:>10.6f} {pct:>7.1f}%  {bar}{marker}")

    # Top layers
    sorted_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 divergent layers:")
    for layer, total in sorted_layers[:5]:
        pct = (layer + 1) / num_layers * 100
        print(f"  Layer {layer} ({pct:.1f}% depth): {total:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Weight Divergence Analysis")
    parser.add_argument("--base", default=None, help="Base model ID")
    parser.add_argument("--ft", default=None, help="Fine-tuned model ID")
    parser.add_argument(
        "--pair", choices=list(DEFAULT_PAIRS.keys()), default="gemma", help="Predefined model pair"
    )
    args = parser.parse_args()

    # Get model IDs
    if args.base and args.ft:
        base_id = args.base
        ft_id = args.ft
    else:
        pair = DEFAULT_PAIRS[args.pair]
        base_id = pair["base"]
        ft_id = pair["ft"]

    print("=" * 80)
    print("Weight Divergence Analysis")
    print(f"Base: {base_id}")
    print(f"Fine-tuned: {ft_id}")
    print("=" * 80)

    # Load models
    base_model, _, base_config, _ = load_model(base_id)
    ft_model, _, ft_config, _ = load_model(ft_id)

    # Verify compatible
    if base_config.num_hidden_layers != ft_config.num_hidden_layers:
        print("ERROR: Layer count mismatch")
        return
    if base_config.hidden_size != ft_config.hidden_size:
        print("ERROR: Hidden size mismatch")
        return

    num_layers = base_config.num_hidden_layers
    print(f"\nArchitecture: {num_layers} layers, {base_config.hidden_size} hidden")

    # Compute divergence
    print("\nComputing weight divergences...")
    divergences = compute_divergence(base_model, ft_model, num_layers)
    print(f"Computed {len(divergences)} component divergences")

    # Print results
    print_heatmap(divergences, num_layers)
    print_summary(divergences, num_layers)

    # Save results
    output_path = Path("weight_divergence_results.json")
    results = [
        {
            "layer": d.layer,
            "component": d.component,
            "frobenius_norm_diff": d.frobenius_norm_diff,
            "cosine_similarity": d.cosine_similarity,
        }
        for d in divergences
    ]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    del base_model, ft_model
    mx.metal.clear_cache()


if __name__ == "__main__":
    main()
