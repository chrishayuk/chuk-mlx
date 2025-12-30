#!/usr/bin/env python3
"""
Activation Divergence Analysis - Where Do Representations Diverge?

Compare hidden state representations between a base model and fine-tuned model
at each layer to find where tool-calling representations emerge.

This is a model-agnostic version that works with any supported model family.

Run: uv run python examples/introspection/activation_divergence.py
     uv run python examples/introspection/activation_divergence.py --base gemma-3-270m --ft functiongemma-270m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from _loader import (
    ActivationDivergence,
    compare_activations,
    format_tool_prompt,
    load_chat_template,
    load_model,
)

# Default model pairs to compare
DEFAULT_PAIRS = {
    "gemma": {
        "base": "mlx-community/gemma-3-270m-it-bf16",
        "ft": "mlx-community/functiongemma-270m-it-bf16",
    },
}


def analyze_divergence(
    base_model,
    ft_model,
    tokenizer,
    prompts: dict[str, list[str]],
) -> dict[str, list[list[ActivationDivergence]]]:
    """Analyze activation divergence for multiple prompt categories."""
    results = {}

    for prompt_type, prompt_list in prompts.items():
        print(f"\nProcessing {prompt_type} prompts ({len(prompt_list)} prompts)...")
        results[prompt_type] = []

        for i, prompt in enumerate(prompt_list):
            short = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"  [{i+1}/{len(prompt_list)}] {short}")

            divs = compare_activations(base_model, ft_model, tokenizer, prompt)
            results[prompt_type].append(divs)

    return results


def print_divergence_summary(
    results: dict[str, list[list[ActivationDivergence]]],
    num_layers: int,
):
    """Print a summary of divergence by prompt type."""
    print("\n" + "=" * 80)
    print("ACTIVATION DIVERGENCE SUMMARY")
    print("=" * 80)

    for prompt_type, all_divs in results.items():
        print(f"\n--- {prompt_type} ---")

        # Average across prompts per layer
        layer_cos = {l: [] for l in range(num_layers)}
        for divs in all_divs:
            for d in divs:
                layer_cos[d.layer].append(d.cosine_similarity)

        print(f"{'Layer':<8} {'Cos Sim':>10} {'Divergence':>40}")
        print("-" * 60)

        for layer in range(num_layers):
            if not layer_cos[layer]:
                continue
            avg_cos = sum(layer_cos[layer]) / len(layer_cos[layer])
            divergence = 1 - avg_cos
            bar_len = int(divergence * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            marker = ""
            if divergence > 0.3:
                marker = " *** HIGH"
            elif divergence > 0.15:
                marker = " ** mid"

            print(f"{layer:<8} {avg_cos:>10.4f} {bar}{marker}")


def find_emergence_layer(
    results: dict[str, list[list[ActivationDivergence]]],
    num_layers: int,
) -> dict[str, int | None]:
    """Find where tool prompts start diverging more than neutral."""
    emergence = {}

    # Get neutral baseline
    if "neutral" not in results:
        return emergence

    neutral_by_layer = {l: [] for l in range(num_layers)}
    for divs in results["neutral"]:
        for d in divs:
            neutral_by_layer[d.layer].append(1 - d.cosine_similarity)

    neutral_avg = {l: sum(v) / len(v) if v else 0 for l, v in neutral_by_layer.items()}

    for prompt_type, all_divs in results.items():
        if prompt_type == "neutral":
            continue

        pt_by_layer = {l: [] for l in range(num_layers)}
        for divs in all_divs:
            for d in divs:
                pt_by_layer[d.layer].append(1 - d.cosine_similarity)

        pt_avg = {l: sum(v) / len(v) if v else 0 for l, v in pt_by_layer.items()}

        # Find where tool divergence exceeds neutral by >10%
        for layer in range(num_layers):
            neutral_div = neutral_avg.get(layer, 0)
            pt_div = pt_avg.get(layer, 0)

            if pt_div > neutral_div * 1.1 and pt_div > 0.05:
                pct = (layer + 1) / num_layers * 100
                emergence[prompt_type] = layer
                print(f"\n{prompt_type}: Emergence at layer {layer} ({pct:.1f}% depth)")
                break
        else:
            emergence[prompt_type] = None
            print(f"\n{prompt_type}: No clear emergence point")

    return emergence


def main():
    parser = argparse.ArgumentParser(description="Activation Divergence Analysis")
    parser.add_argument("--base", default=None, help="Base model ID")
    parser.add_argument("--ft", default=None, help="Fine-tuned model ID")
    parser.add_argument("--pair", choices=list(DEFAULT_PAIRS.keys()), default="gemma",
                        help="Predefined model pair")
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
    print("Activation Divergence Analysis")
    print(f"Base: {base_id}")
    print(f"Fine-tuned: {ft_id}")
    print("=" * 80)

    # Load models
    base_model, _, base_config, _ = load_model(base_id)
    ft_model, ft_tokenizer, ft_config, _ = load_model(ft_id)
    template = load_chat_template(ft_id)

    tokenizer = ft_tokenizer
    num_layers = ft_config.num_hidden_layers

    # Define test prompts
    test_prompts = {
        "neutral": [
            "The capital of France is",
            "Once upon a time there was",
            "The color of the sky is",
            "2 + 2 equals",
        ],
        "tool_suggestive": [
            "What is the weather in",
            "Get the current temperature for",
            "Search for restaurants near",
            "Send an email to",
        ],
        "tool_explicit": [],
    }

    # Generate tool prompts with template if available
    if template:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }]

        for query in ["What is the weather in Tokyo?", "Tell me the weather in London"]:
            prompt = format_tool_prompt(template, query, tools)
            test_prompts["tool_explicit"].append(prompt)

    # Analyze
    results = analyze_divergence(base_model, ft_model, tokenizer, test_prompts)

    # Print summary
    print_divergence_summary(results, num_layers)

    # Find emergence layers
    print("\n" + "=" * 80)
    print("EMERGENCE ANALYSIS")
    print("=" * 80)
    emergence = find_emergence_layer(results, num_layers)

    # Save results
    output_path = Path("activation_divergence_results.json")
    output_data = {
        prompt_type: [
            [{"layer": d.layer, "cos_sim": d.cosine_similarity, "l2": d.l2_distance}
             for d in divs]
            for divs in all_divs
        ]
        for prompt_type, all_divs in results.items()
    }
    output_data["emergence"] = emergence
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    del base_model, ft_model
    mx.metal.clear_cache()


if __name__ == "__main__":
    main()
