#!/usr/bin/env python3
"""
MLP Ablation Study - Where Does the Tool Decision Live?

Test MLP ablation at different layers to find where the tool-calling
decision actually lives. Works with any supported model family.

Run: uv run python examples/introspection/mlp_ablation.py
     uv run python examples/introspection/mlp_ablation.py --model mlx-community/functiongemma-270m-it-bf16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from _loader import (
    format_tool_prompt,
    generate,
    generate_with_layer_ablation,
    has_tool_call,
    load_chat_template,
    load_model,
)


@dataclass
class AblationResult:
    """Result of an ablation experiment."""
    experiment_name: str
    layer: int
    original_output: str
    ablated_output: str
    original_has_tool: bool
    ablated_has_tool: bool
    broken: bool
    coherent: bool


def is_coherent(text: str) -> bool:
    """Check if output is coherent (not gibberish)."""
    if text.count("<escape>") > 5:
        return False
    if text.count("\n") > 20 and len(text) < 100:
        return False
    if len(set(text)) < 10 and len(text) > 50:
        return False
    return True


def run_single_ablation(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    ablation_type: str = "mlp",
) -> AblationResult:
    """Run a single ablation experiment."""
    # Generate without ablation
    original = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)

    # Generate with ablation
    ablated = generate_with_layer_ablation(
        model, tokenizer, prompt,
        ablate_layer=layer,
        ablation_type=ablation_type,
        max_new_tokens=50,
    )

    original_has = has_tool_call(original)
    ablated_has = has_tool_call(ablated)

    return AblationResult(
        experiment_name=f"{ablation_type.upper()} L{layer}",
        layer=layer,
        original_output=original,
        ablated_output=ablated,
        original_has_tool=original_has,
        ablated_has_tool=ablated_has,
        broken=original_has and not ablated_has,
        coherent=is_coherent(ablated),
    )


def main():
    parser = argparse.ArgumentParser(description="MLP Ablation Study")
    parser.add_argument("--model", default="mlx-community/functiongemma-270m-it-bf16",
                        help="Model ID to test")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices to test")
    args = parser.parse_args()

    print("=" * 80)
    print("MLP Ablation Study")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model
    model, tokenizer, config, _ = load_model(args.model)
    template = load_chat_template(args.model)
    num_layers = config.num_hidden_layers

    if not template:
        print("ERROR: Could not load chat template")
        return

    # Create test prompts
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

    prompts = [
        format_tool_prompt(template, "What is the weather in Tokyo?", tools),
        format_tool_prompt(template, "Tell me the weather in London", tools),
    ]

    # Determine layers to test
    if args.layers:
        layers_to_test = [int(x) for x in args.layers.split(",")]
    else:
        # Default: key layers based on typical architecture
        layers_to_test = [
            int(num_layers * 0.3),   # ~30% depth
            int(num_layers * 0.5),   # ~50% depth
            int(num_layers * 0.6),   # ~60% depth (typical decision zone)
            int(num_layers * 0.7),   # ~70% depth
            num_layers - 2,          # near-final
            num_layers - 1,          # final
        ]
        layers_to_test = sorted(set(layers_to_test))

    print(f"\nTesting layers: {layers_to_test}")

    # Run experiments
    all_results = []
    for layer in layers_to_test:
        print(f"\nLayer {layer} ({(layer+1)/num_layers*100:.0f}% depth):")

        for i, prompt in enumerate(prompts):
            result = run_single_ablation(model, tokenizer, prompt, layer, "mlp")
            all_results.append(result)

            status = "BROKE" if result.broken else "survives"
            coherent = "coherent" if result.coherent else "gibberish"
            print(f"  Prompt {i+1}: {status} ({coherent})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Layer':<8} {'Depth':>8} {'Broken':>10} {'Coherent':>10} {'Status'}")
    print("-" * 50)

    for layer in layers_to_test:
        layer_results = [r for r in all_results if r.layer == layer]
        broken = sum(1 for r in layer_results if r.broken)
        coherent = sum(1 for r in layer_results if r.broken and r.coherent)
        total = len(layer_results)
        pct = (layer + 1) / num_layers * 100

        if broken == total:
            if coherent == total:
                status = "*** CAUSAL ***"
            else:
                status = "BREAKS MODEL"
        elif broken > 0:
            status = "partial"
        else:
            status = "survives"

        print(f"{layer:<8} {pct:>7.0f}% {broken}/{total:<8} {coherent}/{broken if broken else '-':<8} {status}")

    # Find causal layers
    causal_layers = []
    for layer in layers_to_test:
        layer_results = [r for r in all_results if r.layer == layer]
        if all(r.broken and r.coherent for r in layer_results):
            causal_layers.append(layer)

    if causal_layers:
        print(f"\nCAUSAL layers (break tool-calling with coherent fallback): {causal_layers}")
        pct = [(l + 1) / num_layers * 100 for l in causal_layers]
        print(f"Relative positions: {[f'{p:.0f}%' for p in pct]}")

    # Save results
    output_path = Path("mlp_ablation_results.json")
    output_data = [
        {
            "layer": r.layer,
            "original_output": r.original_output[:200],
            "ablated_output": r.ablated_output[:200],
            "broken": r.broken,
            "coherent": r.coherent,
        }
        for r in all_results
    ]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
