#!/usr/bin/env python3
"""
Head Ablation - Causal Intervention on Attention Heads

Test causality by ablating (zeroing out) specific attention heads and
observing whether tool-calling behavior breaks.

Run: uv run python examples/introspection/head_ablation.py
     uv run python examples/introspection/head_ablation.py --model mlx-community/functiongemma-270m-it-bf16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from _loader import (
    format_tool_prompt,
    generate,
    generate_with_head_ablation,
    has_tool_call,
    load_chat_template,
    load_model,
)


@dataclass
class AblationResult:
    """Result of a head ablation experiment."""

    experiment_name: str
    heads_ablated: list[tuple[int, int]]
    original_output: str
    ablated_output: str
    original_has_tool: bool
    ablated_has_tool: bool
    broken: bool


def main():
    parser = argparse.ArgumentParser(description="Head Ablation Study")
    parser.add_argument(
        "--model", default="mlx-community/functiongemma-270m-it-bf16", help="Model ID to test"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Head Ablation Study")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model
    model, tokenizer, config, _ = load_model(args.model)
    template = load_chat_template(args.model)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    if not template:
        print("ERROR: Could not load chat template")
        return

    # Create test prompts
    tools = [
        {
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
        }
    ]

    prompts = [
        format_tool_prompt(template, "What is the weather in Tokyo?", tools),
        format_tool_prompt(template, "Tell me the weather in London", tools),
    ]

    # Define experiments - progressive multi-head ablation
    # Focus on late layers typically important for tool-calling
    late_start = int(num_layers * 0.8)
    mid_layer = int(num_layers * 0.6)

    experiments = [
        # Single heads in late layers
        (f"L{num_layers - 1}-H0", [(num_layers - 1, 0)]),
        (f"L{num_layers - 2}-H0", [(num_layers - 2, 0)]),
        # Full layer ablation
        (f"L{num_layers - 1} all heads", [(num_layers - 1, h) for h in range(num_heads)]),
        (f"L{num_layers - 2} all heads", [(num_layers - 2, h) for h in range(num_heads)]),
        # Two layer ablation
        (
            f"L{num_layers - 2}+L{num_layers - 1} all",
            [(num_layers - 2, h) for h in range(num_heads)]
            + [(num_layers - 1, h) for h in range(num_heads)],
        ),
        # Mid layer (control)
        (f"L{mid_layer} all heads (control)", [(mid_layer, h) for h in range(num_heads)]),
    ]

    print(f"\nModel: {num_layers} layers, {num_heads} heads")
    print(f"Running {len(experiments)} ablation experiments...")

    all_results = []

    for exp_name, heads in experiments:
        print(f"\n{exp_name}:")
        print(f"  Ablating: {len(heads)} heads")

        exp_results = []
        for i, prompt in enumerate(prompts):
            # Generate without ablation
            original = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)

            # Generate with ablation
            ablated = generate_with_head_ablation(
                model, tokenizer, prompt, heads, max_new_tokens=50
            )

            original_has = has_tool_call(original)
            ablated_has = has_tool_call(ablated)

            result = AblationResult(
                experiment_name=exp_name,
                heads_ablated=heads,
                original_output=original,
                ablated_output=ablated,
                original_has_tool=original_has,
                ablated_has_tool=ablated_has,
                broken=original_has and not ablated_has,
            )
            exp_results.append(result)

            status = "BROKE" if result.broken else "survives"
            print(f"  Prompt {i + 1}: {status}")

        all_results.extend(exp_results)

        # Summary
        broken = sum(1 for r in exp_results if r.broken)
        total = len(exp_results)
        if broken == total:
            print(f"  >>> CAUSAL: {broken}/{total} broken")
        elif broken > 0:
            print(f"  >>> PARTIAL: {broken}/{total} broken")
        else:
            print(f"  >>> SURVIVES: 0/{total} broken")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Experiment':<35} {'Heads':<8} {'Broken':<10} {'Status'}")
    print("-" * 65)

    for exp_name, heads in experiments:
        exp_results = [r for r in all_results if r.experiment_name == exp_name]
        broken = sum(1 for r in exp_results if r.broken)
        total = len(exp_results)

        if broken == total:
            status = "*** CAUSAL ***"
        elif broken > 0:
            status = "partial"
        else:
            status = "survives"

        print(f"{exp_name:<35} {len(heads):<8} {broken}/{total:<8} {status}")

    # Save results
    output_path = Path("head_ablation_results.json")
    output_data = [
        {
            "experiment_name": r.experiment_name,
            "heads_ablated": r.heads_ablated,
            "original_output": r.original_output[:200],
            "ablated_output": r.ablated_output[:200],
            "broken": r.broken,
        }
        for r in all_results
    ]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
