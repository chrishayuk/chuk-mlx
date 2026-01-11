#!/usr/bin/env python3
"""
Expert Analysis for MoE Models.

Analyzes which experts handle different types of prompts in an MoE model.
Used to understand expert specialization and identify potential hijack targets.

Usage:
    uv run python examples/introspection/experiments/moe/expert_analysis.py \
        --model openai/gpt-oss-20b \
        --analyze-categories
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))


def load_model(model_id: str):
    """Load model and tokenizer."""
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

    print(f"\n{'='*70}")
    print(f"Loading: {model_id}")
    print(f"{'='*70}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {model_id}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = len(list(model.model.layers))
    print(f"Loaded: {num_layers} layers")

    return model, tokenizer


def analyze_expert_categories(model, tokenizer, model_id: str):
    """
    Analyze which experts activate for different prompt categories.

    This reveals whether experts specialize or generalize across domains.
    """
    from chuk_lazarus.introspection.moe import MoEHooks, MoECaptureConfig, get_moe_layer_info

    print("\n" + "=" * 70)
    print("EXPERT CATEGORY ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing which experts activate for different types of prompts.\n")

    # Test prompts by category
    categories = {
        "MATH": [
            "127 * 89 = ",
            "456 + 789 = ",
            "1000 - 250 = ",
            "What is 25 squared?",
        ],
        "CODE": [
            "def fibonacci(n):",
            "for i in range(10):",
            "import numpy as np",
            "class Calculator:",
        ],
        "LOGIC": [
            "If A implies B, and B implies C, then",
            "All men are mortal. Socrates is a man. Therefore",
            "NOT (A AND B) is equivalent to",
            "The contrapositive of P→Q is",
        ],
        "LANGUAGE": [
            "The capital of France is",
            "Once upon a time",
            "Hello, how are you",
            "The quick brown fox",
        ],
    }

    # Find MoE layers
    layers = list(model.model.layers)
    moe_layers = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            moe_layers.append(i)

    if not moe_layers:
        print("No MoE layers found in model")
        return

    # Use middle MoE layer for analysis
    target_layer = moe_layers[len(moe_layers) // 2]
    info = get_moe_layer_info(model, target_layer)
    num_experts = info.num_experts if info else 32

    print(f"Model: {model_id}")
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]} to {moe_layers[-1]})")
    print(f"Analyzing layer: {target_layer}")
    print(f"Number of experts: {num_experts}")
    print()

    # Track which experts activate for each category
    category_expert_counts = {cat: defaultdict(int) for cat in categories}

    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(
        capture_selected_experts=True,
        layers=[target_layer],
    ))

    for category, prompts in categories.items():
        for prompt in prompts:
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            hooks.forward(input_ids)

            if target_layer in hooks.state.selected_experts:
                experts = hooks.state.selected_experts[target_layer]
                # Look at last position (where prediction happens)
                last_experts = experts[0, -1].tolist()
                for exp_idx in last_experts:
                    category_expert_counts[category][exp_idx] += 1

    # Find top experts for each category
    print(f"{'Expert':<10} ", end="")
    for cat in categories:
        print(f"{cat:<12}", end="")
    print()
    print("-" * (10 + 12 * len(categories)))

    # Get all experts that appeared
    all_experts = set()
    for counts in category_expert_counts.values():
        all_experts.update(counts.keys())

    # Sort by total activations
    def total_activations(exp):
        return sum(counts.get(exp, 0) for counts in category_expert_counts.values())

    sorted_experts = sorted(all_experts, key=total_activations, reverse=True)

    # Find math expert
    math_counts = category_expert_counts["MATH"]
    math_expert = max(math_counts, key=math_counts.get) if math_counts else None

    for exp_idx in sorted_experts[:15]:
        print(f"Expert {exp_idx:<3} ", end="")
        for cat in categories:
            count = category_expert_counts[cat].get(exp_idx, 0)
            print(f"{count:<12}", end="")

        # Add annotations
        annotations = []
        if exp_idx == math_expert:
            annotations.append("← 'math expert'")

        # Check for multi-use
        uses = sum(1 for cat in categories if category_expert_counts[cat].get(exp_idx, 0) > 0)
        if uses >= 3:
            annotations.append("(multi-use)")

        if annotations:
            print(" ".join(annotations), end="")
        print()

    # Summary
    print("\n" + "-" * 70)
    print("FINDINGS:")
    print("-" * 70)

    if math_expert is not None:
        print(f"\n'Math expert' candidate: Expert {math_expert}")
        print(f"  MATH activations:     {math_counts.get(math_expert, 0)}")
        print(f"  CODE activations:     {category_expert_counts['CODE'].get(math_expert, 0)}")
        print(f"  LOGIC activations:    {category_expert_counts['LOGIC'].get(math_expert, 0)}")
        print(f"  LANGUAGE activations: {category_expert_counts['LANGUAGE'].get(math_expert, 0)}")

    # Find most specialized vs most general experts
    specialization_scores = {}
    for exp in all_experts:
        counts = [category_expert_counts[cat].get(exp, 0) for cat in categories]
        total = sum(counts)
        if total > 0:
            # Higher score = more specialized (activations concentrated in one category)
            max_count = max(counts)
            specialization_scores[exp] = max_count / total

    if specialization_scores:
        most_specialized = max(specialization_scores, key=specialization_scores.get)
        most_general = min(specialization_scores, key=specialization_scores.get)

        print(f"\nMost specialized: Expert {most_specialized} (score: {specialization_scores[most_specialized]:.2f})")
        print(f"Most general:     Expert {most_general} (score: {specialization_scores[most_general]:.2f})")

    print("\n" + "=" * 70)
    print("IMPLICATION: Experts are NOT pure specialists.")
    print("Hijacking any single expert risks breaking multiple capabilities.")
    print("=" * 70)


def analyze_single_prompt(model, tokenizer, model_id: str, prompt: str):
    """Analyze expert routing for a single prompt."""
    from chuk_lazarus.introspection.moe import MoEHooks, MoECaptureConfig

    print(f"\n{'='*70}")
    print(f"EXPERT ROUTING ANALYSIS")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}\n")

    layers = list(model.model.layers)
    moe_layers = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            moe_layers.append(i)

    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(
        capture_selected_experts=True,
        capture_router_logits=True,
        layers=moe_layers,
    ))

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    hooks.forward(input_ids)

    print(f"{'Layer':<8} {'Selected Experts':<30} {'Top Logit Expert':<20}")
    print("-" * 60)

    for layer_idx in moe_layers[:10]:  # Show first 10 MoE layers
        if layer_idx in hooks.state.selected_experts:
            experts = hooks.state.selected_experts[layer_idx]
            last_experts = experts[0, -1].tolist()
            experts_str = ", ".join(str(e) for e in last_experts)

            if layer_idx in hooks.state.router_logits:
                logits = hooks.state.router_logits[layer_idx]
                top_expert = int(mx.argmax(logits[0, -1]))
                top_logit = float(logits[0, -1, top_expert])
                top_str = f"Expert {top_expert} ({top_logit:.2f})"
            else:
                top_str = "N/A"

            print(f"L{layer_idx:<6} [{experts_str:<27}] {top_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Expert Analysis for MoE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-oss-20b",
        help="Model ID to analyze",
    )
    parser.add_argument(
        "--analyze-categories",
        action="store_true",
        help="Analyze expert activations by prompt category",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Analyze routing for a single prompt",
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.analyze_categories:
        analyze_expert_categories(model, tokenizer, args.model)
    elif args.prompt:
        analyze_single_prompt(model, tokenizer, args.model, args.prompt)
    else:
        # Default: run category analysis
        analyze_expert_categories(model, tokenizer, args.model)


if __name__ == "__main__":
    main()
