#!/usr/bin/env python3
"""
Expert Capability Probing for Virtual Expert Strategy

This script probes which experts activate for different task categories
to identify:
1. EXTERNALIZABLE experts (math, time, weather) - can be replaced by tools
2. FLUENCY experts (language, style) - must be kept for quality

The goal is to flip the compression strategy:
- OLD: Prune by frequency → Lose fluency, keep bad math
- NEW: Prune by externalizability → Keep fluency, externalize math to tools

Usage:
    python probe_expert_capabilities.py --model openai/gpt-oss-120b
    python probe_expert_capabilities.py --model openai/gpt-oss-120b --detailed
"""

from __future__ import annotations

import argparse
import json
import gc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# =============================================================================
# Task Categories - Externalizable vs Core LLM
# =============================================================================

TASK_PROMPTS = {
    # =========================================================================
    # EXTERNALIZABLE TASKS (candidates for virtual experts)
    # =========================================================================

    "arithmetic": {
        "externalizable": True,
        "tool": "calculator",
        "prompts": [
            "127 * 89 = ",
            "456 + 789 = ",
            "1024 / 16 = ",
            "15% of 200 is ",
            "2^10 = ",
            "sqrt(144) = ",
            "7 * 8 + 3 = ",
            "999 - 123 = ",
            "What is 13 * 17?",
            "Calculate 81 / 9",
        ],
    },

    "symbolic_math": {
        "externalizable": True,
        "tool": "sympy",
        "prompts": [
            "Solve for x: 2x + 5 = 15",
            "Factor: x^2 - 9",
            "Derivative of x^3",
            "Integral of 2x dx",
            "Simplify: (x+1)(x-1)",
            "Expand: (a+b)^2",
            "Solve: x^2 = 16",
            "Find the roots of x^2 - 5x + 6",
        ],
    },

    "datetime": {
        "externalizable": True,
        "tool": "system_clock",
        "prompts": [
            "What day is today?",
            "What time is it?",
            "How many days until Christmas?",
            "What day of the week is January 1, 2025?",
            "How many hours in 3 days?",
            "What month comes after July?",
            "Days between March 1 and March 15?",
            "Is 2024 a leap year?",
        ],
    },

    "current_data": {
        "externalizable": True,
        "tool": "web_api",
        "prompts": [
            "What is the weather today?",
            "What is the current stock price of Apple?",
            "Who won the game last night?",
            "What is the latest news?",
            "Current temperature in Tokyo?",
            "What is the exchange rate USD to EUR?",
            "Is it raining in London right now?",
            "What time is it in Sydney?",
        ],
    },

    "code_execution": {
        "externalizable": True,
        "tool": "interpreter",
        "prompts": [
            "Run this Python code: print(2+2)",
            "What does this output: [x**2 for x in range(5)]",
            "Execute: sorted([3,1,4,1,5,9])",
            "What is the result of len('hello')?",
            "Run: sum(range(10))",
            "Execute this and show output: 'hello'.upper()",
        ],
    },

    "unit_conversion": {
        "externalizable": True,
        "tool": "pint",
        "prompts": [
            "Convert 100 meters to feet",
            "How many kilograms in 10 pounds?",
            "Convert 32 Fahrenheit to Celsius",
            "How many milliliters in a gallon?",
            "Convert 60 mph to km/h",
            "How many inches in 2 meters?",
        ],
    },

    # =========================================================================
    # CORE LLM TASKS (must keep these experts)
    # =========================================================================

    "language_fluency": {
        "externalizable": False,
        "tool": None,
        "prompts": [
            "Once upon a time, in a land far away,",
            "The quick brown fox jumps over the",
            "To be or not to be, that is the",
            "It was the best of times, it was the",
            "In the beginning, there was",
            "She walked into the room and",
            "The rain fell softly on the",
            "As the sun set over the horizon,",
            "He opened the letter and read:",
            "The old house stood at the end of",
        ],
    },

    "style_tone": {
        "externalizable": False,
        "tool": None,
        "prompts": [
            "Write this formally: gonna grab some food",
            "Make this casual: I would like to request",
            "Say this sarcastically: What a great idea",
            "Write this as a poem: the cat sat on the mat",
            "Make this dramatic: He left the room",
            "Write this as a news headline: Dog saves child",
            "Make this sound academic: People like pizza",
            "Write this humorously: I'm tired",
        ],
    },

    "reasoning": {
        "externalizable": False,
        "tool": None,
        "prompts": [
            "If all cats are mammals, and all mammals are animals, then",
            "The argument fails because",
            "On one hand... on the other hand",
            "The main difference between A and B is",
            "This suggests that",
            "Therefore, we can conclude",
            "The evidence indicates",
            "Considering the alternatives,",
        ],
    },

    "world_knowledge": {
        "externalizable": False,  # Static knowledge is core LLM
        "tool": None,
        "prompts": [
            "The capital of France is",
            "Shakespeare wrote",
            "Water freezes at",
            "The chemical symbol for gold is",
            "Photosynthesis is the process by which",
            "The Great Wall of China was built",
            "DNA stands for",
            "The Pythagorean theorem states",
        ],
    },

    "code_understanding": {
        "externalizable": False,  # Understanding != execution
        "tool": None,
        "prompts": [
            "This code has a bug because",
            "The time complexity of this algorithm is",
            "This function does the following:",
            "To improve this code, I would",
            "The purpose of this class is",
            "This pattern is called",
            "The difference between recursion and iteration is",
            "This code follows the principle of",
        ],
    },

    "creative": {
        "externalizable": False,
        "tool": None,
        "prompts": [
            "Write a haiku about",
            "Describe a sunset in",
            "Create a metaphor for",
            "Invent a new word that means",
            "Write the opening line of a mystery",
            "Describe happiness without using happy",
            "Create a slogan for",
            "Write a limerick about",
        ],
    },
}


@dataclass
class ExpertActivation:
    """Activation data for an expert."""
    layer: int
    expert_idx: int
    activation_count: int = 0
    categories: dict[str, int] = field(default_factory=dict)

    @property
    def primary_category(self) -> str:
        if not self.categories:
            return "unknown"
        return max(self.categories, key=self.categories.get)

    @property
    def externalizable_score(self) -> float:
        """Score 0-1 indicating how externalizable this expert is."""
        if not self.categories:
            return 0.5

        external_count = sum(
            count for cat, count in self.categories.items()
            if TASK_PROMPTS.get(cat, {}).get("externalizable", False)
        )
        total = sum(self.categories.values())
        return external_count / total if total > 0 else 0.5


@dataclass
class CapabilityProfile:
    """Complete capability profile for the model."""
    model_name: str
    num_layers: int
    num_experts: int
    expert_activations: dict[tuple[int, int], ExpertActivation]
    category_experts: dict[str, list[tuple[int, int, int]]]  # category -> [(layer, expert, count)]
    externalizable_experts: list[tuple[int, int, float]]  # (layer, expert, score)
    fluency_experts: list[tuple[int, int, float]]  # (layer, expert, score)


def probe_expert_activations(
    model,
    tokenizer,
    prompts: dict[str, dict],
    verbose: bool = True,
) -> CapabilityProfile:
    """
    Probe which experts activate for each task category.
    """
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = model.args.num_local_experts
    top_k = model.args.num_experts_per_tok
    hidden_size = model.args.hidden_size

    if verbose:
        print(f"Probing {num_layers} layers, {num_experts} experts, top-{top_k}")
        print()

    # Initialize tracking
    expert_activations: dict[tuple[int, int], ExpertActivation] = {}
    for layer_idx in range(num_layers):
        for exp_idx in range(num_experts):
            expert_activations[(layer_idx, exp_idx)] = ExpertActivation(
                layer=layer_idx,
                expert_idx=exp_idx,
            )

    category_experts: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    # Process each category
    for category, config in prompts.items():
        category_prompts = config["prompts"]
        externalizable = config["externalizable"]
        tool = config.get("tool", "none")

        if verbose:
            ext_str = f"[EXTERNAL:{tool}]" if externalizable else "[CORE LLM]"
            print(f"Probing: {category} {ext_str} ({len(category_prompts)} prompts)")

        # Track activations for this category
        category_counts: dict[tuple[int, int], int] = defaultdict(int)

        for prompt in category_prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            if hasattr(model, 'model'):
                h = model.model.embed_tokens(input_ids)
            else:
                h = model.embed_tokens(input_ids)

            batch_size, seq_len, _ = h.shape
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

            for layer_idx, layer in enumerate(layers):
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                mlp = layer.mlp
                if hasattr(mlp, 'router'):
                    x_flat = normed.reshape(-1, hidden_size)
                    logits = mlp.router(x_flat)
                    top_k_indices = mx.argsort(logits, axis=-1)[:, -top_k:]

                    for tok_idx in range(seq_len):
                        for k in range(top_k):
                            expert_idx = int(top_k_indices[tok_idx, k])
                            category_counts[(layer_idx, expert_idx)] += 1

                h = layer(h, mask=mask)

            mx.eval(h)

        # Update expert activations with category data
        for (layer_idx, exp_idx), count in category_counts.items():
            ea = expert_activations[(layer_idx, exp_idx)]
            ea.activation_count += count
            ea.categories[category] = ea.categories.get(category, 0) + count
            category_experts[category].append((layer_idx, exp_idx, count))

        gc.collect()

    # Sort category experts by activation count
    for category in category_experts:
        category_experts[category].sort(key=lambda x: x[2], reverse=True)

    # Identify externalizable vs fluency experts
    externalizable_experts = []
    fluency_experts = []

    for (layer_idx, exp_idx), ea in expert_activations.items():
        if ea.activation_count == 0:
            continue

        score = ea.externalizable_score
        if score > 0.6:  # Primarily handles externalizable tasks
            externalizable_experts.append((layer_idx, exp_idx, score))
        elif score < 0.4:  # Primarily handles core LLM tasks
            fluency_experts.append((layer_idx, exp_idx, 1 - score))

    # Sort by score
    externalizable_experts.sort(key=lambda x: x[2], reverse=True)
    fluency_experts.sort(key=lambda x: x[2], reverse=True)

    return CapabilityProfile(
        model_name=model.args.model_type if hasattr(model.args, 'model_type') else "unknown",
        num_layers=num_layers,
        num_experts=num_experts,
        expert_activations=expert_activations,
        category_experts=dict(category_experts),
        externalizable_experts=externalizable_experts,
        fluency_experts=fluency_experts,
    )


def print_capability_report(profile: CapabilityProfile):
    """Print detailed capability report."""
    print()
    print("=" * 80)
    print("EXPERT CAPABILITY ANALYSIS")
    print("=" * 80)
    print()
    print(f"Model: {profile.model_name}")
    print(f"Architecture: {profile.num_layers} layers × {profile.num_experts} experts")
    print()

    # Summary
    total_experts = profile.num_layers * profile.num_experts
    active_experts = sum(1 for ea in profile.expert_activations.values() if ea.activation_count > 0)

    print(f"Active experts: {active_experts}/{total_experts} ({100*active_experts/total_experts:.1f}%)")
    print(f"Externalizable experts: {len(profile.externalizable_experts)}")
    print(f"Fluency experts: {len(profile.fluency_experts)}")
    print()

    # Top experts by category
    print("-" * 80)
    print("TOP EXPERTS BY CATEGORY")
    print("-" * 80)
    print()

    for category, config in TASK_PROMPTS.items():
        ext_label = "[EXTERNAL]" if config["externalizable"] else "[CORE]"
        tool = config.get("tool", "-")

        experts = profile.category_experts.get(category, [])[:5]
        if not experts:
            continue

        print(f"{category} {ext_label} (tool: {tool})")
        for layer, exp, count in experts:
            print(f"  L{layer:2d}/E{exp:3d}: {count} activations")
        print()

    # Externalizable experts (candidates for removal)
    print("-" * 80)
    print("EXTERNALIZABLE EXPERTS (candidates for virtual expert replacement)")
    print("-" * 80)
    print()

    # Group by layer
    ext_by_layer: dict[int, list] = defaultdict(list)
    for layer, exp, score in profile.externalizable_experts[:50]:
        ext_by_layer[layer].append((exp, score))

    for layer in sorted(ext_by_layer.keys()):
        experts = ext_by_layer[layer]
        exp_str = ", ".join(f"E{e}({s:.2f})" for e, s in experts[:5])
        print(f"  Layer {layer:2d}: {exp_str}")

    print()
    print(f"Total externalizable: {len(profile.externalizable_experts)} experts")
    print("These experts primarily handle: arithmetic, datetime, current data, code execution")
    print("Strategy: REMOVE these and route to virtual experts (tools)")
    print()

    # Fluency experts (must keep)
    print("-" * 80)
    print("FLUENCY EXPERTS (must keep for language quality)")
    print("-" * 80)
    print()

    flu_by_layer: dict[int, list] = defaultdict(list)
    for layer, exp, score in profile.fluency_experts[:50]:
        flu_by_layer[layer].append((exp, score))

    for layer in sorted(flu_by_layer.keys()):
        experts = flu_by_layer[layer]
        exp_str = ", ".join(f"E{e}({s:.2f})" for e, s in experts[:5])
        print(f"  Layer {layer:2d}: {exp_str}")

    print()
    print(f"Total fluency experts: {len(profile.fluency_experts)} experts")
    print("These experts primarily handle: language fluency, style, reasoning, creativity")
    print("Strategy: KEEP these for language model quality")
    print()

    # Virtual expert recommendations
    print("=" * 80)
    print("VIRTUAL EXPERT RECOMMENDATIONS")
    print("=" * 80)
    print()

    recommendations = [
        ("calculator", "arithmetic, symbolic_math", "sympy, python eval"),
        ("datetime", "datetime", "system clock, dateutil"),
        ("web_api", "current_data", "weather API, search API"),
        ("interpreter", "code_execution", "sandboxed python"),
        ("unit_converter", "unit_conversion", "pint library"),
    ]

    for name, categories, backend in recommendations:
        # Count experts that could be replaced
        replaceable = 0
        for cat in categories.split(", "):
            replaceable += len(profile.category_experts.get(cat, []))

        print(f"Virtual Expert: {name}")
        print(f"  Replaces: {categories}")
        print(f"  Backend: {backend}")
        print(f"  Experts affected: ~{replaceable//5} per layer")
        print()

    # Compression strategy
    print("=" * 80)
    print("NEW COMPRESSION STRATEGY")
    print("=" * 80)
    print()
    print("OLD: Prune by frequency → Lose fluency, keep bad math")
    print("NEW: Prune by externalizability → Keep fluency, externalize math")
    print()
    print("Estimated impact:")
    print(f"  - Remove {len(profile.externalizable_experts)} externalizable experts")
    print(f"  - Keep {len(profile.fluency_experts)} fluency experts")
    print(f"  - Add 5-6 virtual experts (external tools)")
    print()
    print("Expected outcome:")
    print("  - Perplexity: IMPROVED (keeping fluency experts)")
    print("  - Math accuracy: 100% (calculator is exact)")
    print("  - Current data: ACCURATE (live APIs)")
    print("  - Code execution: CORRECT (actual interpreter)")
    print()


def save_capability_profile(profile: CapabilityProfile, output_path: Path):
    """Save capability profile to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": profile.model_name,
        "num_layers": profile.num_layers,
        "num_experts": profile.num_experts,
        "summary": {
            "total_experts": profile.num_layers * profile.num_experts,
            "externalizable_count": len(profile.externalizable_experts),
            "fluency_count": len(profile.fluency_experts),
        },
        "externalizable_experts": [
            {"layer": l, "expert": e, "score": s}
            for l, e, s in profile.externalizable_experts
        ],
        "fluency_experts": [
            {"layer": l, "expert": e, "score": s}
            for l, e, s in profile.fluency_experts
        ],
        "category_top_experts": {
            cat: [{"layer": l, "expert": e, "count": c} for l, e, c in experts[:10]]
            for cat, experts in profile.category_experts.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Profile saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Expert Capability Probing")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--output", type=str, default="results/capability_profile_120b.json")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Load model
    from mlx_lm import load
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    # Probe capabilities
    print()
    profile = probe_expert_activations(model, tokenizer, TASK_PROMPTS, verbose=True)

    # Print report
    print_capability_report(profile)

    # Save profile
    save_capability_profile(profile, Path(args.output))


if __name__ == "__main__":
    main()
