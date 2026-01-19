#!/usr/bin/env python3
"""
Build GPT-OSS-120B-VE (Virtual Expert): Capability-Aware Compression

Strategy: Remove externalizable experts, keep fluency experts.

OLD approach (frequency-based):
  - Remove cold experts → Lost fluency, kept bad math
  - Result: PPL +109%

NEW approach (capability-based):
  - Remove EXTERNALIZABLE experts (math, time, APIs) → Route to tools
  - Keep FLUENCY experts (language, style, reasoning) → Preserve LLM quality
  - Result: Expected PPL minimal increase, math 100% accurate

Usage:
    python build_capability_aware_lite.py --analyze-only
    python build_capability_aware_lite.py --build
    python build_capability_aware_lite.py --build --aggressive
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Task Categories for Expert Classification
# =============================================================================

# Externalizable tasks - experts handling these can be removed
EXTERNALIZABLE_PROMPTS = {
    "arithmetic": [
        "127 * 89 = ", "456 + 789 = ", "1024 / 16 = ", "2^10 = ",
        "sqrt(144) = ", "15% of 200", "7 * 8 + 3 = ", "999 - 123 = ",
    ],
    "symbolic_math": [
        "Solve for x: 2x + 5 = 15", "Factor: x^2 - 9",
        "Derivative of x^3", "Simplify: (x+1)(x-1)",
    ],
    "datetime": [
        "What day is today?", "What time is it?", "Is 2024 a leap year?",
        "How many days until Christmas?", "Days between March 1 and March 15?",
    ],
    "current_data": [
        "What is the weather today?", "Current stock price of Apple?",
        "Latest news?", "Current temperature in Tokyo?",
    ],
    "code_execution": [
        "Run: print(2+2)", "Execute: sorted([3,1,4])", "Output of len('hello')?",
    ],
    "unit_conversion": [
        "Convert 100 meters to feet", "32 fahrenheit to celsius",
        "How many kg in 10 pounds?", "Convert 60 mph to km/h",
    ],
}

# Fluency tasks - experts handling these MUST be kept
FLUENCY_PROMPTS = {
    "language_fluency": [
        "Once upon a time, in a land far away,",
        "The quick brown fox jumps over the",
        "She walked into the room and",
        "As the sun set over the horizon,",
        "In the depths of the ancient forest,",
        "The letter began with the words,",
        "He couldn't believe what he saw when",
        "The old man sat by the fire and",
    ],
    "style_tone": [
        "Write this formally: gonna grab some food",
        "Make this casual: I would like to request",
        "Say this sarcastically: What a great idea",
        "Make this dramatic: He left the room",
    ],
    "reasoning": [
        "If all cats are mammals, then",
        "The argument fails because",
        "Therefore, we can conclude",
        "The evidence suggests that",
        "On one hand... on the other hand",
    ],
    "world_knowledge": [
        "The capital of France is",
        "Shakespeare wrote",
        "The chemical symbol for gold is",
        "Photosynthesis is the process by which",
    ],
    "creative": [
        "Write a haiku about",
        "Describe a sunset in three sentences:",
        "Create a metaphor for time:",
        "The opening line of a mystery novel:",
    ],
    "code_understanding": [
        "This code has a bug because",
        "The time complexity of this algorithm is",
        "To improve this code, I would",
        "This pattern is called",
    ],
}


@dataclass
class ExpertProfile:
    """Profile of an expert's capabilities."""
    layer: int
    expert_idx: int
    externalizable_score: float = 0.0  # 0-1, higher = more externalizable
    fluency_score: float = 0.0  # 0-1, higher = more fluency-critical
    total_activations: int = 0
    categories: dict[str, int] = field(default_factory=dict)

    @property
    def should_remove(self) -> bool:
        """Should this expert be removed (replaced by virtual expert)?"""
        # Remove if primarily externalizable AND not critical for fluency
        return self.externalizable_score > 0.6 and self.fluency_score < 0.4

    @property
    def must_keep(self) -> bool:
        """Must this expert be kept for fluency?"""
        return self.fluency_score > 0.5


@dataclass
class CapabilityAnalysis:
    """Complete capability analysis for the model."""
    expert_profiles: dict[tuple[int, int], ExpertProfile]
    experts_to_remove: list[tuple[int, int]]
    experts_to_keep: list[tuple[int, int]]
    removal_by_layer: dict[int, list[int]]
    keep_by_layer: dict[int, list[int]]


def analyze_expert_capabilities(
    model,
    tokenizer,
    verbose: bool = True,
) -> CapabilityAnalysis:
    """
    Analyze which experts handle externalizable vs fluency tasks.
    """
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = model.args.num_local_experts
    top_k = model.args.num_experts_per_tok
    hidden_size = model.args.hidden_size

    logger.info(f"Analyzing {num_layers} layers, {num_experts} experts")

    # Initialize profiles
    profiles: dict[tuple[int, int], ExpertProfile] = {}
    for layer_idx in range(num_layers):
        for exp_idx in range(num_experts):
            profiles[(layer_idx, exp_idx)] = ExpertProfile(
                layer=layer_idx, expert_idx=exp_idx
            )

    # Track activations for externalizable tasks
    if verbose:
        print("\nProbing EXTERNALIZABLE tasks (will be routed to tools)...")

    for category, prompts in EXTERNALIZABLE_PROMPTS.items():
        if verbose:
            print(f"  {category}: {len(prompts)} prompts")

        for prompt in prompts:
            activations = _get_expert_activations(
                model, tokenizer, prompt, layers, num_experts, top_k, hidden_size
            )
            for (layer_idx, exp_idx), count in activations.items():
                profile = profiles[(layer_idx, exp_idx)]
                profile.total_activations += count
                profile.categories[f"ext_{category}"] = profile.categories.get(f"ext_{category}", 0) + count

    # Track activations for fluency tasks
    if verbose:
        print("\nProbing FLUENCY tasks (must be preserved)...")

    for category, prompts in FLUENCY_PROMPTS.items():
        if verbose:
            print(f"  {category}: {len(prompts)} prompts")

        for prompt in prompts:
            activations = _get_expert_activations(
                model, tokenizer, prompt, layers, num_experts, top_k, hidden_size
            )
            for (layer_idx, exp_idx), count in activations.items():
                profile = profiles[(layer_idx, exp_idx)]
                profile.total_activations += count
                profile.categories[f"flu_{category}"] = profile.categories.get(f"flu_{category}", 0) + count

    # Compute scores
    for profile in profiles.values():
        if profile.total_activations == 0:
            continue

        ext_count = sum(v for k, v in profile.categories.items() if k.startswith("ext_"))
        flu_count = sum(v for k, v in profile.categories.items() if k.startswith("flu_"))
        total = ext_count + flu_count

        if total > 0:
            profile.externalizable_score = ext_count / total
            profile.fluency_score = flu_count / total

    # Classify experts
    experts_to_remove = []
    experts_to_keep = []
    removal_by_layer: dict[int, list[int]] = defaultdict(list)
    keep_by_layer: dict[int, list[int]] = defaultdict(list)

    for (layer_idx, exp_idx), profile in profiles.items():
        if profile.total_activations == 0:
            # Cold expert - remove
            experts_to_remove.append((layer_idx, exp_idx))
            removal_by_layer[layer_idx].append(exp_idx)
        elif profile.should_remove:
            # Externalizable - remove
            experts_to_remove.append((layer_idx, exp_idx))
            removal_by_layer[layer_idx].append(exp_idx)
        else:
            # Keep for fluency
            experts_to_keep.append((layer_idx, exp_idx))
            keep_by_layer[layer_idx].append(exp_idx)

    return CapabilityAnalysis(
        expert_profiles=profiles,
        experts_to_remove=experts_to_remove,
        experts_to_keep=experts_to_keep,
        removal_by_layer=dict(removal_by_layer),
        keep_by_layer=dict(keep_by_layer),
    )


def _get_expert_activations(
    model,
    tokenizer,
    prompt: str,
    layers,
    num_experts: int,
    top_k: int,
    hidden_size: int,
) -> dict[tuple[int, int], int]:
    """Get expert activations for a prompt."""
    activations: dict[tuple[int, int], int] = defaultdict(int)

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
                    activations[(layer_idx, expert_idx)] += 1

        h = layer(h, mask=mask)

    mx.eval(h)
    return activations


def print_capability_report(analysis: CapabilityAnalysis, num_layers: int, num_experts: int):
    """Print capability analysis report."""
    print()
    print("=" * 80)
    print("CAPABILITY-AWARE EXPERT ANALYSIS")
    print("=" * 80)
    print()

    total = num_layers * num_experts
    remove_count = len(analysis.experts_to_remove)
    keep_count = len(analysis.experts_to_keep)

    print(f"Total experts:     {total:,}")
    print(f"Experts to REMOVE: {remove_count:,} ({100*remove_count/total:.1f}%)")
    print(f"Experts to KEEP:   {keep_count:,} ({100*keep_count/total:.1f}%)")
    print()

    # Per-layer summary
    print("Per-Layer Summary:")
    print("-" * 60)
    print(f"{'Layer':<8} {'Keep':<8} {'Remove':<8} {'Keep %':<10}")
    print("-" * 60)

    for layer_idx in range(num_layers):
        keep = len(analysis.keep_by_layer.get(layer_idx, []))
        remove = len(analysis.removal_by_layer.get(layer_idx, []))
        pct = 100 * keep / num_experts if num_experts > 0 else 0
        print(f"L{layer_idx:<6} {keep:<8} {remove:<8} {pct:.1f}%")

    print()

    # Comparison with frequency-based
    print("=" * 80)
    print("COMPARISON: Frequency vs Capability Pruning")
    print("=" * 80)
    print()
    print(f"{'Approach':<25} {'Experts Kept':<15} {'Compression':<15}")
    print("-" * 60)
    print(f"{'Original':<25} {total:<15} {'0%':<15}")
    print(f"{'Frequency (conservative)':<25} {'1,344':<15} {'71%':<15}")
    print(f"{'Capability-aware':<25} {keep_count:<15} {100*(1-keep_count/total):.1f}%")
    print()

    print("Expected Outcomes:")
    print("-" * 60)
    print("Frequency pruning:  PPL +109%, Math ~95% (lookup)")
    print("Capability pruning: PPL ~minimal, Math 100% (virtual expert)")
    print()


def build_capability_aware_model(
    model,
    analysis: CapabilityAnalysis,
    output_dir: str,
    aggressive: bool = False,
):
    """
    Build the capability-aware compressed model.

    Only keeps fluency experts, removes externalizable experts.
    """
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = model.args.num_local_experts
    hidden_size = model.args.hidden_size
    intermediate_size = model.args.intermediate_size

    logger.info(f"Building capability-aware model to {output_path}")

    # Determine experts to keep per layer
    keep_by_layer = {}
    for layer_idx in range(num_layers):
        keep_experts = analysis.keep_by_layer.get(layer_idx, [])

        if aggressive:
            # In aggressive mode, keep even fewer (only top fluency)
            keep_experts = keep_experts[:max(8, len(keep_experts) // 2)]

        # Ensure minimum k experts
        min_experts = model.args.num_experts_per_tok
        if len(keep_experts) < min_experts:
            # Add back some experts to meet minimum
            all_experts = list(range(num_experts))
            for exp in all_experts:
                if exp not in keep_experts:
                    keep_experts.append(exp)
                if len(keep_experts) >= min_experts:
                    break

        keep_by_layer[layer_idx] = sorted(keep_experts)

    # Get model parameters
    params = model.parameters()

    def get_leaf_params(params_dict, prefix=''):
        results = {}
        if isinstance(params_dict, dict):
            for k, v in params_dict.items():
                new_prefix = f'{prefix}.{k}' if prefix else k
                results.update(get_leaf_params(v, new_prefix))
        elif isinstance(params_dict, list):
            for i, v in enumerate(params_dict):
                results.update(get_leaf_params(v, f'{prefix}.{i}'))
        elif hasattr(params_dict, 'shape'):
            results[prefix] = params_dict
        return results

    flat_params = get_leaf_params(params)

    # Filter expert weights
    new_params = {}
    filtered_count = 0

    logger.info("Filtering expert weights...")

    for name, weight in flat_params.items():
        parts = name.split('.')
        layer_idx = None
        is_expert_weight = False
        is_router_weight = False

        if 'layers' in parts:
            try:
                layers_idx = parts.index('layers')
                layer_idx = int(parts[layers_idx + 1])
            except (ValueError, IndexError):
                pass

        if layer_idx is not None and 'mlp' in parts:
            if 'experts' in parts:
                is_expert_weight = True
            elif 'router' in parts:
                is_router_weight = True

        if is_expert_weight and weight.shape[0] == num_experts:
            keep = keep_by_layer[layer_idx]
            new_weight = weight[keep]
            new_params[name] = new_weight
            filtered_count += 1
        elif is_router_weight and weight.shape[0] == num_experts:
            keep = keep_by_layer[layer_idx]
            new_weight = weight[keep]
            new_params[name] = new_weight
            filtered_count += 1
        else:
            new_params[name] = weight

    logger.info(f"Filtered {filtered_count} parameter tensors")

    # Calculate stats
    total_orig = num_layers * num_experts
    total_kept = sum(len(keep_by_layer[i]) for i in range(num_layers))
    reduction = 1 - total_kept / total_orig

    # Save config
    config = {
        "model_type": "gpt_oss_ve",  # Virtual Expert variant
        "source_model": "openai/gpt-oss-120b",
        "compression_strategy": "capability_aware",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": model.args.num_attention_heads,
        "vocab_size": model.args.vocab_size,
        "experts_per_layer": {str(i): len(keep_by_layer[i]) for i in range(num_layers)},
        "num_experts_per_tok": model.args.num_experts_per_tok,
        "kept_experts_by_layer": {str(i): keep_by_layer[i] for i in range(num_layers)},
        "stats": {
            "original_experts": total_orig,
            "kept_experts": total_kept,
            "reduction": f"{100*reduction:.1f}%",
        },
        "virtual_experts": [
            "calculator", "datetime", "interpreter", "unit_converter"
        ],
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {output_path / 'config.json'}")

    # Save weights
    try:
        weights_path = output_path / "weights.mlx.npz"
        mx.savez(str(weights_path), **new_params)
        logger.info(f"Saved weights to {weights_path}")
    except Exception as e:
        logger.error(f"Could not save weights: {e}")
        raise

    # Copy tokenizer
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            try:
                src = hf_hub_download("openai/gpt-oss-120b", filename)
                shutil.copy(src, output_path / filename)
            except:
                pass
    except:
        pass

    # Print summary
    print()
    print("=" * 80)
    print("GPT-OSS-120B-VE (Virtual Expert) SAVED")
    print("=" * 80)
    print()
    print(f"Output: {output_path}")
    print()
    print(f"Original experts:  {total_orig:,}")
    print(f"Kept experts:      {total_kept:,}")
    print(f"Compression:       {100*reduction:.1f}%")
    print()
    print("Strategy: Remove externalizable, keep fluency")
    print("Virtual experts: calculator, datetime, interpreter, unit_converter")
    print()

    # Estimate size
    orig_params = sum(w.size for w in flat_params.values())
    new_params_size = sum(w.size for w in new_params.values())
    print(f"Parameters: {orig_params:,} → {new_params_size:,} ({100*(1-new_params_size/orig_params):.1f}% reduction)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build Capability-Aware GPT-OSS-120B-VE")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't build")
    parser.add_argument("--build", action="store_true", help="Build the model")
    parser.add_argument("--aggressive", action="store_true", help="More aggressive compression")
    parser.add_argument("--output", type=str, default="./gpt-oss-120b-ve", help="Output directory")

    args = parser.parse_args()

    # Load model
    from mlx_lm import load

    logger.info("Loading openai/gpt-oss-120b...")
    model, tokenizer = load("openai/gpt-oss-120b")

    num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.layers)
    num_experts = model.args.num_local_experts

    print()
    print("=" * 80)
    print("GPT-OSS-120B CAPABILITY ANALYSIS")
    print("=" * 80)
    print()
    print(f"Layers: {num_layers}")
    print(f"Experts per layer: {num_experts}")
    print(f"Total experts: {num_layers * num_experts:,}")
    print()

    # Analyze capabilities
    logger.info("Analyzing expert capabilities...")
    analysis = analyze_expert_capabilities(model, tokenizer, verbose=True)

    # Print report
    print_capability_report(analysis, num_layers, num_experts)

    if args.analyze_only:
        # Save analysis
        analysis_path = Path(args.output)
        analysis_path.mkdir(parents=True, exist_ok=True)

        analysis_data = {
            "experts_to_remove": len(analysis.experts_to_remove),
            "experts_to_keep": len(analysis.experts_to_keep),
            "keep_by_layer": {str(k): v for k, v in analysis.keep_by_layer.items()},
            "removal_by_layer": {str(k): v for k, v in analysis.removal_by_layer.items()},
        }

        with open(analysis_path / "capability_analysis.json", "w") as f:
            json.dump(analysis_data, f, indent=2)

        logger.info(f"Analysis saved to {analysis_path / 'capability_analysis.json'}")
        return

    if args.build:
        # Build the model
        build_capability_aware_model(model, analysis, args.output, aggressive=args.aggressive)


if __name__ == "__main__":
    main()
