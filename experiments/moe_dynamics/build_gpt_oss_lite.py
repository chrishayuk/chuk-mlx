#!/usr/bin/env python3
"""
Build GPT-OSS-Lite: A recut of GPT-OSS with only hot experts.

This script:
1. Loads GPT-OSS-20B and runs prompts to identify hot experts
2. Creates a new model with only the hot experts per layer
3. Saves the smaller model that fits on 8GB machines

Target Configuration:
- Early layers (0-7):   6 experts (vs 32)  -> 81% reduction
- Middle layers (8-17): 12 experts (vs 32) -> 62% reduction
- Late layers (18-23):  8 experts (vs 32)  -> 75% reduction
- Total: 216 experts (vs 768) -> 72% expert reduction

Usage:
    python experiments/moe_dynamics/build_gpt_oss_lite.py
    python experiments/moe_dynamics/build_gpt_oss_lite.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration for GPT-OSS-Lite
# Less aggressive pruning for better quality on small machines
# Target: ~5.3GB memory (vs ~9.6GB original)
LITE_CONFIG = {
    'early': {'layers': list(range(0, 8)), 'keep': 16},   # 50% coverage
    'middle': {'layers': list(range(8, 18)), 'keep': 20}, # 62.5% coverage
    'late': {'layers': list(range(18, 24)), 'keep': 16},  # 50% coverage
}

# Prompts for analyzing expert activation
ANALYSIS_PROMPTS = [
    # Math
    "127 * 89 = ",
    "456 + 789 = ",
    "sqrt(144) = ",
    "What is 15% of 200?",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    "for item in items:",
    # Language
    "The capital of France is",
    "Shakespeare wrote many",
    "The opposite of hot is",
    "In the beginning there was",
    # Reasoning
    "If all cats are mammals, then",
    "To solve this equation, first",
    "Therefore, we can conclude that",
    # General
    "Once upon a time",
    "The quick brown fox",
    "Hello, my name is",
    "To summarize the main points",
    "In conclusion,",
]


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer."""
    layer_idx: int
    activation_counts: dict[int, int]  # expert_idx -> count
    total_tokens: int
    hot_experts: list[int]  # sorted by frequency


def analyze_expert_activation(model, tokenizer) -> dict[int, LayerAnalysis]:
    """
    Run prompts through the model and track which experts activate.

    Returns analysis per layer.
    """
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = model.args.num_local_experts
    top_k = model.args.num_experts_per_tok

    logger.info(f"Analyzing {num_layers} layers, {num_experts} experts, top-{top_k}")

    # Initialize counters
    layer_counts = {i: {e: 0 for e in range(num_experts)} for i in range(num_layers)}
    total_tokens = 0

    for prompt in ANALYSIS_PROMPTS:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # We need to trace through the model to capture routing decisions
        # For each layer, capture the top-k indices
        if hasattr(model, 'model'):
            h = model.model.embed_tokens(input_ids)
        else:
            h = model.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = h.shape
        total_tokens += seq_len

        # Create causal mask (match model dtype)
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        for layer_idx, layer in enumerate(layers):
            # Apply input layernorm
            if hasattr(layer, 'input_layernorm'):
                normed = layer.input_layernorm(h)
            else:
                normed = h

            # Get MoE routing
            mlp = layer.mlp
            if hasattr(mlp, 'router'):
                # Flatten for router
                x_flat = normed.reshape(-1, hidden_size)

                # Get router logits
                logits = mlp.router(x_flat)  # (seq, num_experts)

                # Get top-k indices
                top_k_indices = mx.argsort(logits, axis=-1)[:, -top_k:]

                # Count activations
                for tok_idx in range(seq_len):
                    for k in range(top_k):
                        expert_idx = int(top_k_indices[tok_idx, k])
                        layer_counts[layer_idx][expert_idx] += 1

            # Forward through layer with mask
            h = layer(h, mask=mask)

    # Build analysis results
    results = {}
    for layer_idx in range(num_layers):
        counts = layer_counts[layer_idx]
        sorted_experts = sorted(range(num_experts), key=lambda e: counts[e], reverse=True)

        results[layer_idx] = LayerAnalysis(
            layer_idx=layer_idx,
            activation_counts=counts,
            total_tokens=total_tokens,
            hot_experts=sorted_experts,
        )

    return results


def get_tier_for_layer(layer_idx: int) -> str:
    """Get tier name for layer."""
    for tier, config in LITE_CONFIG.items():
        if layer_idx in config['layers']:
            return tier
    return 'middle'


def print_analysis_report(analysis: dict[int, LayerAnalysis]) -> None:
    """Print analysis report showing hot experts per layer."""
    print()
    print("=" * 80)
    print("EXPERT ACTIVATION ANALYSIS")
    print("=" * 80)
    print()

    for tier, config in LITE_CONFIG.items():
        keep_n = config['keep']
        print(f"{tier.upper()} LAYERS (L{config['layers'][0]}-L{config['layers'][-1]}): keeping {keep_n} experts")
        print("-" * 60)

        for layer_idx in config['layers']:
            if layer_idx not in analysis:
                continue

            layer = analysis[layer_idx]
            hot = layer.hot_experts[:keep_n]
            cold = layer.hot_experts[keep_n:]

            # Calculate activation percentages
            total_activations = sum(layer.activation_counts.values())
            hot_activations = sum(layer.activation_counts[e] for e in hot)
            hot_pct = 100 * hot_activations / total_activations if total_activations > 0 else 0

            print(f"  L{layer_idx:2d}: hot={hot[:6]}... ({hot_pct:.1f}% of activations)")

        print()

    # Summary
    total_hot = sum(LITE_CONFIG[get_tier_for_layer(i)]['keep'] for i in range(24))
    total_orig = 24 * 32

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Original: {total_orig} experts")
    print(f"Lite:     {total_hot} experts")
    print(f"Reduction: {100 * (1 - total_hot/total_orig):.1f}%")
    print()


def save_lite_model(
    model,
    tokenizer,
    analysis: dict[int, LayerAnalysis],
    output_dir: str,
) -> None:
    """
    Save GPT-OSS-Lite with only hot experts.

    This creates a new model directory with:
    - Reduced expert weights
    - Modified config
    - Same tokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving GPT-OSS-Lite to {output_path}")

    # Build hot expert indices per layer
    hot_experts_by_layer = {}
    for layer_idx in range(24):
        tier = get_tier_for_layer(layer_idx)
        keep_n = LITE_CONFIG[tier]['keep']
        hot_experts_by_layer[layer_idx] = analysis[layer_idx].hot_experts[:keep_n]

    # Get flat parameter dict
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
        elif hasattr(params_dict, 'shape'):  # mx.array check
            results[prefix] = params_dict
        return results

    flat_params = get_leaf_params(params)

    # Filter expert weights
    new_params = {}
    filtered_count = 0
    original_expert_params = 0
    new_expert_params = 0

    for name, weight in flat_params.items():
        # Check if this is a layer expert weight
        # Pattern: model.layers.{N}.mlp.experts.{gate|up|down}_proj.{weight|scales|bias}
        # Or: model.layers.{N}.mlp.router.{weight|bias}

        parts = name.split('.')
        layer_idx = None
        is_expert_weight = False
        is_router_weight = False

        # Extract layer index
        if 'layers' in parts:
            try:
                layers_idx = parts.index('layers')
                layer_idx = int(parts[layers_idx + 1])
            except (ValueError, IndexError):
                pass

        # Check if it's expert or router weight
        if layer_idx is not None and 'mlp' in parts:
            if 'experts' in parts:
                is_expert_weight = True
            elif 'router' in parts:
                is_router_weight = True

        if is_expert_weight and weight.shape[0] == 32:
            # Filter to hot experts
            hot = hot_experts_by_layer[layer_idx]

            original_expert_params += weight.size
            # Index with list directly
            new_weight = weight[hot]
            new_expert_params += new_weight.size

            new_params[name] = new_weight
            filtered_count += 1
            logger.debug(f"Filtered {name}: {weight.shape} -> {new_weight.shape}")

        elif is_router_weight and weight.shape[0] == 32:
            # Filter router to only include hot experts
            hot = hot_experts_by_layer[layer_idx]

            original_expert_params += weight.size
            # Index with list directly
            new_weight = weight[hot]
            new_expert_params += new_weight.size

            new_params[name] = new_weight
            filtered_count += 1
            logger.debug(f"Filtered {name}: {weight.shape} -> {new_weight.shape}")

        else:
            # Keep unchanged
            new_params[name] = weight

    logger.info(f"Filtered {filtered_count} parameter tensors")
    logger.info(f"Expert params: {original_expert_params:,} -> {new_expert_params:,} ({100*(1-new_expert_params/original_expert_params):.1f}% reduction)")

    # Update config
    config = {
        "model_type": "gpt_oss_lite",
        "hidden_size": model.args.hidden_size,
        "intermediate_size": model.args.intermediate_size,
        "num_hidden_layers": 24,
        "num_attention_heads": model.args.num_attention_heads,
        "vocab_size": model.args.vocab_size,
        # Variable experts per layer
        "experts_per_layer": {
            str(i): len(hot_experts_by_layer[i]) for i in range(24)
        },
        "num_experts_per_tok": min(4, min(len(h) for h in hot_experts_by_layer.values())),
        "hot_experts_by_layer": {
            str(i): hot_experts_by_layer[i] for i in range(24)
        },
        "lite_config": LITE_CONFIG,
        "source_model": "openai/gpt-oss-20b",
    }

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Save analysis metadata
    analysis_data = {
        str(layer_idx): {
            "hot_experts": data.hot_experts[:LITE_CONFIG[get_tier_for_layer(layer_idx)]['keep']],
            "activation_counts": data.activation_counts,
            "total_tokens": data.total_tokens,
        }
        for layer_idx, data in analysis.items()
    }
    analysis_path = output_path / "expert_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    logger.info(f"Saved analysis to {analysis_path}")

    # Save weights using mlx save
    try:
        from mlx.utils import save_safetensors

        weights_path = output_path / "model.safetensors"
        save_safetensors(str(weights_path), new_params)
        logger.info(f"Saved weights to {weights_path}")

    except Exception as e:
        logger.warning(f"Could not save as safetensors: {e}")
        # Try mlx native save
        try:
            import mlx.core as mx_core
            weights_path = output_path / "weights.npz"
            mx_core.savez(str(weights_path), **new_params)
            logger.info(f"Saved weights to {weights_path}")
        except Exception as e2:
            logger.error(f"Could not save weights: {e2}")

    # Copy tokenizer if available
    try:
        from huggingface_hub import hf_hub_download

        for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            try:
                src = hf_hub_download("openai/gpt-oss-20b", filename)
                dst = output_path / filename
                shutil.copy(src, dst)
                logger.info(f"Copied {filename}")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not copy tokenizer files: {e}")

    print()
    print("=" * 80)
    print("GPT-OSS-LITE SAVED")
    print("=" * 80)
    print()
    print(f"Output directory: {output_path}")
    print()

    # Calculate size reduction
    total_orig_experts = 24 * 32
    total_lite_experts = sum(len(hot_experts_by_layer[i]) for i in range(24))
    expert_reduction = 1 - total_lite_experts / total_orig_experts

    total_orig_params = sum(w.size for w in flat_params.values())
    total_new_params = sum(w.size for w in new_params.values())
    param_reduction = 1 - total_new_params / total_orig_params

    print(f"Expert reduction: {total_orig_experts} -> {total_lite_experts} ({expert_reduction*100:.1f}%)")
    print(f"Parameter reduction: {total_orig_params:,} -> {total_new_params:,} ({param_reduction*100:.1f}%)")
    print()

    # Estimate memory
    orig_bytes = total_orig_params * 2  # bfloat16
    new_bytes = total_new_params * 2
    print(f"Estimated memory: {orig_bytes/1e9:.2f}GB -> {new_bytes/1e9:.2f}GB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build GPT-OSS-Lite")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis, don't save model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./gpt-oss-lite",
        help="Output directory for lite model",
    )

    args = parser.parse_args()

    # Load model
    from mlx_lm import load

    model_id = "openai/gpt-oss-20b"
    logger.info(f"Loading {model_id}...")
    model, tokenizer = load(model_id)

    # Print model info
    print()
    print("=" * 80)
    print("MODEL INFO")
    print("=" * 80)
    print()
    print(f"Hidden size: {model.args.hidden_size}")
    print(f"Experts: {model.args.num_local_experts}")
    print(f"Top-k: {model.args.num_experts_per_tok}")
    print(f"Layers: 24")
    print()

    # Analyze expert activation
    logger.info("Analyzing expert activation patterns...")
    analysis = analyze_expert_activation(model, tokenizer)

    # Print report
    print_analysis_report(analysis)

    if args.analyze_only:
        logger.info("Analysis complete (--analyze-only mode)")
        return

    # Save lite model
    save_lite_model(model, tokenizer, analysis, args.output)


if __name__ == "__main__":
    main()
