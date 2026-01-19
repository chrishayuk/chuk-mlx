#!/usr/bin/env python3
"""
Build GPT-OSS-120B-Lite: A compressed version of GPT-OSS-120B.

Architecture:
- GPT-OSS-120B: 36 layers, 128 experts/layer, k=4 routing
- Total experts: 4,608 (6x more than 20B's 768)

Based on GPT-OSS-20B findings:
- 87% of experts are cold at 1% threshold
- TieredLightweightMoE achieves 92% reduction
- k=4 cooperation is essential

Target Configurations:
1. Conservative (71% reduction):
   - Early layers (0-11):  32 experts (vs 128) = 25%
   - Middle layers (12-23): 48 experts (vs 128) = 37.5%
   - Late layers (24-35):  32 experts (vs 128) = 25%
   - Total: 1,344 experts -> ~34B params

2. Aggressive (90% reduction, matching 20B findings):
   - Early layers:  8 experts (4 teams × 2)
   - Middle layers: 16 experts (8 teams × 2)
   - Late layers:  12 experts (6 teams × 2)
   - Total: 432 experts -> ~12B params

Usage:
    python experiments/moe_dynamics/build_gpt_oss_120b_lite.py --analyze-only
    python experiments/moe_dynamics/build_gpt_oss_120b_lite.py --mode conservative
    python experiments/moe_dynamics/build_gpt_oss_120b_lite.py --mode aggressive
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration for GPT-OSS-120B-Lite variants
# ============================================================================

# 120B architecture constants
NUM_LAYERS = 36
NUM_EXPERTS = 128
TOP_K = 4
TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS  # 4,608

# Conservative: 71% reduction (safe, validated on 20B)
LITE_CONFIG_CONSERVATIVE = {
    'early': {'layers': list(range(0, 12)), 'keep': 32},   # 25% of 128
    'middle': {'layers': list(range(12, 24)), 'keep': 48}, # 37.5% of 128
    'late': {'layers': list(range(24, 36)), 'keep': 32},   # 25% of 128
}

# Aggressive: 90% reduction (extrapolated from 20B's 92%)
LITE_CONFIG_AGGRESSIVE = {
    'early': {'layers': list(range(0, 12)), 'keep': 8},    # 4 teams × 2
    'middle': {'layers': list(range(12, 24)), 'keep': 16}, # 8 teams × 2
    'late': {'layers': list(range(24, 36)), 'keep': 12},   # 6 teams × 2
}

# Analysis prompts (same as 20B for comparison)
ANALYSIS_PROMPTS = [
    # Math
    "127 * 89 = ",
    "456 + 789 = ",
    "sqrt(144) = ",
    "What is 15% of 200?",
    "Calculate 2^10",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    "for item in items:",
    "async def fetch_data():",
    # Language
    "The capital of France is",
    "Shakespeare wrote many",
    "The opposite of hot is",
    "In the beginning there was",
    "To be or not to be",
    # Reasoning
    "If all cats are mammals, then",
    "To solve this equation, first",
    "Therefore, we can conclude that",
    "The logical consequence is",
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
    activation_counts: dict[int, int]
    total_tokens: int
    hot_experts: list[int]
    cold_experts: list[int] = field(default_factory=list)
    hot_coverage: float = 0.0


@dataclass
class ModelAnalysis:
    """Complete analysis results for the model."""
    layers: dict[int, LayerAnalysis]
    total_tokens: int
    cold_expert_rate: float
    hot_expert_rate: float
    compression_estimate: dict[str, float]


def analyze_expert_activation(
    model,
    tokenizer,
    prompts: list[str] | None = None,
    max_prompts: int = 25,
) -> ModelAnalysis:
    """
    Run prompts through the model and track which experts activate.

    Memory-optimized for 120B model.
    """
    prompts = prompts or ANALYSIS_PROMPTS
    prompts = prompts[:max_prompts]

    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    num_layers = len(layers)
    num_experts = model.args.num_local_experts
    top_k = model.args.num_experts_per_tok

    logger.info(f"Analyzing {num_layers} layers, {num_experts} experts, top-{top_k}")
    logger.info(f"Using {len(prompts)} prompts")

    # Initialize counters
    layer_counts = {i: {e: 0 for e in range(num_experts)} for i in range(num_layers)}
    total_tokens = 0

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx % 5 == 0:
            logger.info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        if hasattr(model, 'model'):
            h = model.model.embed_tokens(input_ids)
        else:
            h = model.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = h.shape
        total_tokens += seq_len

        # Create causal mask
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
                x_flat = normed.reshape(-1, hidden_size)
                logits = mlp.router(x_flat)
                top_k_indices = mx.argsort(logits, axis=-1)[:, -top_k:]

                for tok_idx in range(seq_len):
                    for k in range(top_k):
                        expert_idx = int(top_k_indices[tok_idx, k])
                        layer_counts[layer_idx][expert_idx] += 1

            # Forward through layer
            h = layer(h, mask=mask)

        # Memory cleanup after each prompt
        mx.eval(h)
        del h
        gc.collect()

    # Build analysis results
    layer_analyses = {}
    total_cold = 0
    total_hot = 0

    for layer_idx in range(num_layers):
        counts = layer_counts[layer_idx]
        total_layer = sum(counts.values())

        # Sort experts by activation frequency
        sorted_experts = sorted(range(num_experts), key=lambda e: counts[e], reverse=True)

        # Identify cold experts (< 1% activation)
        threshold = total_layer * 0.01
        cold = [e for e in range(num_experts) if counts[e] < threshold]
        hot = [e for e in sorted_experts if counts[e] >= threshold]

        # Calculate hot coverage
        hot_activations = sum(counts[e] for e in hot)
        hot_coverage = hot_activations / total_layer if total_layer > 0 else 0

        layer_analyses[layer_idx] = LayerAnalysis(
            layer_idx=layer_idx,
            activation_counts=counts,
            total_tokens=total_tokens,
            hot_experts=sorted_experts,
            cold_experts=cold,
            hot_coverage=hot_coverage,
        )

        total_cold += len(cold)
        total_hot += num_experts - len(cold)

    total_experts = num_layers * num_experts
    cold_rate = total_cold / total_experts
    hot_rate = total_hot / total_experts

    # Estimate compression potential
    compression = {
        'conservative': calculate_compression(layer_analyses, LITE_CONFIG_CONSERVATIVE),
        'aggressive': calculate_compression(layer_analyses, LITE_CONFIG_AGGRESSIVE),
    }

    return ModelAnalysis(
        layers=layer_analyses,
        total_tokens=total_tokens,
        cold_expert_rate=cold_rate,
        hot_expert_rate=hot_rate,
        compression_estimate=compression,
    )


def calculate_compression(
    analysis: dict[int, LayerAnalysis],
    config: dict[str, Any],
) -> float:
    """Calculate expected compression ratio for a config."""
    total_kept = 0
    for tier, cfg in config.items():
        total_kept += len(cfg['layers']) * cfg['keep']

    return 1.0 - (total_kept / TOTAL_EXPERTS)


def get_tier_for_layer(layer_idx: int, config: dict) -> str:
    """Get tier name for layer."""
    for tier, cfg in config.items():
        if layer_idx in cfg['layers']:
            return tier
    return 'middle'


def print_analysis_report(analysis: ModelAnalysis, config: dict) -> None:
    """Print analysis report showing hot experts per layer."""
    print()
    print("=" * 80)
    print("GPT-OSS-120B EXPERT ACTIVATION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Total prompts analyzed: {analysis.total_tokens} tokens")
    print(f"Cold expert rate (< 1%): {analysis.cold_expert_rate:.1%}")
    print(f"Hot expert rate (>= 1%): {analysis.hot_expert_rate:.1%}")
    print()

    for tier, cfg in config.items():
        keep_n = cfg['keep']
        layers = cfg['layers']
        print(f"{tier.upper()} LAYERS (L{layers[0]}-L{layers[-1]}): keeping {keep_n}/{NUM_EXPERTS} experts")
        print("-" * 60)

        # Sample a few layers from each tier
        sample_layers = [layers[0], layers[len(layers)//2], layers[-1]]
        for layer_idx in sample_layers:
            if layer_idx not in analysis.layers:
                continue

            layer = analysis.layers[layer_idx]
            hot = layer.hot_experts[:keep_n]

            total_activations = sum(layer.activation_counts.values())
            hot_activations = sum(layer.activation_counts[e] for e in hot)
            hot_pct = 100 * hot_activations / total_activations if total_activations > 0 else 0

            print(f"  L{layer_idx:2d}: hot={hot[:4]}... cold={len(layer.cold_experts)}/{NUM_EXPERTS} ({hot_pct:.1f}% coverage)")

        print()

    # Summary
    total_hot = sum(config[get_tier_for_layer(i, config)]['keep'] for i in range(NUM_LAYERS))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Original:  {TOTAL_EXPERTS:,} experts ({NUM_LAYERS} layers × {NUM_EXPERTS} experts)")
    print(f"Lite:      {total_hot:,} experts")
    print(f"Reduction: {100 * (1 - total_hot/TOTAL_EXPERTS):.1f}%")
    print()

    # Compare with 20B findings
    print("Comparison with GPT-OSS-20B findings:")
    print(f"  20B cold rate:  87% at 1% threshold")
    print(f"  120B cold rate: {analysis.cold_expert_rate:.1%}")
    print(f"  Prediction: {'VALIDATED' if analysis.cold_expert_rate > 0.70 else 'DIFFERENT'}")
    print()


def save_lite_model(
    model,
    analysis: ModelAnalysis,
    config: dict,
    output_dir: str,
) -> None:
    """
    Save GPT-OSS-120B-Lite with only hot experts.

    Memory-optimized: streams weights to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving GPT-OSS-120B-Lite to {output_path}")

    # Build hot expert indices per layer
    hot_experts_by_layer = {}
    for layer_idx in range(NUM_LAYERS):
        tier = get_tier_for_layer(layer_idx, config)
        keep_n = config[tier]['keep']
        hot_experts_by_layer[layer_idx] = analysis.layers[layer_idx].hot_experts[:keep_n]

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
        elif hasattr(params_dict, 'shape'):
            results[prefix] = params_dict
        return results

    flat_params = get_leaf_params(params)

    # Filter expert weights
    new_params = {}
    filtered_count = 0
    original_expert_params = 0
    new_expert_params = 0

    logger.info(f"Processing {len(flat_params)} parameter tensors...")

    for name, weight in flat_params.items():
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

        if is_expert_weight and weight.shape[0] == NUM_EXPERTS:
            hot = hot_experts_by_layer[layer_idx]

            original_expert_params += weight.size
            new_weight = weight[hot]
            new_expert_params += new_weight.size

            new_params[name] = new_weight
            filtered_count += 1

        elif is_router_weight and weight.shape[0] == NUM_EXPERTS:
            hot = hot_experts_by_layer[layer_idx]

            original_expert_params += weight.size
            new_weight = weight[hot]
            new_expert_params += new_weight.size

            new_params[name] = new_weight
            filtered_count += 1

        else:
            new_params[name] = weight

    logger.info(f"Filtered {filtered_count} parameter tensors")
    logger.info(f"Expert params: {original_expert_params:,} -> {new_expert_params:,} ({100*(1-new_expert_params/original_expert_params):.1f}% reduction)")

    # Update config
    model_config = {
        "model_type": "gpt_oss_lite",
        "source_model": "openai/gpt-oss-120b",
        "hidden_size": model.args.hidden_size,
        "intermediate_size": model.args.intermediate_size,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": model.args.num_attention_heads,
        "vocab_size": model.args.vocab_size,
        "experts_per_layer": {
            str(i): len(hot_experts_by_layer[i]) for i in range(NUM_LAYERS)
        },
        "num_experts_per_tok": min(TOP_K, min(len(h) for h in hot_experts_by_layer.values())),
        "hot_experts_by_layer": {
            str(i): hot_experts_by_layer[i] for i in range(NUM_LAYERS)
        },
        "lite_config": config,
        "analysis": {
            "cold_expert_rate": analysis.cold_expert_rate,
            "total_tokens_analyzed": analysis.total_tokens,
        },
    }

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Save analysis metadata
    analysis_data = {
        str(layer_idx): {
            "hot_experts": data.hot_experts[:config[get_tier_for_layer(layer_idx, config)]['keep']],
            "cold_experts": data.cold_experts,
            "hot_coverage": data.hot_coverage,
        }
        for layer_idx, data in analysis.layers.items()
    }
    analysis_path = output_path / "expert_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    logger.info(f"Saved analysis to {analysis_path}")

    # Save weights using MLX native format (handles quantized types)
    try:
        from mlx.utils import save_safetensors
        weights_path = output_path / "model.safetensors"
        save_safetensors(str(weights_path), new_params)
        logger.info(f"Saved weights to {weights_path}")
    except Exception as e:
        logger.warning(f"Could not save as safetensors: {e}")
        # Try MLX savez as fallback
        try:
            weights_path = output_path / "weights.mlx"
            mx.savez(str(weights_path), **new_params)
            logger.info(f"Saved weights to {weights_path}")
        except Exception as e2:
            logger.warning(f"Could not save with mx.savez: {e2}")
            # Last resort: save individual arrays
            try:
                weights_dir = output_path / "weights"
                weights_dir.mkdir(exist_ok=True)
                for name, weight in new_params.items():
                    safe_name = name.replace("/", "_").replace(".", "_")
                    mx.save(str(weights_dir / f"{safe_name}.npy"), weight)
                logger.info(f"Saved weights to {weights_dir}/")
            except Exception as e3:
                logger.error(f"Could not save weights: {e3}")
                # Save just the metadata so we can reconstruct later
                metadata_path = output_path / "weight_shapes.json"
                shapes = {k: list(v.shape) for k, v in new_params.items()}
                with open(metadata_path, "w") as f:
                    json.dump(shapes, f, indent=2)
                logger.info(f"Saved weight shapes to {metadata_path}")

    # Copy tokenizer
    try:
        from huggingface_hub import hf_hub_download

        for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            try:
                src = hf_hub_download("openai/gpt-oss-120b", filename)
                dst = output_path / filename
                shutil.copy(src, dst)
                logger.info(f"Copied {filename}")
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not copy tokenizer files: {e}")

    # Print summary
    print()
    print("=" * 80)
    print("GPT-OSS-120B-LITE SAVED")
    print("=" * 80)
    print()
    print(f"Output directory: {output_path}")
    print()

    total_orig_experts = TOTAL_EXPERTS
    total_lite_experts = sum(len(hot_experts_by_layer[i]) for i in range(NUM_LAYERS))
    expert_reduction = 1 - total_lite_experts / total_orig_experts

    total_orig_params = sum(w.size for w in flat_params.values())
    total_new_params = sum(w.size for w in new_params.values())
    param_reduction = 1 - total_new_params / total_orig_params

    print(f"Expert reduction: {total_orig_experts:,} -> {total_lite_experts:,} ({expert_reduction*100:.1f}%)")
    print(f"Parameter reduction: {total_orig_params:,} -> {total_new_params:,} ({param_reduction*100:.1f}%)")
    print()

    # Estimate memory
    orig_bytes = total_orig_params * 2  # bfloat16
    new_bytes = total_new_params * 2
    print(f"Estimated memory: {orig_bytes/1e9:.1f}GB -> {new_bytes/1e9:.1f}GB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build GPT-OSS-120B-Lite")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis, don't save model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Compression mode (default: conservative)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for lite model",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=25,
        help="Maximum prompts to analyze (memory limited)",
    )

    args = parser.parse_args()

    # Select config
    if args.mode == "aggressive":
        config = LITE_CONFIG_AGGRESSIVE
        default_output = "./gpt-oss-120b-lite-aggressive"
    else:
        config = LITE_CONFIG_CONSERVATIVE
        default_output = "./gpt-oss-120b-lite"

    output_dir = args.output or default_output

    # Load model
    from mlx_lm import load

    model_id = "openai/gpt-oss-120b"
    logger.info(f"Loading {model_id}...")
    logger.info("This may take several minutes for the 120B model...")

    model, tokenizer = load(model_id)

    # Print model info
    print()
    print("=" * 80)
    print("GPT-OSS-120B MODEL INFO")
    print("=" * 80)
    print()
    print(f"Hidden size: {model.args.hidden_size}")
    print(f"Experts: {model.args.num_local_experts}")
    print(f"Top-k: {model.args.num_experts_per_tok}")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Total experts: {TOTAL_EXPERTS:,}")
    print()

    # Analyze expert activation
    logger.info("Analyzing expert activation patterns...")
    analysis = analyze_expert_activation(model, tokenizer, max_prompts=args.max_prompts)

    # Print report
    print_analysis_report(analysis, config)

    if args.analyze_only:
        logger.info("Analysis complete (--analyze-only mode)")

        # Save analysis to file
        analysis_output = Path(output_dir)
        analysis_output.mkdir(parents=True, exist_ok=True)

        analysis_data = {
            "model": model_id,
            "cold_expert_rate": analysis.cold_expert_rate,
            "hot_expert_rate": analysis.hot_expert_rate,
            "compression_estimates": analysis.compression_estimate,
            "layers": {
                str(idx): {
                    "cold_count": len(layer.cold_experts),
                    "hot_coverage": layer.hot_coverage,
                    "top_5_hot": layer.hot_experts[:5],
                }
                for idx, layer in analysis.layers.items()
            }
        }

        with open(analysis_output / "analysis_120b.json", "w") as f:
            json.dump(analysis_data, f, indent=2)

        logger.info(f"Analysis saved to {analysis_output / 'analysis_120b.json'}")
        return

    # Save lite model
    save_lite_model(model, analysis, config, output_dir)


if __name__ == "__main__":
    main()
