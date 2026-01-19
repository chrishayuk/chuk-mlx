#!/usr/bin/env python3
"""
Weight Transfer: Copy expert weights from GPT-OSS to TieredLightweightMoE.

Uses the clustering metadata from convert_to_tiered_lightweight.py to:
1. Load the source model's expert weights
2. Map them to team members based on discovered clusters
3. Initialize mixing weights from routing frequencies
4. Save the converted model

Usage:
    python experiments/moe_dynamics/weight_transfer.py
    python experiments/moe_dynamics/weight_transfer.py --metadata converted_tiered_lightweight/conversion_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.ffn.moe_experimental import (
    ExperimentalMoEConfig,
    TieredLightweightMoE,
    LightweightTeam,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_source_model_experts(model_id: str) -> tuple[dict, dict]:
    """
    Load expert weights from source MoE model.

    Returns:
        expert_weights: dict[layer_idx][expert_idx] -> weight dict
        model_config: model configuration
    """
    from mlx_lm import load

    logger.info(f"Loading source model: {model_id}")
    model, tokenizer = load(model_id)

    # Extract expert weights by layer
    expert_weights = {}
    model_config = {}

    # Get model parameters
    params = model.parameters()

    # Find MoE layers and extract expert weights
    # Structure varies by model, but typically:
    # model.layers[i].block_sparse_moe.experts[j].{w1, w2, w3}

    for layer_idx in range(len(model.layers)):
        layer = model.layers[layer_idx]

        # Check if this layer has MoE
        if hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
            expert_weights[layer_idx] = {}

            for expert_idx, expert in enumerate(moe.experts):
                expert_weights[layer_idx][expert_idx] = {
                    'w1': expert.w1.weight if hasattr(expert.w1, 'weight') else expert.w1,
                    'w2': expert.w2.weight if hasattr(expert.w2, 'weight') else expert.w2,
                    'w3': expert.w3.weight if hasattr(expert.w3, 'weight') else expert.w3,
                }

            # Get config from first expert
            if expert_weights[layer_idx]:
                first_expert = expert_weights[layer_idx][0]
                if 'hidden_size' not in model_config:
                    model_config['hidden_size'] = first_expert['w1'].shape[1]
                    model_config['intermediate_size'] = first_expert['w1'].shape[0]
                    model_config['num_experts'] = len(moe.experts)

    logger.info(f"Loaded {len(expert_weights)} MoE layers")
    logger.info(f"Config: hidden={model_config.get('hidden_size')}, "
                f"intermediate={model_config.get('intermediate_size')}, "
                f"experts={model_config.get('num_experts')}")

    return expert_weights, model_config, model, tokenizer


def create_tiered_lightweight_model(
    model_config: dict,
    metadata: dict,
) -> dict[int, TieredLightweightMoE]:
    """
    Create TieredLightweightMoE layers based on metadata.

    Returns dict of layer_idx -> TieredLightweightMoE module.
    """
    config = ExperimentalMoEConfig(
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        num_experts=model_config['num_experts'],
        num_experts_per_tok=4,
        team_size=metadata['config']['team_size'],
        variant='tiered_lightweight',
    )

    layers = {}
    for layer_str in metadata['layers'].keys():
        layer_idx = int(layer_str)
        layers[layer_idx] = TieredLightweightMoE(config, layer_idx)

    return layers, config


def transfer_weights(
    source_experts: dict,
    tiered_layers: dict[int, TieredLightweightMoE],
    metadata: dict,
) -> None:
    """
    Transfer weights from source experts to tiered lightweight teams.

    For each team:
    1. Get the expert indices assigned to that team
    2. Copy expert weights to team members
    3. Initialize mixing weights from routing frequencies
    """
    for layer_str, layer_data in metadata['layers'].items():
        layer_idx = int(layer_str)

        if layer_idx not in source_experts:
            logger.warning(f"Layer {layer_idx} not found in source model")
            continue

        tiered_layer = tiered_layers[layer_idx]
        teams = layer_data['teams']
        mixing_weights = layer_data['mixing_weights']

        logger.info(f"Layer {layer_idx}: transferring {len(teams)} teams")

        for team_idx, (team_experts, team_mix) in enumerate(zip(teams, mixing_weights)):
            if team_idx >= len(tiered_layer.teams):
                logger.warning(f"Layer {layer_idx}: team {team_idx} exceeds available teams")
                continue

            team = tiered_layer.teams[team_idx]

            # Transfer expert weights to team members
            for member_idx, expert_idx in enumerate(team_experts):
                if member_idx >= len(team.members):
                    break

                if expert_idx not in source_experts[layer_idx]:
                    logger.warning(f"Expert {expert_idx} not found at layer {layer_idx}")
                    continue

                src = source_experts[layer_idx][expert_idx]
                member = team.members[member_idx]

                # Copy weights (GLU structure: gate_proj, up_proj, down_proj)
                # Source: w1 (gate), w3 (up), w2 (down)
                if hasattr(member, 'gate_proj'):
                    member.gate_proj.weight = src['w1']
                if hasattr(member, 'up_proj'):
                    member.up_proj.weight = src['w3']
                if hasattr(member, 'down_proj'):
                    member.down_proj.weight = src['w2']

            # Initialize mixing weights from routing frequencies
            team._mix_weights = mx.array(team_mix[:len(team.members)])

        logger.info(f"Layer {layer_idx}: transferred {len(teams)} teams")


def save_converted_model(
    tiered_layers: dict[int, TieredLightweightMoE],
    config: ExperimentalMoEConfig,
    output_dir: Path,
) -> None:
    """Save the converted model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all weights
    all_weights = {}
    for layer_idx, layer in tiered_layers.items():
        layer_weights = layer.parameters()
        for name, param in _flatten_params(layer_weights, f"layers.{layer_idx}"):
            all_weights[name] = param

    # Save weights
    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), all_weights)

    # Save config
    config_dict = {
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_experts': config.num_experts,
        'num_experts_per_tok': config.num_experts_per_tok,
        'team_size': config.team_size,
        'variant': 'tiered_lightweight',
    }
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved converted model to {output_dir}")


def _flatten_params(params, prefix=""):
    """Flatten nested parameter dict."""
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            yield from _flatten_params(v, new_prefix)
    elif isinstance(params, list):
        for i, v in enumerate(params):
            yield from _flatten_params(v, f"{prefix}.{i}")
    elif isinstance(params, mx.array):
        yield prefix, params


def main():
    parser = argparse.ArgumentParser(description="Transfer weights to TieredLightweight")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Source model ID",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="converted_tiered_lightweight/conversion_metadata.json",
        help="Path to conversion metadata",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted_tiered_lightweight/model",
        help="Output directory for converted model",
    )

    args = parser.parse_args()

    # Load metadata
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.error("Run convert_to_tiered_lightweight.py first")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load source model
    source_experts, model_config, source_model, tokenizer = load_source_model_experts(args.model)

    # Create tiered lightweight model
    tiered_layers, config = create_tiered_lightweight_model(model_config, metadata)

    # Transfer weights
    transfer_weights(source_experts, tiered_layers, metadata)

    # Save converted model
    output_dir = Path(args.output)
    save_converted_model(tiered_layers, config, output_dir)

    print()
    print("=" * 70)
    print("WEIGHT TRANSFER COMPLETE")
    print("=" * 70)
    print()
    print(f"Source model: {args.model}")
    print(f"Converted model: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run evaluation: python experiments/moe_dynamics/evaluate_quality.py")
    print("  2. If gap is large, run distillation: python experiments/moe_dynamics/distill.py")
    print()


if __name__ == "__main__":
    main()
