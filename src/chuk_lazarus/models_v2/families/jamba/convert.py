"""
Weight conversion utilities for Jamba models.

Converts HuggingFace checkpoint weights to our format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def load_hf_config(model_path: str | Path) -> dict[str, Any]:
    """
    Load config.json from HuggingFace model directory.

    Args:
        model_path: Path to model directory

    Returns:
        Config dict from config.json
    """
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_weights(model_path: str | Path) -> dict[str, mx.array]:
    """
    Load and convert weights from HuggingFace safetensors.

    Args:
        model_path: Path to model directory with *.safetensors files

    Returns:
        Dict of converted MLX weights ready to apply to model

    Example:
        >>> weights = load_weights("/path/to/jamba-v0.1")
        >>> model.update(tree_unflatten(list(weights.items())))
    """
    model_path = Path(model_path)
    raw_weights: dict[str, mx.array] = {}

    # Load all safetensor files
    for sf_path in sorted(model_path.glob("*.safetensors")):
        file_weights = mx.load(str(sf_path))
        raw_weights.update(file_weights)

    # Convert HF weight names to our format
    converted: dict[str, mx.array] = {}
    for name, weight in raw_weights.items():
        new_name = _map_weight_name(name)
        if new_name is not None:
            converted[new_name] = weight

    return converted


# Mapping from HuggingFace weight names to our weight names
JAMBA_WEIGHT_MAP = {
    # Embeddings (our TokenEmbedding wraps nn.Embedding, so weight.weight)
    "model.embed_tokens.weight": "model.embed_tokens.weight.weight",
    # Final norm
    "model.final_layernorm.weight": "model.norm.weight",
    # LM head
    "lm_head.weight": "lm_head.lm_head.weight",
}


def convert_hf_weights(
    hf_weights: dict[str, np.ndarray],
    tie_word_embeddings: bool = False,
) -> dict[str, np.ndarray]:
    """
    Convert HuggingFace Jamba weights to our format.

    Args:
        hf_weights: Dictionary of HuggingFace weights
        tie_word_embeddings: Whether to tie lm_head to embeddings

    Returns:
        Dictionary of converted weights
    """
    converted = {}

    for hf_name, weight in hf_weights.items():
        our_name = _map_weight_name(hf_name)

        if our_name is None:
            continue

        if tie_word_embeddings and hf_name == "lm_head.weight":
            continue

        converted[our_name] = weight

    return converted


def _map_weight_name(hf_name: str) -> str | None:
    """
    Map HuggingFace weight name to our weight name.

    Jamba layer structure (HF):
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.mamba.* (for Mamba layers)
    - model.layers.{i}.self_attn.* (for attention layers)
    - model.layers.{i}.pre_ff_layernorm.weight
    - model.layers.{i}.feed_forward.* (dense or MoE)

    Note: HF Jamba has extra layernorms (dt_layernorm, b_layernorm, c_layernorm)
    that our model doesn't use. These are skipped.
    """
    # Check direct mapping first
    if hf_name in JAMBA_WEIGHT_MAP:
        return JAMBA_WEIGHT_MAP[hf_name]

    # Pattern for layer weights
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(hf_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Handle Mamba-specific naming
        # HF: mamba.in_proj.weight -> mamba.ssm.in_proj.weight
        if rest.startswith("mamba."):
            mamba_rest = rest[6:]  # Remove "mamba."

            # Map Mamba2-style layernorms (used by Jamba Reasoning 3B)
            # HF: mamba.dt_layernorm.weight -> mamba.ssm.dt_layernorm.weight
            # These are optional - only present in Mamba2-based models
            if mamba_rest.startswith(("dt_layernorm", "b_layernorm", "c_layernorm")):
                return f"model.layers.{layer_idx}.mamba.ssm.{mamba_rest}"

            return f"model.layers.{layer_idx}.mamba.ssm.{mamba_rest}"

        # Handle MoE router
        if "feed_forward.router" in rest:
            rest = rest.replace("feed_forward.router", "feed_forward.router.gate")

        # Handle MoE experts
        # HF: feed_forward.experts.{i}.* -> feed_forward.experts.{i}.*
        # This should work as-is

        return f"model.layers.{layer_idx}.{rest}"

    return None


def convert_mlx_to_hf(
    mlx_weights: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Convert our weights back to HuggingFace format.
    """
    converted = {}

    for our_name, weight in mlx_weights.items():
        if hasattr(weight, "numpy"):
            weight = weight.numpy()
        elif hasattr(weight, "__array__"):
            weight = np.asarray(weight)

        hf_name = _reverse_map_weight_name(our_name)
        if hf_name:
            converted[hf_name] = weight

    return converted


def _reverse_map_weight_name(our_name: str) -> str | None:
    """Map our weight name back to HuggingFace format."""
    # Reverse lookup in direct map
    for hf_name, mapped_name in JAMBA_WEIGHT_MAP.items():
        if mapped_name == our_name:
            return hf_name

    # Pattern matching for layers
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(our_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Reverse Mamba naming
        if rest.startswith("mamba.ssm."):
            mamba_rest = rest[10:]  # Remove "mamba.ssm."
            return f"model.layers.{layer_idx}.mamba.{mamba_rest}"

        # Reverse MoE router
        if "feed_forward.router.gate" in rest:
            rest = rest.replace("feed_forward.router.gate", "feed_forward.router")

        return f"model.layers.{layer_idx}.{rest}"

    return None


def get_num_params(weights: dict[str, np.ndarray]) -> int:
    """Count total parameters in weights dict."""
    total = 0
    for weight in weights.values():
        total += weight.size
    return total


def print_weight_shapes(weights: dict[str, np.ndarray]) -> None:
    """Print shapes of all weights for debugging."""
    for name, weight in sorted(weights.items()):
        print(f"{name}: {weight.shape}")
