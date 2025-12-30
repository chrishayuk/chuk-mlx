"""
Weight conversion utilities for StarCoder2 models.

Converts HuggingFace checkpoint weights to our format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# Mapping from HuggingFace weight names to our weight names
STARCODER2_WEIGHT_MAP = {
    # Embeddings
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    # Final norm (LayerNorm has weight and bias)
    "model.norm.weight": "model.norm.weight",
    "model.norm.bias": "model.norm.bias",
    # LM head
    "lm_head.weight": "lm_head.lm_head.weight",
}


def convert_hf_weights(
    hf_weights: dict[str, np.ndarray],
    tie_word_embeddings: bool = False,
) -> dict[str, np.ndarray]:
    """
    Convert HuggingFace StarCoder2 weights to our format.

    Args:
        hf_weights: Dictionary of HuggingFace weights
        tie_word_embeddings: Whether to tie lm_head to embeddings

    Returns:
        Dictionary of converted weights
    """
    converted = {}

    for hf_name, weight in hf_weights.items():
        # Direct mapping for most weights
        our_name = _map_weight_name(hf_name)

        if our_name is None:
            # Skip unmapped weights
            continue

        # Handle tied embeddings
        if tie_word_embeddings and hf_name == "lm_head.weight":
            # Skip - will use embedding weights
            continue

        converted[our_name] = weight

    return converted


def _map_weight_name(hf_name: str) -> str | None:
    """
    Map HuggingFace weight name to our weight name.

    Args:
        hf_name: HuggingFace weight name

    Returns:
        Our weight name, or None if not mapped
    """
    # Check direct mapping first
    if hf_name in STARCODER2_WEIGHT_MAP:
        return STARCODER2_WEIGHT_MAP[hf_name]

    # Pattern for layer weights
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(hf_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # StarCoder2 uses:
        # - input_layernorm.{weight,bias}
        # - self_attn.{q,k,v,o}_proj.{weight,bias}
        # - post_attention_layernorm.{weight,bias}
        # - mlp.{c_fc,c_proj}.{weight,bias} or mlp.{up_proj,down_proj}.{weight,bias}

        # Handle MLP naming (c_fc -> up_proj, c_proj -> down_proj)
        if "mlp.c_fc" in rest:
            rest = rest.replace("mlp.c_fc", "mlp.up_proj")
        elif "mlp.c_proj" in rest:
            rest = rest.replace("mlp.c_proj", "mlp.down_proj")

        # Map the rest of the path
        return f"model.layers.{layer_idx}.{rest}"

    # Unrecognized weight
    return None


def convert_mlx_to_hf(
    mlx_weights: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Convert our weights back to HuggingFace format.

    Args:
        mlx_weights: Dictionary of our weights

    Returns:
        Dictionary of HuggingFace weights
    """
    converted = {}

    for our_name, weight in mlx_weights.items():
        # Convert to numpy
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
    for hf_name, mapped_name in STARCODER2_WEIGHT_MAP.items():
        if mapped_name == our_name:
            return hf_name

    # Pattern matching for layers
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(our_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Reverse MLP naming
        if "mlp.up_proj" in rest:
            rest = rest.replace("mlp.up_proj", "mlp.c_fc")
        elif "mlp.down_proj" in rest:
            rest = rest.replace("mlp.down_proj", "mlp.c_proj")

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
        >>> weights = load_weights("/path/to/starcoder2-3b")
        >>> model.update(tree_unflatten(list(weights.items())))
    """
    model_path = Path(model_path)
    raw_weights: dict[str, mx.array] = {}

    # Load all safetensor files
    for sf_path in sorted(model_path.glob("*.safetensors")):
        file_weights = mx.load(str(sf_path))
        raw_weights.update(file_weights)

    # Convert weight names
    converted: dict[str, mx.array] = {}
    for name, weight in raw_weights.items():
        new_name = name

        # MLP naming: c_fc -> up_proj, c_proj -> down_proj
        new_name = new_name.replace("mlp.c_fc", "mlp.up_proj")
        new_name = new_name.replace("mlp.c_proj", "mlp.down_proj")

        # Embedding: add nested .weight for MLX nn.Embedding
        if new_name == "model.embed_tokens.weight":
            new_name = "model.embed_tokens.weight.weight"

        # LM head: add nested path
        if new_name == "lm_head.weight":
            new_name = "lm_head.lm_head.weight"

        converted[new_name] = weight

    return converted
