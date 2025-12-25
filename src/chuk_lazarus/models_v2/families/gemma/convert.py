"""
Weight conversion utilities for Gemma 3 models.

Converts HuggingFace/MLX-community checkpoint weights to our format.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Mapping from HuggingFace weight names to our weight names
# Gemma 3 has similar structure to Llama but with additional norm layers
GEMMA_WEIGHT_MAP = {
    # Embeddings
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    # Final norm
    "model.norm.weight": "model.norm.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
}

# Layer-level weight patterns for Gemma 3
GEMMA_LAYER_PATTERNS = {
    # Attention
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    # Query/key norms (Gemma-specific)
    "self_attn.q_norm.weight": "self_attn.q_norm.weight",
    "self_attn.k_norm.weight": "self_attn.k_norm.weight",
    # MLP
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    # 4 normalization layers (Gemma-specific)
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
}


def convert_hf_weights(
    hf_weights: dict[str, np.ndarray],
    tie_word_embeddings: bool = False,
) -> dict[str, np.ndarray]:
    """
    Convert HuggingFace Gemma weights to our format.

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
            # Skip unmapped weights (e.g., rotary embeddings if precomputed)
            continue

        # Handle tied embeddings
        if tie_word_embeddings and hf_name == "lm_head.weight":
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
    if hf_name in GEMMA_WEIGHT_MAP:
        return GEMMA_WEIGHT_MAP[hf_name]

    # Pattern for layer weights
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(hf_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Check if it's a known layer pattern
        if rest in GEMMA_LAYER_PATTERNS:
            return f"model.layers.{layer_idx}.{GEMMA_LAYER_PATTERNS[rest]}"

        # Otherwise pass through (might be new/unknown)
        return f"model.layers.{layer_idx}.{rest}"

    # Unrecognized weight
    return None


def convert_mlx_community_weights(
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Convert weights from mlx-community format.

    The mlx-community models typically use the same naming convention
    as our implementation, so this is mostly a pass-through.

    Args:
        weights: Dictionary of mlx-community weights

    Returns:
        Dictionary of converted weights
    """
    # mlx-community weights are usually already in the right format
    return convert_hf_weights(weights)


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
    for hf_name, mapped_name in GEMMA_WEIGHT_MAP.items():
        if mapped_name == our_name:
            return hf_name

    # Pattern matching for layers
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(our_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Reverse lookup in layer patterns
        for hf_suffix, our_suffix in GEMMA_LAYER_PATTERNS.items():
            if our_suffix == rest:
                return f"model.layers.{layer_idx}.{hf_suffix}"

        # Pass through
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
