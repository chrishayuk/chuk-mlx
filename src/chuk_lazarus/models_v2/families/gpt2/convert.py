"""
Weight conversion utilities for GPT-2 models.

Converts HuggingFace checkpoint weights to our format.
GPT-2 uses different naming conventions than Llama-style models.
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
        >>> weights = load_weights("/path/to/gpt2")
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
            # GPT-2 Conv1D weights need transposition for nn.Linear
            # HF stores as (in_features, out_features), MLX expects (out_features, in_features)
            if ".weight" in name and weight.ndim == 2 and "wte" not in name and "wpe" not in name:
                weight = weight.T
            converted[new_name] = weight

    return converted


# Mapping from HuggingFace weight names to our weight names
# GPT-2 uses different conventions:
# - wte (word token embeddings) -> transformer.wte
# - wpe (word position embeddings) -> transformer.wpe
# - h.{i} (hidden layers) -> transformer.h.{i}
# - ln_f (final layer norm) -> transformer.ln_f
GPT2_WEIGHT_MAP = {
    # Embeddings (TokenEmbedding wraps nn.Embedding)
    "wte.weight": "transformer.wte.weight.weight",
    "wpe.weight": "transformer.wpe.weight",
    # Final norm
    "ln_f.weight": "transformer.ln_f.weight",
    "ln_f.bias": "transformer.ln_f.bias",
    # LM head (tied to embeddings by default)
    "lm_head.weight": "lm_head.lm_head.weight",
}


def convert_hf_weights(
    hf_weights: dict[str, np.ndarray],
    tie_word_embeddings: bool = True,
) -> dict[str, np.ndarray]:
    """
    Convert HuggingFace GPT-2 weights to our format.

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

        # Handle tied embeddings
        if tie_word_embeddings and hf_name == "lm_head.weight":
            continue

        converted[our_name] = weight

    return converted


def _map_weight_name(hf_name: str) -> str | None:
    """
    Map HuggingFace weight name to our weight name.

    GPT-2 layer structure (HF -> ours):
    - h.{i}.ln_1.* -> transformer.h.{i}.ln_1.*
    - h.{i}.attn.c_attn.* -> transformer.h.{i}.attn.c_attn.*
    - h.{i}.attn.c_proj.* -> transformer.h.{i}.attn.c_proj.*
    - h.{i}.ln_2.* -> transformer.h.{i}.ln_2.*
    - h.{i}.mlp.c_fc.* -> transformer.h.{i}.mlp.c_fc.*
    - h.{i}.mlp.c_proj.* -> transformer.h.{i}.mlp.c_proj.*
    """
    # Check direct mapping first
    if hf_name in GPT2_WEIGHT_MAP:
        return GPT2_WEIGHT_MAP[hf_name]

    # Pattern for layer weights: h.{i}.*
    layer_pattern = re.compile(r"h\.(\d+)\.(.*)")
    match = layer_pattern.match(hf_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Layer norms (keep original names)
        if rest == "ln_1.weight":
            return f"transformer.h.{layer_idx}.ln_1.weight"
        if rest == "ln_1.bias":
            return f"transformer.h.{layer_idx}.ln_1.bias"
        if rest == "ln_2.weight":
            return f"transformer.h.{layer_idx}.ln_2.weight"
        if rest == "ln_2.bias":
            return f"transformer.h.{layer_idx}.ln_2.bias"

        # Attention (keep original names)
        if rest == "attn.c_attn.weight":
            return f"transformer.h.{layer_idx}.attn.c_attn.weight"
        if rest == "attn.c_attn.bias":
            return f"transformer.h.{layer_idx}.attn.c_attn.bias"
        if rest == "attn.c_proj.weight":
            return f"transformer.h.{layer_idx}.attn.c_proj.weight"
        if rest == "attn.c_proj.bias":
            return f"transformer.h.{layer_idx}.attn.c_proj.bias"

        # MLP (keep original names)
        if rest == "mlp.c_fc.weight":
            return f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        if rest == "mlp.c_fc.bias":
            return f"transformer.h.{layer_idx}.mlp.c_fc.bias"
        if rest == "mlp.c_proj.weight":
            return f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        if rest == "mlp.c_proj.bias":
            return f"transformer.h.{layer_idx}.mlp.c_proj.bias"

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
    for hf_name, mapped_name in GPT2_WEIGHT_MAP.items():
        if mapped_name == our_name:
            return hf_name

    # Pattern matching for layers
    layer_pattern = re.compile(r"transformer\.h\.(\d+)\.(.*)")
    match = layer_pattern.match(our_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)

        # Reverse layer norms
        if rest == "ln_1.weight":
            return f"h.{layer_idx}.ln_1.weight"
        if rest == "ln_1.bias":
            return f"h.{layer_idx}.ln_1.bias"
        if rest == "ln_2.weight":
            return f"h.{layer_idx}.ln_2.weight"
        if rest == "ln_2.bias":
            return f"h.{layer_idx}.ln_2.bias"

        # Reverse attention
        if rest == "attn.c_attn.weight":
            return f"h.{layer_idx}.attn.c_attn.weight"
        if rest == "attn.c_attn.bias":
            return f"h.{layer_idx}.attn.c_attn.bias"
        if rest == "attn.c_proj.weight":
            return f"h.{layer_idx}.attn.c_proj.weight"
        if rest == "attn.c_proj.bias":
            return f"h.{layer_idx}.attn.c_proj.bias"

        # Reverse MLP
        if rest == "mlp.c_fc.weight":
            return f"h.{layer_idx}.mlp.c_fc.weight"
        if rest == "mlp.c_fc.bias":
            return f"h.{layer_idx}.mlp.c_fc.bias"
        if rest == "mlp.c_proj.weight":
            return f"h.{layer_idx}.mlp.c_proj.weight"
        if rest == "mlp.c_proj.bias":
            return f"h.{layer_idx}.mlp.c_proj.bias"

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
