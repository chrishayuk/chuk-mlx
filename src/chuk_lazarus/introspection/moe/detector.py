"""MoE architecture detection."""

from __future__ import annotations

import mlx.nn as nn

from .enums import MoEArchitecture
from .models import MoELayerInfo


def detect_moe_architecture(model: nn.Module) -> MoEArchitecture:
    """
    Detect which MoE architecture a model uses.

    Args:
        model: The model to analyze

    Returns:
        Detected MoEArchitecture
    """
    layers = _get_layers(model)
    if not layers:
        return MoEArchitecture.GENERIC

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Check for GPT-OSS batched experts
        if hasattr(mlp, "experts") and hasattr(mlp.experts, "gate_up_proj_blocks"):
            return MoEArchitecture.GPT_OSS

        # Check for Llama4 shared expert
        if hasattr(mlp, "shared_expert"):
            return MoEArchitecture.LLAMA4

        # Check for Granite hybrid
        if hasattr(layer, "mamba") or hasattr(layer, "mamba_block"):
            return MoEArchitecture.GRANITE_HYBRID

        # Check for standard Mixtral-style
        if hasattr(mlp, "experts") and isinstance(getattr(mlp, "experts", None), list):
            return MoEArchitecture.MIXTRAL

        # Check for generic router
        if hasattr(mlp, "router"):
            return MoEArchitecture.GENERIC

    return MoEArchitecture.GENERIC


def get_moe_layer_info(model: nn.Module, layer_idx: int) -> MoELayerInfo | None:
    """
    Get detailed information about an MoE layer.

    Args:
        model: The model
        layer_idx: Layer index to analyze

    Returns:
        MoELayerInfo or None if not an MoE layer
    """
    layers = _get_layers(model)
    if layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    router = getattr(mlp, "router", None)
    if router is None:
        return None

    # Get expert count
    num_experts = getattr(router, "num_experts", 8)
    num_experts_per_tok = getattr(router, "num_experts_per_tok", 2)

    # Check for shared expert
    has_shared = hasattr(mlp, "shared_expert")

    # Detect architecture
    architecture = detect_moe_architecture(model)

    # Determine router type
    router_type = "linear"
    uses_softmax = True
    uses_sigmoid = False

    if hasattr(router, "use_sigmoid") and router.use_sigmoid:
        uses_sigmoid = True
        uses_softmax = False

    return MoELayerInfo(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        has_shared_expert=has_shared,
        architecture=architecture,
        router_type=router_type,
        uses_softmax=uses_softmax,
        uses_sigmoid=uses_sigmoid,
    )


def get_moe_layers(model: nn.Module) -> list[int]:
    """
    Get indices of all MoE layers in a model.

    Args:
        model: The model

    Returns:
        List of layer indices that have MoE
    """
    layers = _get_layers(model)
    moe_layers = []

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp and hasattr(mlp, "router"):
            moe_layers.append(i)

    return moe_layers


def is_moe_model(model: nn.Module) -> bool:
    """Check if a model has any MoE layers."""
    return len(get_moe_layers(model)) > 0


def _get_layers(model: nn.Module) -> list[nn.Module]:
    """Extract transformer layers from a model."""
    # Try common attribute names
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            layers = getattr(submodel, "layers", None)
            if layers is not None:
                return list(layers)

    # Try direct layers attribute
    layers = getattr(model, "layers", None)
    if layers is not None:
        return list(layers)

    return []
