"""
LoRA (Low-Rank Adaptation) implementation.

LoRA adds trainable low-rank decomposition matrices to frozen weights:
    output = base_output + (x @ A @ B) * scaling

This allows efficient fine-tuning with minimal additional parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class LoRALinear(nn.Module):
    """
    LoRA adapter layer.

    Wraps a frozen linear layer and adds low-rank adaptation.

    Args:
        base_layer: The frozen linear layer to adapt
        rank: LoRA rank (low-rank dimension)
        alpha: Scaling factor (scaling = alpha / rank)
        dropout: Dropout rate for LoRA path

    Example:
        >>> base = nn.Linear(768, 768)
        >>> lora = LoRALinear(base, rank=8, alpha=16.0)
        >>> output = lora(x)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_rate = dropout

        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        # Initialize A with small random values, B with zeros
        # This ensures initial output equals base output
        self.lora_A = mx.random.normal((in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))

        # Freeze base layer
        self.base_layer.freeze()

        # Default to eval mode
        self._training = False

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation."""
        # Base output (frozen)
        base_output = self.base_layer(x)

        # LoRA output
        lora_input = x
        if self.dropout_rate > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_rate, x.shape)
            lora_input = x * mask / (1 - self.dropout_rate)

        lora_output = (lora_input @ self.lora_A @ self.lora_B) * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base layer for inference.

        Returns a new Linear layer with merged weights.
        """
        merged_weight = self.base_layer.weight + (self.lora_B.T @ self.lora_A.T) * self.scaling

        # Check if base layer has bias (MLX uses different attribute)
        has_bias = hasattr(self.base_layer, "bias") and self.base_layer.bias is not None

        merged = nn.Linear(
            self.base_layer.weight.shape[1],
            self.base_layer.weight.shape[0],
            bias=has_bias,
        )
        merged.weight = merged_weight
        if has_bias:
            merged.bias = self.base_layer.bias

        return merged

    @property
    def training(self) -> bool:
        """Check if in training mode."""
        if not hasattr(self, "_training"):
            self._training = False
        return self._training

    @training.setter
    def training(self, value: bool):
        self._training = value


def apply_lora(model: nn.Module, config: LoRAConfig) -> dict[str, LoRALinear]:
    """
    Apply LoRA adapters to target modules in a model.

    Args:
        model: The model to adapt
        config: LoRA configuration

    Returns:
        Dict mapping layer names to LoRALinear instances

    Example:
        >>> from chuk_lazarus.models_v2.families.llama import LlamaForCausalLM, LlamaConfig
        >>> model = LlamaForCausalLM(LlamaConfig.tiny())
        >>> lora_config = LoRAConfig(rank=8, alpha=16.0)
        >>> lora_layers = apply_lora(model, lora_config)
    """
    lora_layers = {}

    # Get the inner model if wrapped
    inner_model = model.model if hasattr(model, "model") else model

    # Find transformer layers
    layers = None
    if hasattr(inner_model, "layers"):
        layers = inner_model.layers
    elif hasattr(inner_model, "model") and hasattr(inner_model.model, "layers"):
        layers = inner_model.model.layers
    elif hasattr(inner_model, "backbone") and hasattr(inner_model.backbone, "layers"):
        layers = inner_model.backbone.layers

    if layers is None:
        logger.warning("Could not find transformer layers for LoRA")
        return lora_layers

    # Freeze all parameters first
    model.freeze()

    # Apply LoRA to each layer
    for layer_idx, layer in enumerate(layers):
        # Apply to attention projections
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for proj_name in config.target_modules:
                if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        if isinstance(proj, nn.Linear):
                            lora = LoRALinear(
                                proj,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(attn, proj_name, lora)
                            lora_layers[f"layers.{layer_idx}.self_attn.{proj_name}"] = lora

        # Apply to MLP projections
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            for proj_name in config.target_modules:
                if proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(mlp, proj_name):
                        proj = getattr(mlp, proj_name)
                        if isinstance(proj, nn.Linear):
                            lora = LoRALinear(
                                proj,
                                rank=config.rank,
                                alpha=config.alpha,
                                dropout=config.dropout,
                            )
                            setattr(mlp, proj_name, lora)
                            lora_layers[f"layers.{layer_idx}.mlp.{proj_name}"] = lora

    logger.info(f"Applied LoRA to {len(lora_layers)} layers")
    return lora_layers


def merge_lora_weights(model: nn.Module, lora_layers: dict[str, LoRALinear]):
    """
    Merge all LoRA weights into base model for efficient inference.

    This modifies the model in-place.

    Args:
        model: The model with LoRA adapters
        lora_layers: Dict of LoRA layers from apply_lora
    """
    inner_model = model.model if hasattr(model, "model") else model
    layers = None

    if hasattr(inner_model, "layers"):
        layers = inner_model.layers
    elif hasattr(inner_model, "backbone") and hasattr(inner_model.backbone, "layers"):
        layers = inner_model.backbone.layers

    if layers is None:
        logger.warning("Could not find layers to merge")
        return

    for name, lora_layer in lora_layers.items():
        parts = name.split(".")
        layer_idx = int(parts[1])
        module_name = parts[2]  # 'self_attn' or 'mlp'
        proj_name = parts[3]  # 'q_proj', etc.

        layer = layers[layer_idx]
        module = getattr(layer, module_name)

        merged = lora_layer.merge_weights()
        setattr(module, proj_name, merged)

    logger.info(f"Merged {len(lora_layers)} LoRA layers")


def count_lora_parameters(lora_layers: dict[str, LoRALinear]) -> int:
    """Count total trainable LoRA parameters."""
    total = 0
    for layer in lora_layers.values():
        total += layer.lora_A.size + layer.lora_B.size
    return total
