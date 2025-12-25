"""
Model introspection utilities.

Provides tools for understanding model architecture, size, and resource usage:
- Parameter counting (total, trainable, per-layer)
- FLOPs estimation (for planning and optimization)
- Memory footprint estimation
- Architecture introspection

These are essential for:
- MoE routing decisions (which expert to activate)
- Distributed execution planning
- Memory-constrained deployment
- Predictability mode (same config → same behavior)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .core.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterStats:
    """Statistics about model parameters."""

    total: int
    trainable: int
    frozen: int

    # Per-component breakdown
    embedding: int = 0
    attention: int = 0
    ffn: int = 0
    norm: int = 0
    head: int = 0
    other: int = 0

    # Per-layer stats
    per_layer: dict[int, int] = field(default_factory=dict)

    @property
    def trainable_ratio(self) -> float:
        """Ratio of trainable to total parameters."""
        return self.trainable / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Parameters: {self.total:,} total, "
            f"{self.trainable:,} trainable ({self.trainable_ratio:.1%})"
        )


@dataclass
class FLOPsEstimate:
    """
    Estimated FLOPs for forward pass.

    Note: These are estimates, not exact measurements.
    Actual FLOPs depend on hardware, batch size, and optimizations.
    """

    attention: int = 0
    ffn: int = 0
    embedding: int = 0
    norm: int = 0
    total: int = 0

    # Per-token estimates (useful for batching decisions)
    per_token: int = 0

    @classmethod
    def estimate_transformer(
        cls,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        intermediate_size: int,
        seq_length: int,
        num_heads: int,
        batch_size: int = 1,
    ) -> FLOPsEstimate:
        """
        Estimate FLOPs for a transformer model.

        Uses standard formulas for attention and FFN complexity.
        """
        # Embedding: vocab_size * hidden_size
        embedding_flops = batch_size * seq_length * hidden_size

        # Self-attention per layer:
        # - QKV projection: 3 * seq * hidden * hidden
        # - Attention scores: seq * seq * num_heads * head_dim
        # - Output projection: seq * hidden * hidden
        head_dim = hidden_size // num_heads
        attention_per_layer = (
            3 * seq_length * hidden_size * hidden_size  # QKV
            + 2 * seq_length * seq_length * num_heads * head_dim  # Attention
            + seq_length * hidden_size * hidden_size  # Output
        )
        attention_flops = batch_size * num_layers * attention_per_layer

        # FFN per layer: 2 * seq * hidden * intermediate
        ffn_per_layer = 2 * seq_length * hidden_size * intermediate_size
        ffn_flops = batch_size * num_layers * ffn_per_layer

        # LayerNorm: ~5 ops per element
        norm_flops = batch_size * num_layers * 2 * seq_length * hidden_size * 5

        total = embedding_flops + attention_flops + ffn_flops + norm_flops
        per_token = total // (batch_size * seq_length) if seq_length > 0 else 0

        return cls(
            attention=attention_flops,
            ffn=ffn_flops,
            embedding=embedding_flops,
            norm=norm_flops,
            total=total,
            per_token=per_token,
        )

    def summary(self) -> str:
        """Human-readable summary."""
        # Convert to TFLOPs for readability
        tflops = self.total / 1e12
        return f"Estimated FLOPs: {tflops:.2f}T (attention: {self.attention / 1e12:.2f}T, FFN: {self.ffn / 1e12:.2f}T)"


@dataclass
class MemoryEstimate:
    """
    Estimated memory usage.

    Includes:
    - Parameters (weights)
    - Activations (for training)
    - Gradients (for training)
    - Optimizer state (for training)
    """

    parameters_mb: float = 0.0
    activations_mb: float = 0.0
    gradients_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    total_training_mb: float = 0.0
    total_inference_mb: float = 0.0

    @classmethod
    def estimate(
        cls,
        num_parameters: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        batch_size: int = 1,
        dtype_bytes: int = 4,  # float32
        optimizer: str = "adam",  # adam uses 2x state
    ) -> MemoryEstimate:
        """
        Estimate memory usage for training and inference.
        """
        # Parameters
        params_bytes = num_parameters * dtype_bytes
        params_mb = params_bytes / (1024 * 1024)

        # Activations (rough estimate: each layer stores hidden states)
        activation_bytes = batch_size * seq_length * hidden_size * num_layers * dtype_bytes
        activations_mb = activation_bytes / (1024 * 1024)

        # Gradients (same size as parameters)
        gradients_mb = params_mb

        # Optimizer state (Adam: 2x parameters for momentum + variance)
        optimizer_multiplier = 2 if optimizer.lower() == "adam" else 1
        optimizer_state_mb = params_mb * optimizer_multiplier

        # Totals
        total_training_mb = params_mb + activations_mb + gradients_mb + optimizer_state_mb
        total_inference_mb = params_mb + (activations_mb / num_layers)  # Only one layer at a time

        return cls(
            parameters_mb=params_mb,
            activations_mb=activations_mb,
            gradients_mb=gradients_mb,
            optimizer_state_mb=optimizer_state_mb,
            total_training_mb=total_training_mb,
            total_inference_mb=total_inference_mb,
        )

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Memory: {self.total_inference_mb:.1f}MB inference, "
            f"{self.total_training_mb:.1f}MB training"
        )


@dataclass
class ModelInfo:
    """
    Core model information for routing, benchmarking, and deployment.

    This is the stable introspection contract that every Model should expose.
    Mirrors the structure of BatchPlan metadata for consistency.

    Used for:
    - MoE routing decisions
    - Distributed execution planning
    - Memory-constrained deployment
    - Gym-driven model selection
    - Registry queries
    """

    # Identity
    name: str = ""
    family: str = ""  # e.g., "llama", "mamba"

    # Architecture
    params: int = 0
    d_model: int = 0  # hidden_size
    n_layers: int = 0
    n_heads: int = 0
    vocab_size: int = 0

    # Sequence limits
    max_seq_len: int = 0
    context_window: int = 0  # May differ from max_seq_len for sliding window

    # Capabilities (boolean flags for fast filtering)
    supports_kv_cache: bool = False
    supports_generation: bool = False
    supports_lora: bool = True  # Most models do
    is_causal: bool = True

    # Resource estimates (for routing/scheduling)
    memory_mb: float = 0.0  # Inference memory
    flops_per_token: int = 0

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"{self.name}: {self.params:,} params, "
            f"d={self.d_model}, L={self.n_layers}, "
            f"ctx={self.max_seq_len}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "family": self.family,
            "params": self.params,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "context_window": self.context_window,
            "supports_kv_cache": self.supports_kv_cache,
            "supports_generation": self.supports_generation,
            "supports_lora": self.supports_lora,
            "is_causal": self.is_causal,
            "memory_mb": self.memory_mb,
            "flops_per_token": self.flops_per_token,
        }


@dataclass
class ModelCapabilities:
    """
    Capabilities exposed by a model.

    Used for:
    - Model registry (what can this model do?)
    - MoE routing (which expert handles this capability?)
    - Gym-driven selection (which model for which task?)
    """

    # Core capabilities
    is_causal_lm: bool = False
    is_classifier: bool = False
    is_encoder_decoder: bool = False

    # Architecture features
    supports_kv_cache: bool = False
    supports_lora: bool = False
    supports_moe: bool = False

    # Special capabilities (for routing)
    has_memory: bool = False
    has_planning: bool = False
    has_tool_use: bool = False

    # Domain specializations (from fine-tuning)
    domains: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        caps = []
        if self.is_causal_lm:
            caps.append("causal_lm")
        if self.is_classifier:
            caps.append("classifier")
        if self.supports_moe:
            caps.append("moe")
        if self.has_memory:
            caps.append("memory")
        if self.has_planning:
            caps.append("planning")
        if self.has_tool_use:
            caps.append("tool_use")
        return f"Capabilities: [{', '.join(caps)}]"


def count_parameters(model: nn.Module, trainable_only: bool = False) -> ParameterStats:
    """
    Count model parameters with detailed breakdown.

    Args:
        model: The model to analyze
        trainable_only: Only count trainable parameters

    Returns:
        ParameterStats with detailed breakdown
    """
    import mlx.utils

    total = 0
    trainable = 0
    frozen = 0

    embedding = 0
    attention = 0
    ffn = 0
    norm = 0
    head = 0
    other = 0

    per_layer: dict[int, int] = {}

    for name, param in mlx.utils.tree_flatten(model.parameters()):
        if not isinstance(param, mx.array):
            continue

        size = param.size
        total += size
        trainable += size  # MLX doesn't have requires_grad tracking the same way

        # Categorize by name patterns
        name_lower = name.lower()
        if "embed" in name_lower or "token" in name_lower:
            embedding += size
        elif (
            "attn" in name_lower
            or "attention" in name_lower
            or "q_proj" in name_lower
            or "k_proj" in name_lower
            or "v_proj" in name_lower
        ):
            attention += size
        elif (
            "mlp" in name_lower
            or "ffn" in name_lower
            or "feed_forward" in name_lower
            or "gate" in name_lower
            or "up_proj" in name_lower
            or "down_proj" in name_lower
        ):
            ffn += size
        elif "norm" in name_lower or "ln" in name_lower:
            norm += size
        elif "head" in name_lower or "lm_head" in name_lower or "output" in name_lower:
            head += size
        else:
            other += size

        # Try to extract layer number
        import re

        layer_match = re.search(r"layers?[_.]?(\d+)", name)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            per_layer[layer_idx] = per_layer.get(layer_idx, 0) + size

    return ParameterStats(
        total=total,
        trainable=trainable,
        frozen=frozen,
        embedding=embedding,
        attention=attention,
        ffn=ffn,
        norm=norm,
        head=head,
        other=other,
        per_layer=per_layer,
    )


def estimate_flops(
    config: ModelConfig,
    seq_length: int = 2048,
    batch_size: int = 1,
) -> FLOPsEstimate:
    """
    Estimate FLOPs for a model configuration.

    Args:
        config: Model configuration
        seq_length: Sequence length for estimation
        batch_size: Batch size for estimation

    Returns:
        FLOPsEstimate with breakdown
    """
    return FLOPsEstimate.estimate_transformer(
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
        intermediate_size=config.intermediate_size,
        seq_length=seq_length,
        num_heads=config.num_attention_heads,
        batch_size=batch_size,
    )


def estimate_memory(
    model: nn.Module,
    config: ModelConfig,
    seq_length: int = 2048,
    batch_size: int = 1,
    dtype_bytes: int = 4,
) -> MemoryEstimate:
    """
    Estimate memory usage for a model.

    Args:
        model: The model
        config: Model configuration
        seq_length: Sequence length for estimation
        batch_size: Batch size for estimation
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)

    Returns:
        MemoryEstimate with breakdown
    """
    stats = count_parameters(model)

    return MemoryEstimate.estimate(
        num_parameters=stats.total,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        seq_length=seq_length,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
    )


def detect_model_capabilities(model: nn.Module) -> ModelCapabilities:
    """
    Detect model capabilities from architecture.

    This inspects the model structure to determine what it can do.
    """
    from .models import CausalLM, SequenceClassifier, TokenClassifier

    caps = ModelCapabilities()

    # Check model type
    if isinstance(model, CausalLM):
        caps.is_causal_lm = True
        caps.supports_kv_cache = True
    if isinstance(model, (SequenceClassifier, TokenClassifier)):
        caps.is_classifier = True

    # Check for LoRA
    import mlx.utils

    for name, _ in mlx.utils.tree_flatten(model.parameters()):
        if "lora" in name.lower():
            caps.supports_lora = True
            break

    # Check for MoE
    for name, module in model.named_modules() if hasattr(model, "named_modules") else []:
        name_lower = name.lower()
        if "moe" in name_lower or "expert" in name_lower:
            caps.supports_moe = True
            break

    return caps


def introspect(
    model: nn.Module,
    config: ModelConfig | None = None,
    seq_length: int = 2048,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Full introspection of a model.

    Returns a dictionary with:
    - parameters: ParameterStats
    - flops: FLOPsEstimate (if config provided)
    - memory: MemoryEstimate
    - capabilities: ModelCapabilities

    Args:
        model: The model to introspect
        config: Optional model config (for FLOPs estimation)
        seq_length: Sequence length for estimates
        batch_size: Batch size for estimates

    Returns:
        Dictionary with all introspection results
    """
    params = count_parameters(model)
    caps = detect_model_capabilities(model)

    result = {
        "parameters": params,
        "capabilities": caps,
    }

    if config is not None:
        result["flops"] = estimate_flops(config, seq_length, batch_size)
        result["memory"] = estimate_memory(model, config, seq_length, batch_size)

    return result


def print_introspection(model: nn.Module, config: ModelConfig | None = None) -> None:
    """Print a human-readable introspection report."""
    info = introspect(model, config)

    print("=" * 60)
    print("Model Introspection Report")
    print("=" * 60)

    params: ParameterStats = info["parameters"]
    print(f"\n{params.summary()}")
    print(f"  - Embedding: {params.embedding:,}")
    print(f"  - Attention: {params.attention:,}")
    print(f"  - FFN: {params.ffn:,}")
    print(f"  - Norm: {params.norm:,}")
    print(f"  - Head: {params.head:,}")
    print(f"  - Other: {params.other:,}")

    caps: ModelCapabilities = info["capabilities"]
    print(f"\n{caps.summary()}")

    if "flops" in info:
        flops: FLOPsEstimate = info["flops"]
        print(f"\n{flops.summary()}")

    if "memory" in info:
        mem: MemoryEstimate = info["memory"]
        print(f"\n{mem.summary()}")

    print("=" * 60)


def get_model_info(
    model: nn.Module,
    config: ModelConfig | None = None,
    name: str = "",
    family: str = "",
) -> ModelInfo:
    """
    Build ModelInfo from a model instance.

    This is the canonical way to get the stable introspection contract.

    Args:
        model: The model to introspect
        config: Model configuration (required for full info)
        name: Model name (optional, for display)
        family: Model family (optional, e.g., "llama", "mamba")

    Returns:
        ModelInfo with all available information
    """
    params = count_parameters(model)
    caps = detect_model_capabilities(model)

    info = ModelInfo(
        name=name or model.__class__.__name__,
        family=family,
        params=params.total,
        supports_kv_cache=caps.supports_kv_cache,
        supports_generation=caps.is_causal_lm,
        supports_lora=caps.supports_lora,
        is_causal=caps.is_causal_lm,
    )

    if config is not None:
        info.d_model = config.hidden_size
        info.n_layers = config.num_hidden_layers
        info.n_heads = config.num_attention_heads
        info.vocab_size = config.vocab_size
        info.max_seq_len = getattr(config, "max_position_embeddings", 0)
        info.context_window = info.max_seq_len

        # Estimate resources
        flops = estimate_flops(config, seq_length=1, batch_size=1)
        info.flops_per_token = flops.per_token

        memory = estimate_memory(model, config, seq_length=1, batch_size=1)
        info.memory_mb = memory.total_inference_mb

    return info
