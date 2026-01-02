"""
Mixture of Experts (MoE) introspection utilities.

Provides tools for understanding MoE routing decisions, expert specialization,
and per-expert contributions to model predictions.

Key features:
- Router state capture (logits, weights, selected experts)
- Expert utilization analysis
- Per-expert contribution decomposition
- Expert-aware logit lens
- Router circuit analysis

Supported architectures:
- GPT-OSS (32 experts, 4 active)
- Llama4 (shared + routed experts)
- Granite-Hybrid (MoE + Mamba)
- Mixtral (8 experts, 2 active)
- Generic MoE component

Example:
    >>> from chuk_lazarus.introspection.moe import MoEHooks, MoECaptureConfig
    >>>
    >>> hooks = MoEHooks(model)
    >>> hooks.configure(MoECaptureConfig(
    ...     capture_router_logits=True,
    ...     capture_expert_assignments=True,
    ... ))
    >>>
    >>> output = hooks.forward(input_ids)
    >>>
    >>> # Analyze routing decisions
    >>> layer_4_routing = hooks.state.router_weights[4]
    >>> layer_4_experts = hooks.state.selected_experts[4]
    >>>
    >>> # Check expert utilization
    >>> utilization = hooks.get_expert_utilization(layer_idx=4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# MoE Architecture Detection
# =============================================================================


class MoEArchitecture(str, Enum):
    """Supported MoE architecture types."""

    GPT_OSS = "gpt_oss"
    """GPT-OSS: 32 experts, 4 active, MXFP4 quantized."""

    LLAMA4 = "llama4"
    """Llama 4: Shared expert (always active) + routed experts."""

    GRANITE_HYBRID = "granite_hybrid"
    """Granite Hybrid: MoE with Mamba-2/Attention hybrid."""

    MIXTRAL = "mixtral"
    """Mixtral: 8 experts, 2 active, standard routing."""

    GENERIC = "generic"
    """Generic MoE: Uses standard MoE component."""


@dataclass
class MoELayerInfo:
    """Information about an MoE layer."""

    layer_idx: int
    num_experts: int
    num_experts_per_tok: int
    has_shared_expert: bool = False
    architecture: MoEArchitecture = MoEArchitecture.GENERIC

    # Router details
    router_type: str = "linear"  # "linear", "sigmoid", etc.
    uses_softmax: bool = True
    uses_sigmoid: bool = False


def detect_moe_architecture(model: nn.Module) -> MoEArchitecture:
    """
    Detect which MoE architecture a model uses.

    Args:
        model: The model to analyze

    Returns:
        Detected MoEArchitecture
    """
    model_class = type(model).__name__.lower()

    if "gptoss" in model_class or "gpt_oss" in model_class:
        return MoEArchitecture.GPT_OSS
    elif "llama4" in model_class:
        return MoEArchitecture.LLAMA4
    elif "granitehybrid" in model_class or "granite" in model_class:
        # Check if it's the hybrid variant
        if hasattr(model, "config") and hasattr(model.config, "is_moe"):
            if model.config.is_moe:
                return MoEArchitecture.GRANITE_HYBRID
    elif "mixtral" in model_class:
        return MoEArchitecture.MIXTRAL

    # Check for generic MoE patterns
    layers = _get_layers(model)
    if layers and len(layers) > 0:
        layer = layers[0]
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "router") and hasattr(mlp, "experts"):
                return MoEArchitecture.GENERIC

    return MoEArchitecture.GENERIC


def get_moe_layer_info(model: nn.Module, layer_idx: int) -> MoELayerInfo | None:
    """
    Get MoE information for a specific layer.

    Args:
        model: The model
        layer_idx: Layer index

    Returns:
        MoELayerInfo or None if not an MoE layer
    """
    layers = _get_layers(model)
    if layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]
    if not hasattr(layer, "mlp"):
        return None

    mlp = layer.mlp

    # Check for MoE patterns
    if not (hasattr(mlp, "router") or hasattr(mlp, "experts")):
        return None

    arch = detect_moe_architecture(model)

    # Extract MoE parameters based on architecture
    num_experts = 1
    num_experts_per_tok = 1
    has_shared = False
    uses_sigmoid = False

    # Check various attribute names for num_experts
    if hasattr(mlp, "num_experts"):
        num_experts = mlp.num_experts
    elif hasattr(mlp, "num_local_experts"):
        num_experts = mlp.num_local_experts
    elif hasattr(mlp, "router") and hasattr(mlp.router, "num_experts"):
        num_experts = mlp.router.num_experts
    elif hasattr(mlp, "router") and hasattr(mlp.router, "weight"):
        # Infer from router weight shape (num_experts, hidden_size)
        num_experts = mlp.router.weight.shape[0]

    # Check various attribute names for num_experts_per_tok
    if hasattr(mlp, "num_experts_per_tok"):
        num_experts_per_tok = mlp.num_experts_per_tok
    elif hasattr(mlp, "router") and hasattr(mlp.router, "num_experts_per_tok"):
        num_experts_per_tok = mlp.router.num_experts_per_tok

    # Check for shared expert (Llama4, Granite)
    if hasattr(mlp, "shared_expert"):
        has_shared = True

    # Check router type
    if arch in (MoEArchitecture.LLAMA4, MoEArchitecture.GRANITE_HYBRID):
        uses_sigmoid = True

    return MoELayerInfo(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        has_shared_expert=has_shared,
        architecture=arch,
        uses_softmax=not uses_sigmoid,
        uses_sigmoid=uses_sigmoid,
    )


def _get_layers(model: nn.Module) -> list[nn.Module]:
    """Get transformer layers from model."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return list(inner.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


# =============================================================================
# MoE Capture Configuration
# =============================================================================


class MoECaptureConfig(BaseModel):
    """Configuration for MoE state capture."""

    capture_router_logits: bool = Field(
        default=True,
        description="Capture raw router logits before softmax/sigmoid.",
    )

    capture_router_weights: bool = Field(
        default=True,
        description="Capture normalized routing weights.",
    )

    capture_selected_experts: bool = Field(
        default=True,
        description="Capture which experts were selected per token.",
    )

    capture_expert_contributions: bool = Field(
        default=False,
        description="Capture per-expert output contributions. Memory intensive.",
    )

    capture_shared_expert: bool = Field(
        default=True,
        description="Capture shared expert output separately (for Llama4-style).",
    )

    layers: list[int] | None = Field(
        default=None,
        description="Which layers to capture. None = all MoE layers.",
    )

    detach: bool = Field(
        default=True,
        description="Detach captured tensors from computation graph.",
    )

    model_config = ConfigDict(use_enum_values=False)


# =============================================================================
# MoE Captured State
# =============================================================================


class MoECapturedState(BaseModel):
    """Container for captured MoE states."""

    # Router decisions
    router_logits: dict[int, Any] = Field(
        default_factory=dict,
        description="Raw router logits per layer. Shape: [batch*seq, num_experts]",
    )

    router_weights: dict[int, Any] = Field(
        default_factory=dict,
        description="Normalized routing weights. Shape: [batch*seq, num_experts_per_tok]",
    )

    selected_experts: dict[int, Any] = Field(
        default_factory=dict,
        description="Selected expert indices. Shape: [batch*seq, num_experts_per_tok]",
    )

    # Expert outputs
    expert_contributions: dict[int, dict[int, Any]] = Field(
        default_factory=dict,
        description="Per-expert output contributions. expert_contributions[layer][expert_idx]",
    )

    shared_expert_output: dict[int, Any] = Field(
        default_factory=dict,
        description="Shared expert output (for Llama4-style). Shape: [batch, seq, hidden]",
    )

    combined_expert_output: dict[int, Any] = Field(
        default_factory=dict,
        description="Combined MoE output. Shape: [batch, seq, hidden]",
    )

    # Metadata
    batch_size: int = 0
    seq_len: int = 0
    architecture: MoEArchitecture = MoEArchitecture.GENERIC

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def clear(self) -> None:
        """Clear all captured states."""
        self.router_logits.clear()
        self.router_weights.clear()
        self.selected_experts.clear()
        self.expert_contributions.clear()
        self.shared_expert_output.clear()
        self.combined_expert_output.clear()
        self.batch_size = 0
        self.seq_len = 0

    @property
    def captured_layers(self) -> list[int]:
        """List of layers with captured MoE state."""
        return sorted(self.router_weights.keys())

    @property
    def num_layers_captured(self) -> int:
        """Number of layers with captured state."""
        return len(self.router_weights)


# =============================================================================
# MoE Analysis Results
# =============================================================================


@dataclass
class ExpertUtilization:
    """Expert utilization statistics for a layer."""

    layer_idx: int
    num_experts: int
    num_tokens: int

    # Per-expert stats
    token_counts: mx.array  # Shape: [num_experts]
    utilization_pct: mx.array  # Shape: [num_experts]

    # Aggregate stats
    most_used_expert: int
    least_used_expert: int
    load_balance_score: float  # 1.0 = perfectly balanced

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Layer {self.layer_idx}: {self.num_experts} experts, "
            f"balance={self.load_balance_score:.2%}, "
            f"most_used=expert_{self.most_used_expert}, "
            f"least_used=expert_{self.least_used_expert}"
        )


@dataclass
class RouterEntropy:
    """Router decision entropy analysis."""

    layer_idx: int

    # Per-position entropy
    entropy_per_position: mx.array  # Shape: [batch, seq]

    # Aggregate
    mean_entropy: float
    max_entropy: float  # log(num_experts) for uniform
    normalized_entropy: float  # mean / max (0=confident, 1=uniform)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Layer {self.layer_idx}: entropy={self.mean_entropy:.3f} "
            f"(normalized={self.normalized_entropy:.2%})"
        )


@dataclass
class ExpertSpecialization:
    """Analysis of what each expert specializes in."""

    layer_idx: int
    expert_idx: int

    # Token statistics
    assigned_tokens: list[int]
    token_frequencies: dict[int, int]

    # Top tokens this expert handles
    top_tokens: list[tuple[int, int]]  # (token_id, count)

    # Activation patterns
    mean_activation_norm: float
    activation_variance: float


# =============================================================================
# MoE Hooks - Main Interface
# =============================================================================


class MoEHooks:
    """
    Hook manager for capturing MoE layer internals.

    This class wraps a model's forward pass to capture router decisions,
    expert selections, and optionally per-expert contributions.

    Example:
        >>> hooks = MoEHooks(model)
        >>> hooks.configure(MoECaptureConfig(
        ...     capture_router_logits=True,
        ...     capture_expert_contributions=True,
        ... ))
        >>>
        >>> logits = hooks.forward(input_ids)
        >>>
        >>> # Analyze layer 4
        >>> routing = hooks.state.router_weights[4]
        >>> experts = hooks.state.selected_experts[4]
        >>> utilization = hooks.get_expert_utilization(4)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize MoE hooks.

        Args:
            model: The MoE model to hook into
        """
        self.model = model
        self.config = MoECaptureConfig()
        self.state = MoECapturedState()
        self.architecture = detect_moe_architecture(model)
        self._moe_layer_indices: list[int] | None = None

    def configure(self, config: MoECaptureConfig) -> MoEHooks:
        """
        Configure what to capture.

        Args:
            config: Capture configuration

        Returns:
            Self for chaining
        """
        self.config = config
        return self

    @property
    def moe_layer_indices(self) -> list[int]:
        """Get indices of layers that have MoE."""
        if self._moe_layer_indices is None:
            self._moe_layer_indices = []
            layers = _get_layers(self.model)
            for i, layer in enumerate(layers):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    self._moe_layer_indices.append(i)
        return self._moe_layer_indices

    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if we should capture this layer."""
        if layer_idx not in self.moe_layer_indices:
            return False
        if self.config.layers is None:
            return True
        return layer_idx in self.config.layers

    def _capture_gpt_oss_moe(
        self,
        moe_layer: nn.Module,
        x: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Capture GPT-OSS style MoE layer with router interception.

        Args:
            moe_layer: The GptOssMoE layer
            x: Input tensor [batch, seq, hidden]
            layer_idx: Layer index

        Returns:
            MoE output
        """
        batch_size, seq_len, hidden_size = x.shape
        self.state.batch_size = batch_size
        self.state.seq_len = seq_len

        # Flatten for router
        x_flat = x.reshape(-1, hidden_size)

        # Capture router logits before top-k selection
        if self.config.capture_router_logits:
            router = moe_layer.router
            logits = x_flat @ router.weight.T + router.bias
            if self.config.detach:
                logits = mx.stop_gradient(logits)
            self.state.router_logits[layer_idx] = logits

        # Get routing decision
        weights, indices = moe_layer.router(x_flat)

        if self.config.capture_router_weights:
            w = weights if not self.config.detach else mx.stop_gradient(weights)
            self.state.router_weights[layer_idx] = w.reshape(batch_size, seq_len, -1)

        if self.config.capture_selected_experts:
            idx = indices if not self.config.detach else mx.stop_gradient(indices)
            self.state.selected_experts[layer_idx] = idx.reshape(batch_size, seq_len, -1)

        # Apply experts (using original forward)
        output = moe_layer.experts(x_flat, indices, weights)
        output = output.reshape(batch_size, seq_len, hidden_size)

        if self.config.detach:
            self.state.combined_expert_output[layer_idx] = mx.stop_gradient(output)
        else:
            self.state.combined_expert_output[layer_idx] = output

        return output

    def _capture_llama4_moe(
        self,
        moe_layer: nn.Module,
        x: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Capture Llama4 style MoE (shared + routed experts).

        Args:
            moe_layer: The Llama4MoE layer
            x: Input tensor [batch, seq, hidden]
            layer_idx: Layer index

        Returns:
            MoE output
        """
        batch_size, seq_len, hidden_size = x.shape
        self.state.batch_size = batch_size
        self.state.seq_len = seq_len

        # Capture shared expert output
        if self.config.capture_shared_expert and hasattr(moe_layer, "shared_expert"):
            shared_out = moe_layer.shared_expert(x)
            if self.config.detach:
                shared_out = mx.stop_gradient(shared_out)
            self.state.shared_expert_output[layer_idx] = shared_out

        # Capture router logits (before sigmoid)
        if self.config.capture_router_logits:
            router_logits = moe_layer.router(x)
            if self.config.detach:
                router_logits = mx.stop_gradient(router_logits)
            self.state.router_logits[layer_idx] = router_logits

        # Let the layer do its normal forward
        output = moe_layer(x)

        # We can't easily extract weights/indices from Llama4 without modifying
        # the layer, so we'd need to re-run the routing logic here
        # For now, store the combined output
        if self.config.detach:
            self.state.combined_expert_output[layer_idx] = mx.stop_gradient(output)
        else:
            self.state.combined_expert_output[layer_idx] = output

        return output

    def _capture_generic_moe(
        self,
        moe_layer: nn.Module,
        x: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Capture generic MoE layer.

        Args:
            moe_layer: The MoE layer
            x: Input tensor
            layer_idx: Layer index

        Returns:
            MoE output
        """
        batch_size, seq_len, hidden_size = x.shape
        self.state.batch_size = batch_size
        self.state.seq_len = seq_len

        # Try to capture router if it exists
        if self.config.capture_router_logits and hasattr(moe_layer, "router"):
            router = moe_layer.router
            if hasattr(router, "gate"):
                # Standard MoERouter pattern
                logits = router.gate(x)
                if self.config.detach:
                    logits = mx.stop_gradient(logits)
                self.state.router_logits[layer_idx] = logits

        # Run the layer
        output = moe_layer(x)

        if self.config.detach:
            self.state.combined_expert_output[layer_idx] = mx.stop_gradient(output)
        else:
            self.state.combined_expert_output[layer_idx] = output

        return output

    def forward(
        self,
        input_ids: mx.array,
        return_logits: bool = True,
    ) -> mx.array | None:
        """
        Run forward pass with MoE state capture.

        This runs the model's forward pass and captures MoE layer internals
        by intercepting the MoE components before the full layer runs.

        Args:
            input_ids: Input token IDs [batch, seq] or [seq]
            return_logits: Whether to compute and return logits

        Returns:
            Logits if return_logits=True, else None
        """
        self.state.clear()
        self.state.architecture = self.architecture

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        batch_size, seq_len = input_ids.shape
        self.state.batch_size = batch_size
        self.state.seq_len = seq_len

        # Get model components
        layers = _get_layers(self.model)

        # Get embeddings
        if hasattr(self.model, "model"):
            embed = self.model.model.embed_tokens
        elif hasattr(self.model, "embed_tokens"):
            embed = self.model.embed_tokens
        else:
            raise ValueError("Cannot find embedding layer")

        h = embed(input_ids)

        # Process each layer - run the full layer but capture MoE state
        for layer_idx, layer in enumerate(layers):
            # Before running the layer, capture MoE routing info if needed
            if self._should_capture_layer(layer_idx) and hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "router"):
                    # Capture router state before the layer runs
                    # We need the input to the MoE, which is after attention + norm
                    # For now, we'll capture by running the router separately
                    self._pre_capture_moe_state(layer, h, layer_idx)

            # Run the full layer (this handles attention correctly)
            try:
                layer_out = layer(h)
            except TypeError:
                # Some layers might need a mask argument
                try:
                    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                    mask = mask.astype(h.dtype)
                    layer_out = layer(h, mask=mask)
                except TypeError:
                    layer_out = layer(h, mask=None)

            # Extract hidden state from layer output
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        # Final norm
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            h = self.model.model.norm(h)
        elif hasattr(self.model, "norm"):
            h = self.model.norm(h)

        # LM head
        if return_logits:
            if hasattr(self.model, "lm_head"):
                head_out = self.model.lm_head(h)
                if hasattr(head_out, "logits"):
                    return head_out.logits
                return head_out

        return None

    def _pre_capture_moe_state(
        self,
        layer: nn.Module,
        h: mx.array,
        layer_idx: int,
    ) -> None:
        """
        Capture MoE routing state before the layer runs.

        This runs the normalization and router to capture routing decisions
        without modifying the actual computation.

        Args:
            layer: The transformer layer
            h: Hidden state before this layer
            layer_idx: Layer index
        """
        mlp = layer.mlp

        # Get the MoE input (after attention would be applied)
        # For accurate capture, we need to run attention first, but that's complex
        # Instead, we'll use the current hidden state as an approximation
        # This works because we want to understand routing patterns

        # Try to get normalized input
        if hasattr(layer, "post_attention_layernorm"):
            h_normed = layer.post_attention_layernorm(h)
        elif hasattr(layer, "input_layernorm"):
            h_normed = layer.input_layernorm(h)
        else:
            h_normed = h

        batch_size, seq_len, hidden_size = h_normed.shape

        # Capture router logits
        if self.config.capture_router_logits and hasattr(mlp, "router"):
            router = mlp.router
            h_flat = h_normed.reshape(-1, hidden_size)

            # Get raw logits
            if hasattr(router, "weight"):
                logits = h_flat @ router.weight.T
                if hasattr(router, "bias"):
                    logits = logits + router.bias
            elif hasattr(router, "gate"):
                logits = router.gate(h_flat)
            else:
                logits = None

            if logits is not None:
                if self.config.detach:
                    logits = mx.stop_gradient(logits)
                self.state.router_logits[layer_idx] = logits

        # Capture routing weights and indices
        if (self.config.capture_router_weights or self.config.capture_selected_experts) and hasattr(mlp, "router"):
            router = mlp.router
            h_flat = h_normed.reshape(-1, hidden_size)

            try:
                router_output = router(h_flat)

                # Check if router returns (weights, indices) or just logits
                if isinstance(router_output, tuple) and len(router_output) == 2:
                    weights, indices = router_output
                else:
                    # Router returns logits - compute top-k ourselves
                    logits = router_output

                    # Get num_experts_per_tok
                    k = getattr(mlp, "num_experts_per_tok", 4)

                    # Compute top-k
                    indices = mx.argsort(logits, axis=-1)[:, -k:][:, ::-1]
                    weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)

                if self.config.capture_router_weights:
                    w = weights if not self.config.detach else mx.stop_gradient(weights)
                    self.state.router_weights[layer_idx] = w.reshape(batch_size, seq_len, -1)

                if self.config.capture_selected_experts:
                    idx = indices if not self.config.detach else mx.stop_gradient(indices)
                    self.state.selected_experts[layer_idx] = idx.reshape(batch_size, seq_len, -1)
            except Exception as e:
                # Router might have different interface
                pass

        # Capture shared expert output if present
        if self.config.capture_shared_expert and hasattr(mlp, "shared_expert"):
            shared_out = mlp.shared_expert(h_normed)
            if self.config.detach:
                shared_out = mx.stop_gradient(shared_out)
            self.state.shared_expert_output[layer_idx] = shared_out

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_expert_utilization(self, layer_idx: int) -> ExpertUtilization | None:
        """
        Compute expert utilization statistics for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            ExpertUtilization or None if layer not captured
        """
        if layer_idx not in self.state.selected_experts:
            return None

        experts = self.state.selected_experts[layer_idx]

        # Get layer info for num_experts
        info = get_moe_layer_info(self.model, layer_idx)
        if info is None:
            return None

        num_experts = info.num_experts

        # Flatten to get all expert assignments
        flat_experts = experts.flatten()
        num_tokens = flat_experts.size

        # Count per-expert
        token_counts = mx.zeros((num_experts,))
        for expert_idx in range(num_experts):
            count = mx.sum(flat_experts == expert_idx)
            token_counts = token_counts.at[expert_idx].add(count)

        mx.eval(token_counts)

        # Compute utilization
        utilization_pct = token_counts / num_tokens

        # Find most/least used
        counts_list = token_counts.tolist()
        most_used = int(mx.argmax(token_counts))
        least_used = int(mx.argmin(token_counts))

        # Load balance score (1.0 = perfectly uniform)
        # Using coefficient of variation: lower = more balanced
        mean_count = float(mx.mean(token_counts))
        std_count = float(mx.std(token_counts))
        cv = std_count / mean_count if mean_count > 0 else 0
        # Convert to 0-1 score where 1 is perfect balance
        load_balance = max(0, 1 - cv)

        return ExpertUtilization(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_tokens=num_tokens,
            token_counts=token_counts,
            utilization_pct=utilization_pct,
            most_used_expert=most_used,
            least_used_expert=least_used,
            load_balance_score=load_balance,
        )

    def get_router_entropy(self, layer_idx: int) -> RouterEntropy | None:
        """
        Compute router decision entropy for a layer.

        High entropy = uncertain routing (closer to uniform)
        Low entropy = confident routing (clear expert preference)

        Args:
            layer_idx: Layer index

        Returns:
            RouterEntropy or None if layer not captured
        """
        if layer_idx not in self.state.router_logits:
            return None

        logits = self.state.router_logits[layer_idx]

        # Compute softmax probabilities
        probs = mx.softmax(logits, axis=-1)

        # Compute entropy: -sum(p * log(p))
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs, axis=-1)

        # Reshape to [batch, seq] if needed
        if entropy.ndim == 1 and self.state.batch_size > 0:
            entropy = entropy.reshape(self.state.batch_size, self.state.seq_len)

        mx.eval(entropy)

        # Max entropy is log(num_experts)
        num_experts = logits.shape[-1]
        max_entropy = float(mx.log(mx.array(num_experts)))

        mean_ent = float(mx.mean(entropy))
        normalized = mean_ent / max_entropy if max_entropy > 0 else 0

        return RouterEntropy(
            layer_idx=layer_idx,
            entropy_per_position=entropy,
            mean_entropy=mean_ent,
            max_entropy=max_entropy,
            normalized_entropy=normalized,
        )

    def get_routing_pattern(
        self,
        layer_idx: int,
        position: int = -1,
    ) -> dict[str, Any] | None:
        """
        Get detailed routing pattern for a specific position.

        Args:
            layer_idx: Layer index
            position: Sequence position (-1 for last)

        Returns:
            Dict with routing details or None
        """
        if layer_idx not in self.state.router_weights:
            return None

        weights = self.state.router_weights[layer_idx]
        experts = self.state.selected_experts[layer_idx]

        # Get specific position
        if weights.ndim == 3:
            weights = weights[0, position]  # First batch, specified position
            experts = experts[0, position]
        else:
            weights = weights[position]
            experts = experts[position]

        mx.eval(weights, experts)

        return {
            "layer_idx": layer_idx,
            "position": position,
            "selected_experts": experts.tolist(),
            "routing_weights": weights.tolist(),
            "top_expert": int(experts[0]),
            "top_weight": float(weights[0]),
        }

    def compare_routing_across_layers(self) -> dict[int, dict[str, float]]:
        """
        Compare routing statistics across all captured layers.

        Returns:
            Dict mapping layer_idx to stats dict
        """
        results = {}

        for layer_idx in self.state.captured_layers:
            utilization = self.get_expert_utilization(layer_idx)
            entropy = self.get_router_entropy(layer_idx)

            if utilization and entropy:
                results[layer_idx] = {
                    "load_balance": utilization.load_balance_score,
                    "mean_entropy": entropy.mean_entropy,
                    "normalized_entropy": entropy.normalized_entropy,
                    "most_used_expert": utilization.most_used_expert,
                    "least_used_expert": utilization.least_used_expert,
                }

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MoEHooks(arch={self.architecture.value}, "
            f"moe_layers={len(self.moe_layer_indices)}, "
            f"captured={self.state.num_layers_captured})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_moe_model(
    model: nn.Module,
    input_ids: mx.array,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive MoE analysis on a model.

    Args:
        model: The MoE model
        input_ids: Input token IDs
        layers: Which layers to analyze (None = all)

    Returns:
        Dict with analysis results
    """
    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(
        capture_router_logits=True,
        capture_router_weights=True,
        capture_selected_experts=True,
        layers=layers,
    ))

    _ = hooks.forward(input_ids)

    return {
        "architecture": hooks.architecture.value,
        "moe_layers": hooks.moe_layer_indices,
        "layer_stats": hooks.compare_routing_across_layers(),
        "captured_state": hooks.state,
    }


def print_moe_analysis(
    model: nn.Module,
    input_ids: mx.array,
    layers: list[int] | None = None,
) -> None:
    """Print a human-readable MoE analysis report."""
    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(layers=layers))

    _ = hooks.forward(input_ids)

    print("=" * 60)
    print("MoE Analysis Report")
    print("=" * 60)
    print(f"Architecture: {hooks.architecture.value}")
    print(f"MoE Layers: {hooks.moe_layer_indices}")
    print(f"Captured: {hooks.state.num_layers_captured} layers")
    print()

    for layer_idx in hooks.state.captured_layers:
        utilization = hooks.get_expert_utilization(layer_idx)
        entropy = hooks.get_router_entropy(layer_idx)

        if utilization:
            print(utilization.summary())
        if entropy:
            print(f"  {entropy.summary()}")

    print("=" * 60)


# =============================================================================
# MoE Expert Ablation
# =============================================================================


@dataclass
class ExpertAblationResult:
    """Result of ablating specific experts."""

    layer_idx: int
    ablated_experts: list[int]
    original_output: str
    ablated_output: str
    output_changed: bool
    token_diff: int  # Number of tokens that changed


class MoEAblation:
    """
    Expert-level ablation for MoE models.

    Allows ablating specific experts to understand their function,
    or forcing routing to specific experts.

    Example:
        >>> ablation = MoEAblation(model, tokenizer)
        >>>
        >>> # Ablate expert 5 in layer 4
        >>> result = ablation.ablate_expert(
        ...     prompt="Hello world",
        ...     layer_idx=4,
        ...     expert_idx=5,
        ... )
        >>> print(f"Output changed: {result.output_changed}")
        >>>
        >>> # Force routing to expert 0 only
        >>> result = ablation.force_expert(
        ...     prompt="Hello world",
        ...     layer_idx=4,
        ...     expert_idx=0,
        ... )
    """

    def __init__(self, model: nn.Module, tokenizer: Any = None):
        """
        Initialize MoE ablation.

        Args:
            model: The MoE model
            tokenizer: Tokenizer for encoding/decoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.architecture = detect_moe_architecture(model)
        self._original_routers: dict[int, Any] = {}

    def _get_moe_layer(self, layer_idx: int) -> nn.Module | None:
        """Get the MoE module for a layer."""
        layers = _get_layers(self.model)
        if layer_idx >= len(layers):
            return None
        layer = layers[layer_idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            return layer.mlp
        return None

    def ablate_expert(
        self,
        prompt: str | mx.array,
        layer_idx: int,
        expert_idx: int | list[int],
        max_tokens: int = 50,
    ) -> ExpertAblationResult:
        """
        Ablate specific expert(s) by zeroing their contribution.

        This modifies the routing weights to exclude the specified expert(s).

        Args:
            prompt: Input prompt or token IDs
            layer_idx: Layer index
            expert_idx: Expert index or list of indices to ablate
            max_tokens: Max tokens to generate

        Returns:
            ExpertAblationResult with comparison
        """
        if isinstance(expert_idx, int):
            expert_idx = [expert_idx]

        # Get input IDs
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            input_ids = mx.array([self.tokenizer.encode(prompt)])
        else:
            input_ids = prompt if prompt.ndim == 2 else prompt[None, :]

        # Generate original output
        original_output = self._generate(input_ids, max_tokens)

        # Get MoE layer
        moe = self._get_moe_layer(layer_idx)
        if moe is None:
            return ExpertAblationResult(
                layer_idx=layer_idx,
                ablated_experts=expert_idx,
                original_output=original_output,
                ablated_output=original_output,
                output_changed=False,
                token_diff=0,
            )

        # Create ablated router wrapper
        original_router = moe.router

        class AblatedRouter(nn.Module):
            def __init__(self, router, ablated_experts):
                super().__init__()
                self._router = router
                self._ablated = set(ablated_experts)
                # Copy attributes
                for attr in ["num_experts", "num_experts_per_tok", "weight", "bias"]:
                    if hasattr(router, attr):
                        setattr(self, attr, getattr(router, attr))

            def __call__(self, x):
                weights, indices = self._router(x)
                # Zero out ablated experts
                for exp_idx in self._ablated:
                    mask = indices == exp_idx
                    weights = mx.where(mask, mx.zeros_like(weights), weights)
                # Renormalize
                weight_sum = mx.sum(weights, axis=-1, keepdims=True)
                weights = weights / (weight_sum + 1e-10)
                return weights, indices

        # Apply ablation
        moe.router = AblatedRouter(original_router, expert_idx)

        try:
            ablated_output = self._generate(input_ids, max_tokens)
        finally:
            # Restore original router
            moe.router = original_router

        # Compare outputs
        output_changed = original_output != ablated_output
        token_diff = self._count_token_diff(original_output, ablated_output)

        return ExpertAblationResult(
            layer_idx=layer_idx,
            ablated_experts=expert_idx,
            original_output=original_output,
            ablated_output=ablated_output,
            output_changed=output_changed,
            token_diff=token_diff,
        )

    def force_expert(
        self,
        prompt: str | mx.array,
        layer_idx: int,
        expert_idx: int,
        max_tokens: int = 50,
    ) -> ExpertAblationResult:
        """
        Force all tokens to route to a specific expert.

        Useful for understanding what a single expert contributes.

        Args:
            prompt: Input prompt or token IDs
            layer_idx: Layer index
            expert_idx: Expert to force routing to
            max_tokens: Max tokens to generate

        Returns:
            ExpertAblationResult with comparison
        """
        # Get input IDs
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            input_ids = mx.array([self.tokenizer.encode(prompt)])
        else:
            input_ids = prompt if prompt.ndim == 2 else prompt[None, :]

        # Generate original output
        original_output = self._generate(input_ids, max_tokens)

        # Get MoE layer
        moe = self._get_moe_layer(layer_idx)
        if moe is None:
            return ExpertAblationResult(
                layer_idx=layer_idx,
                ablated_experts=[],
                original_output=original_output,
                ablated_output=original_output,
                output_changed=False,
                token_diff=0,
            )

        # Create forced router wrapper
        original_router = moe.router

        class ForcedRouter(nn.Module):
            def __init__(self, router, forced_expert):
                super().__init__()
                self._router = router
                self._forced = forced_expert
                for attr in ["num_experts", "num_experts_per_tok", "weight", "bias"]:
                    if hasattr(router, attr):
                        setattr(self, attr, getattr(router, attr))

            def __call__(self, x):
                weights, indices = self._router(x)
                # Force all routing to single expert
                batch_size = weights.shape[0]
                k = weights.shape[1] if weights.ndim > 1 else 1
                forced_indices = mx.full((batch_size, k), self._forced, dtype=indices.dtype)
                forced_weights = mx.ones((batch_size, k)) / k
                return forced_weights, forced_indices

        # Apply forced routing
        moe.router = ForcedRouter(original_router, expert_idx)

        try:
            forced_output = self._generate(input_ids, max_tokens)
        finally:
            moe.router = original_router

        output_changed = original_output != forced_output
        token_diff = self._count_token_diff(original_output, forced_output)

        return ExpertAblationResult(
            layer_idx=layer_idx,
            ablated_experts=[expert_idx],  # Forced, not ablated
            original_output=original_output,
            ablated_output=forced_output,
            output_changed=output_changed,
            token_diff=token_diff,
        )

    def sweep_experts(
        self,
        prompt: str | mx.array,
        layer_idx: int,
        max_tokens: int = 50,
    ) -> list[ExpertAblationResult]:
        """
        Sweep through all experts, ablating each one individually.

        Args:
            prompt: Input prompt
            layer_idx: Layer index
            max_tokens: Max tokens to generate

        Returns:
            List of ablation results for each expert
        """
        info = get_moe_layer_info(self.model, layer_idx)
        if info is None:
            return []

        results = []
        for expert_idx in range(info.num_experts):
            result = self.ablate_expert(prompt, layer_idx, expert_idx, max_tokens)
            results.append(result)

        return results

    def _generate(self, input_ids: mx.array, max_tokens: int) -> str:
        """Generate text from input IDs."""
        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens)
            mx.eval(output_ids)
            if self.tokenizer:
                return self.tokenizer.decode(output_ids[0].tolist())
            return str(output_ids[0].tolist())
        else:
            # Simple greedy generation
            current_ids = input_ids
            for _ in range(max_tokens):
                logits = self.model(current_ids)
                mx.eval(logits)
                next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                current_ids = mx.concatenate([current_ids, next_token], axis=1)
            mx.eval(current_ids)
            if self.tokenizer:
                return self.tokenizer.decode(current_ids[0].tolist())
            return str(current_ids[0].tolist())

    def _count_token_diff(self, text1: str, text2: str) -> int:
        """Count differing tokens between two texts."""
        if self.tokenizer:
            tokens1 = self.tokenizer.encode(text1)
            tokens2 = self.tokenizer.encode(text2)
        else:
            tokens1 = text1.split()
            tokens2 = text2.split()

        # Count differences
        diff = 0
        for i in range(max(len(tokens1), len(tokens2))):
            t1 = tokens1[i] if i < len(tokens1) else None
            t2 = tokens2[i] if i < len(tokens2) else None
            if t1 != t2:
                diff += 1
        return diff


# =============================================================================
# MoE-Aware Logit Lens
# =============================================================================


@dataclass
class ExpertContribution:
    """Contribution of a single expert to the prediction."""

    expert_idx: int
    routing_weight: float
    top_tokens: list[tuple[str, float]]  # (token, probability)
    contribution_norm: float


@dataclass
class MoELayerPrediction:
    """Prediction state at an MoE layer."""

    layer_idx: int
    top_token: str
    top_prob: float
    selected_experts: list[int]
    routing_weights: list[float]
    expert_contributions: list[ExpertContribution] | None = None


class MoELogitLens:
    """
    Logit lens with MoE awareness.

    Tracks how predictions evolve through layers while also
    capturing which experts contribute to each prediction.

    Example:
        >>> lens = MoELogitLens(model, tokenizer)
        >>> result = lens.analyze("The capital of France is")
        >>>
        >>> for layer_pred in result:
        ...     print(f"Layer {layer_pred.layer_idx}: {layer_pred.top_token}")
        ...     print(f"  Experts: {layer_pred.selected_experts}")
        ...     print(f"  Weights: {layer_pred.routing_weights}")
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initialize MoE logit lens.

        Args:
            model: The MoE model
            tokenizer: Tokenizer for encoding/decoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = MoEHooks(model)
        self.architecture = detect_moe_architecture(model)

    def analyze(
        self,
        prompt: str | mx.array,
        position: int = -1,
        top_k: int = 5,
    ) -> list[MoELayerPrediction]:
        """
        Analyze predictions at each MoE layer.

        Args:
            prompt: Input prompt or token IDs
            position: Which position to analyze (-1 for last)
            top_k: Number of top tokens to return

        Returns:
            List of MoELayerPrediction for each MoE layer
        """
        # Configure hooks
        self.hooks.configure(MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        ))

        # Get input IDs
        if isinstance(prompt, str):
            input_ids = mx.array([self.tokenizer.encode(prompt)])
        else:
            input_ids = prompt if prompt.ndim == 2 else prompt[None, :]

        # Run forward with capture
        logits = self.hooks.forward(input_ids)
        mx.eval(logits)

        # Get final prediction for reference
        final_probs = mx.softmax(logits[0, position], axis=-1)
        final_top_idx = int(mx.argmax(final_probs))
        final_top_token = self.tokenizer.decode([final_top_idx])
        final_top_prob = float(final_probs[final_top_idx])

        results = []

        # Analyze each MoE layer
        for layer_idx in self.hooks.state.captured_layers:
            # Get routing info
            routing_weights = self.hooks.state.router_weights.get(layer_idx)
            selected_experts = self.hooks.state.selected_experts.get(layer_idx)

            if routing_weights is None or selected_experts is None:
                continue

            # Extract for specified position
            if routing_weights.ndim == 3:
                pos_weights = routing_weights[0, position].tolist()
                pos_experts = selected_experts[0, position].tolist()
            else:
                pos_weights = routing_weights[position].tolist()
                pos_experts = selected_experts[position].tolist()

            # For MoE logit lens, we'd ideally project intermediate states
            # to vocabulary, but that requires layer-by-layer capture
            # For now, we report routing info with final prediction

            results.append(MoELayerPrediction(
                layer_idx=layer_idx,
                top_token=final_top_token,
                top_prob=final_top_prob,
                selected_experts=pos_experts,
                routing_weights=pos_weights,
            ))

        return results

    def track_expert_influence(
        self,
        prompt: str,
        target_token: str,
    ) -> dict[int, dict[int, float]]:
        """
        Track which experts influence prediction of a target token.

        Args:
            prompt: Input prompt
            target_token: Token to track

        Returns:
            Dict mapping layer_idx -> expert_idx -> influence score
        """
        # Get target token ID
        target_ids = self.tokenizer.encode(target_token)
        if len(target_ids) == 0:
            return {}
        target_id = target_ids[0]

        # Analyze with and without each expert
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        ablation = MoEAblation(self.model, self.tokenizer)

        influence = {}

        for layer_idx in self.hooks.moe_layer_indices:
            info = get_moe_layer_info(self.model, layer_idx)
            if info is None:
                continue

            influence[layer_idx] = {}

            # Get baseline probability
            logits = self.model(input_ids)
            mx.eval(logits)
            baseline_prob = float(mx.softmax(logits[0, -1], axis=-1)[target_id])

            # Test each expert
            for expert_idx in range(info.num_experts):
                result = ablation.ablate_expert(
                    input_ids,
                    layer_idx,
                    expert_idx,
                    max_tokens=1,
                )
                # Re-run to get probability
                # (This is a simplified version - full implementation
                # would capture probabilities during ablated generation)
                influence[layer_idx][expert_idx] = 0.0  # Placeholder

        return influence

    def print_analysis(self, prompt: str) -> None:
        """Print a formatted analysis."""
        results = self.analyze(prompt)

        print("\n" + "=" * 60)
        print("MoE Logit Lens Analysis")
        print("=" * 60)
        print(f"Prompt: {prompt[:50]}...")
        print()

        for pred in results:
            print(f"Layer {pred.layer_idx}:")
            print(f"  Top token: '{pred.top_token}' (p={pred.top_prob:.3f})")
            print(f"  Experts:   {pred.selected_experts}")
            print(f"  Weights:   {[f'{w:.3f}' for w in pred.routing_weights]}")
            print()

        print("=" * 60)


# =============================================================================
# Expert Specialization Analysis
# =============================================================================


def analyze_expert_specialization(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layer_idx: int,
) -> dict[int, dict[str, Any]]:
    """
    Analyze what each expert specializes in across prompts.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        prompts: List of prompts to analyze
        layer_idx: Layer to analyze

    Returns:
        Dict mapping expert_idx to specialization info
    """
    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(
        capture_router_logits=True,
        capture_selected_experts=True,
        layers=[layer_idx],
    ))

    info = get_moe_layer_info(model, layer_idx)
    if info is None:
        return {}

    # Collect tokens routed to each expert
    expert_tokens: dict[int, list[int]] = {i: [] for i in range(info.num_experts)}

    for prompt in prompts:
        input_ids = mx.array([tokenizer.encode(prompt)])
        hooks.forward(input_ids)

        if layer_idx not in hooks.state.selected_experts:
            continue

        experts = hooks.state.selected_experts[layer_idx]
        token_ids = input_ids[0].tolist()

        # Map tokens to experts
        if experts.ndim == 3:
            experts = experts[0]  # Remove batch dim

        for pos, token_id in enumerate(token_ids):
            if pos < experts.shape[0]:
                for expert_idx in experts[pos].tolist():
                    expert_tokens[expert_idx].append(token_id)

    # Analyze each expert
    results = {}
    for expert_idx, tokens in expert_tokens.items():
        if not tokens:
            results[expert_idx] = {
                "total_tokens": 0,
                "unique_tokens": 0,
                "top_tokens": [],
            }
            continue

        # Count token frequencies
        from collections import Counter
        counter = Counter(tokens)

        top_tokens = [
            (tokenizer.decode([tid]), count)
            for tid, count in counter.most_common(10)
        ]

        results[expert_idx] = {
            "total_tokens": len(tokens),
            "unique_tokens": len(counter),
            "top_tokens": top_tokens,
            "entropy": _compute_entropy(list(counter.values())),
        }

    return results


def _compute_entropy(counts: list[int]) -> float:
    """Compute entropy from counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    import math
    return -sum(p * math.log(p + 1e-10) for p in probs)


# =============================================================================
# Expert Identification System
# =============================================================================


class ExpertCategory(str, Enum):
    """Categories for expert specialization."""

    CODE = "code"
    """Code-related tokens (keywords, symbols, identifiers)."""

    MATH = "math"
    """Mathematical expressions and numbers."""

    LANGUAGE = "language"
    """Natural language tokens."""

    PUNCTUATION = "punctuation"
    """Punctuation and special characters."""

    WHITESPACE = "whitespace"
    """Whitespace and formatting tokens."""

    NAMES = "names"
    """Proper nouns, names, entities."""

    FUNCTION_WORDS = "function_words"
    """Articles, prepositions, conjunctions."""

    CONTENT_WORDS = "content_words"
    """Nouns, verbs, adjectives, adverbs."""

    UNKNOWN = "unknown"
    """Unknown or mixed specialization."""


@dataclass
class ExpertIdentity:
    """Identity profile for a single expert."""

    expert_idx: int
    layer_idx: int

    # Primary categorization
    primary_category: ExpertCategory
    category_confidence: float  # 0-1 how confident we are

    # Category breakdown
    category_scores: dict[str, float]  # category -> activation ratio

    # Token-level analysis
    total_activations: int
    unique_tokens: int
    top_tokens: list[tuple[str, int]]  # (token, count)
    token_entropy: float  # Higher = more diverse

    # Behavioral patterns
    positional_bias: str  # "early", "middle", "late", "uniform"
    context_sensitivity: float  # How much routing depends on context

    # Semantic clusters (if detected)
    semantic_clusters: list[str]  # e.g., ["python_keywords", "json_syntax"]

    def summary(self) -> str:
        """Return a short summary."""
        top_3 = ", ".join(f"'{t}'" for t, _ in self.top_tokens[:3])
        return (
            f"Expert {self.expert_idx} @ Layer {self.layer_idx}: "
            f"{self.primary_category.value} ({self.category_confidence:.0%}) "
            f"| Top: {top_3}"
        )

    def detailed_report(self) -> str:
        """Return a detailed report."""
        lines = [
            f"Expert {self.expert_idx} Identity Report (Layer {self.layer_idx})",
            "=" * 50,
            f"Primary Category: {self.primary_category.value} ({self.category_confidence:.1%} confidence)",
            "",
            "Category Breakdown:",
        ]

        for cat, score in sorted(
            self.category_scores.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {cat:<20} {bar} {score:.1%}")

        lines.extend([
            "",
            f"Activations: {self.total_activations} total, {self.unique_tokens} unique tokens",
            f"Token Entropy: {self.token_entropy:.2f} (higher = more diverse)",
            f"Positional Bias: {self.positional_bias}",
            f"Context Sensitivity: {self.context_sensitivity:.2f}",
            "",
            "Top Tokens:",
        ])

        for token, count in self.top_tokens[:10]:
            lines.append(f"  '{token}': {count}")

        if self.semantic_clusters:
            lines.extend([
                "",
                "Semantic Clusters:",
                *[f"  - {c}" for c in self.semantic_clusters],
            ])

        return "\n".join(lines)


@dataclass
class ExpertIdentificationResult:
    """Complete expert identification result for a model."""

    model_name: str
    layer_idx: int
    num_experts: int
    expert_identities: dict[int, ExpertIdentity]

    # Cross-expert analysis
    category_experts: dict[str, list[int]]  # category -> expert indices
    redundant_pairs: list[tuple[int, int, float]]  # (exp1, exp2, similarity)
    specialist_experts: list[int]  # Highly specialized (low entropy)
    generalist_experts: list[int]  # Diverse (high entropy)

    def summary(self) -> str:
        """Return a summary of all experts."""
        lines = [
            f"Expert Identification: {self.model_name}",
            f"Layer {self.layer_idx} ({self.num_experts} experts)",
            "=" * 60,
            "",
        ]

        # Group by category
        for category, experts in sorted(self.category_experts.items()):
            if experts:
                lines.append(f"{category.upper()}: Experts {experts}")

        lines.extend([
            "",
            f"Specialists (focused): {self.specialist_experts}",
            f"Generalists (diverse): {self.generalist_experts}",
        ])

        if self.redundant_pairs:
            lines.append("")
            lines.append("Redundant pairs (high similarity):")
            for e1, e2, sim in self.redundant_pairs[:5]:
                lines.append(f"  Experts {e1} & {e2}: {sim:.2%} similar")

        return "\n".join(lines)

    def print_all_identities(self) -> None:
        """Print detailed reports for all experts."""
        print(self.summary())
        print()
        for expert_idx in sorted(self.expert_identities.keys()):
            print(self.expert_identities[expert_idx].detailed_report())
            print()


class ExpertIdentifier:
    """
    Identifies what each expert in an MoE model specializes in.

    This class runs comprehensive analysis to determine:
    - What types of tokens each expert handles (code, math, language, etc.)
    - Semantic clusters within each expert's domain
    - Positional and contextual biases
    - Redundancy between experts

    Example:
        >>> identifier = ExpertIdentifier(model, tokenizer)
        >>> result = identifier.identify_all_experts(layer_idx=12)
        >>> print(result.summary())
        >>>
        >>> # Get specific expert identity
        >>> expert_5 = result.expert_identities[5]
        >>> print(expert_5.detailed_report())
    """

    # Token patterns for categorization
    CODE_KEYWORDS = {
        "def", "class", "import", "from", "return", "if", "else", "elif",
        "for", "while", "try", "except", "with", "as", "lambda", "yield",
        "async", "await", "raise", "pass", "break", "continue", "global",
        "nonlocal", "assert", "del", "in", "is", "not", "and", "or",
        "True", "False", "None", "self", "cls",
        # Common across languages
        "function", "var", "let", "const", "new", "this", "null", "undefined",
        "public", "private", "static", "void", "int", "string", "bool",
        "fn", "let", "mut", "impl", "struct", "enum", "trait", "pub", "mod",
    }

    CODE_SYMBOLS = {
        "=", "==", "!=", "+=", "-=", "*=", "/=", "//", "**", "->", "=>",
        "{", "}", "[", "]", "(", ")", "<", ">", "<=", ">=", "<<", ">>",
        "::", ".", ",", ";", ":", "|", "&", "^", "~", "@", "#", "$", "%",
        "++", "--", "&&", "||", "?", "!", "`", "'", '"',
    }

    MATH_PATTERNS = {
        "+", "-", "*", "/", "=", "<", ">", "^", "√", "∑", "∏", "∫", "∂",
        "sin", "cos", "tan", "log", "exp", "sqrt", "abs", "min", "max",
        "pi", "inf", "nan",
    }

    FUNCTION_WORDS = {
        "the", "a", "an", "of", "to", "in", "for", "on", "with", "at",
        "by", "from", "as", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "that", "which", "who", "whom", "whose", "this", "these", "those",
        "it", "its", "he", "she", "they", "we", "you", "i", "me", "him",
        "her", "us", "them", "my", "your", "his", "our", "their",
        "and", "but", "or", "nor", "so", "yet", "because", "although",
        "if", "when", "while", "where", "how", "what", "why",
    }

    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initialize expert identifier.

        Args:
            model: The MoE model
            tokenizer: Tokenizer for encoding/decoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = MoEHooks(model)
        self.architecture = detect_moe_architecture(model)

    def identify_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        test_prompts: list[str] | None = None,
        num_samples: int = 1000,
    ) -> ExpertIdentity:
        """
        Identify what a single expert specializes in.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            test_prompts: Optional custom test prompts
            num_samples: Number of tokens to sample for analysis

        Returns:
            ExpertIdentity with full analysis
        """
        if test_prompts is None:
            test_prompts = self._get_default_test_prompts()

        # Collect routing data
        expert_tokens, positional_data, context_data = self._collect_expert_data(
            layer_idx, expert_idx, test_prompts
        )

        # Categorize tokens
        category_scores = self._categorize_tokens(expert_tokens)

        # Determine primary category
        primary_category, confidence = self._determine_primary_category(category_scores)

        # Analyze token distribution
        from collections import Counter
        token_counter = Counter(expert_tokens)
        top_tokens = [
            (self._decode_token(tid), count)
            for tid, count in token_counter.most_common(20)
        ]
        token_entropy = _compute_entropy(list(token_counter.values()))

        # Analyze positional bias
        positional_bias = self._analyze_positional_bias(positional_data)

        # Analyze context sensitivity
        context_sensitivity = self._analyze_context_sensitivity(context_data)

        # Detect semantic clusters
        semantic_clusters = self._detect_semantic_clusters(expert_tokens, top_tokens)

        return ExpertIdentity(
            expert_idx=expert_idx,
            layer_idx=layer_idx,
            primary_category=primary_category,
            category_confidence=confidence,
            category_scores=category_scores,
            total_activations=len(expert_tokens),
            unique_tokens=len(token_counter),
            top_tokens=top_tokens,
            token_entropy=token_entropy,
            positional_bias=positional_bias,
            context_sensitivity=context_sensitivity,
            semantic_clusters=semantic_clusters,
        )

    def identify_all_experts(
        self,
        layer_idx: int,
        test_prompts: list[str] | None = None,
    ) -> ExpertIdentificationResult:
        """
        Identify all experts in a layer.

        Args:
            layer_idx: Layer index
            test_prompts: Optional custom test prompts

        Returns:
            ExpertIdentificationResult with all expert identities
        """
        info = get_moe_layer_info(self.model, layer_idx)
        if info is None:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")

        if test_prompts is None:
            test_prompts = self._get_default_test_prompts()

        # Collect all routing data at once (more efficient)
        all_expert_data = self._collect_all_expert_data(layer_idx, test_prompts)

        # Identify each expert
        expert_identities = {}
        for expert_idx in range(info.num_experts):
            expert_tokens = all_expert_data.get(expert_idx, {}).get("tokens", [])
            positional_data = all_expert_data.get(expert_idx, {}).get("positions", [])
            context_data = all_expert_data.get(expert_idx, {}).get("contexts", [])

            if not expert_tokens:
                # Expert never activated
                expert_identities[expert_idx] = ExpertIdentity(
                    expert_idx=expert_idx,
                    layer_idx=layer_idx,
                    primary_category=ExpertCategory.UNKNOWN,
                    category_confidence=0.0,
                    category_scores={},
                    total_activations=0,
                    unique_tokens=0,
                    top_tokens=[],
                    token_entropy=0.0,
                    positional_bias="none",
                    context_sensitivity=0.0,
                    semantic_clusters=[],
                )
                continue

            # Categorize
            category_scores = self._categorize_tokens(expert_tokens)
            primary_category, confidence = self._determine_primary_category(category_scores)

            from collections import Counter
            token_counter = Counter(expert_tokens)
            top_tokens = [
                (self._decode_token(tid), count)
                for tid, count in token_counter.most_common(20)
            ]

            expert_identities[expert_idx] = ExpertIdentity(
                expert_idx=expert_idx,
                layer_idx=layer_idx,
                primary_category=primary_category,
                category_confidence=confidence,
                category_scores=category_scores,
                total_activations=len(expert_tokens),
                unique_tokens=len(token_counter),
                top_tokens=top_tokens,
                token_entropy=_compute_entropy(list(token_counter.values())),
                positional_bias=self._analyze_positional_bias(positional_data),
                context_sensitivity=self._analyze_context_sensitivity(context_data),
                semantic_clusters=self._detect_semantic_clusters(expert_tokens, top_tokens),
            )

        # Cross-expert analysis
        category_experts = self._group_by_category(expert_identities)
        redundant_pairs = self._find_redundant_pairs(expert_identities)
        specialist_experts = self._find_specialists(expert_identities)
        generalist_experts = self._find_generalists(expert_identities)

        return ExpertIdentificationResult(
            model_name=self.architecture.value,
            layer_idx=layer_idx,
            num_experts=info.num_experts,
            expert_identities=expert_identities,
            category_experts=category_experts,
            redundant_pairs=redundant_pairs,
            specialist_experts=specialist_experts,
            generalist_experts=generalist_experts,
        )

    def _get_default_test_prompts(self) -> list[str]:
        """Get default prompts covering various domains."""
        return [
            # Code
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class DataProcessor:\n    def __init__(self, data):\n        self.data = data",
            "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
            "const express = require('express');\nconst app = express();\napp.listen(3000);",
            "SELECT * FROM users WHERE id = 1 AND status = 'active' ORDER BY created_at DESC;",
            '{"name": "John", "age": 30, "city": "New York", "active": true}',

            # Math
            "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a",
            "Calculate: 2 + 2 = 4, 10 * 5 = 50, 100 / 4 = 25",
            "The derivative of f(x) = x³ + 2x² - 5x + 3 is f'(x) = 3x² + 4x - 5",
            "∫(x² + 2x)dx = x³/3 + x² + C",

            # Natural language
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time, in a land far away, there lived a princess.",
            "The meeting will be held on Tuesday at 3 PM in the conference room.",
            "Please review the attached document and provide your feedback by Friday.",
            "The research paper discusses the implications of climate change on biodiversity.",

            # Names and entities
            "John Smith met with Dr. Sarah Johnson at Microsoft headquarters in Seattle.",
            "The Eiffel Tower in Paris, France was built by Gustave Eiffel in 1889.",
            "Apple Inc. announced the new iPhone at their headquarters in Cupertino, California.",

            # Mixed content
            "```python\ndef greet(name):\n    return f'Hello, {name}!'\n```\nThis function greets the user.",
            "Error: TypeError at line 42 - cannot read property 'undefined' of null",
            "TODO: Implement caching for API responses. See issue #123 for details.",
        ]

    def _collect_expert_data(
        self,
        layer_idx: int,
        expert_idx: int,
        prompts: list[str],
    ) -> tuple[list[int], list[int], list[tuple[int, list[int]]]]:
        """Collect routing data for a specific expert."""
        self.hooks.configure(MoECaptureConfig(
            capture_selected_experts=True,
            capture_router_weights=True,
            layers=[layer_idx],
        ))

        expert_tokens = []
        positional_data = []
        context_data = []

        for prompt in prompts:
            input_ids = mx.array([self.tokenizer.encode(prompt)])
            self.hooks.forward(input_ids)
            mx.eval(self.hooks.state.selected_experts)

            if layer_idx not in self.hooks.state.selected_experts:
                continue

            selected = self.hooks.state.selected_experts[layer_idx]
            if selected.ndim == 3:
                selected = selected[0]

            token_ids = input_ids[0].tolist()

            for pos, token_id in enumerate(token_ids):
                if pos < selected.shape[0]:
                    experts_at_pos = selected[pos].tolist()
                    if expert_idx in experts_at_pos:
                        expert_tokens.append(token_id)
                        positional_data.append(pos)
                        # Store context: (position, preceding tokens)
                        context = token_ids[max(0, pos-3):pos]
                        context_data.append((pos, context))

        return expert_tokens, positional_data, context_data

    def _collect_all_expert_data(
        self,
        layer_idx: int,
        prompts: list[str],
    ) -> dict[int, dict[str, list]]:
        """Collect routing data for all experts efficiently."""
        self.hooks.configure(MoECaptureConfig(
            capture_selected_experts=True,
            capture_router_weights=True,
            layers=[layer_idx],
        ))

        info = get_moe_layer_info(self.model, layer_idx)
        if info is None:
            return {}

        all_data: dict[int, dict[str, list]] = {
            i: {"tokens": [], "positions": [], "contexts": []}
            for i in range(info.num_experts)
        }

        for prompt in prompts:
            input_ids = mx.array([self.tokenizer.encode(prompt)])
            self.hooks.forward(input_ids)
            mx.eval(self.hooks.state.selected_experts)

            if layer_idx not in self.hooks.state.selected_experts:
                continue

            selected = self.hooks.state.selected_experts[layer_idx]
            if selected.ndim == 3:
                selected = selected[0]

            token_ids = input_ids[0].tolist()

            for pos, token_id in enumerate(token_ids):
                if pos < selected.shape[0]:
                    experts_at_pos = selected[pos].tolist()
                    context = token_ids[max(0, pos-3):pos]

                    for expert_idx in experts_at_pos:
                        if isinstance(expert_idx, (int, float)):
                            expert_idx = int(expert_idx)
                            all_data[expert_idx]["tokens"].append(token_id)
                            all_data[expert_idx]["positions"].append(pos)
                            all_data[expert_idx]["contexts"].append((pos, context))

        return all_data

    def _categorize_tokens(self, token_ids: list[int]) -> dict[str, float]:
        """Categorize tokens and return category scores."""
        if not token_ids:
            return {}

        categories = {
            "code": 0,
            "math": 0,
            "punctuation": 0,
            "whitespace": 0,
            "function_words": 0,
            "content_words": 0,
            "names": 0,
            "numbers": 0,
        }

        for tid in token_ids:
            token = self._decode_token(tid).strip()
            token_lower = token.lower()

            # Check categories
            if token in self.CODE_KEYWORDS or token_lower in self.CODE_KEYWORDS:
                categories["code"] += 1
            elif token in self.CODE_SYMBOLS:
                categories["code"] += 0.5
                categories["punctuation"] += 0.5
            elif token in self.MATH_PATTERNS or self._is_number(token):
                categories["math"] += 1
            elif token_lower in self.FUNCTION_WORDS:
                categories["function_words"] += 1
            elif len(token) == 1 and not token.isalnum():
                categories["punctuation"] += 1
            elif token.isspace() or token in {"\n", "\t", "\\n", "\\t"}:
                categories["whitespace"] += 1
            elif token and token[0].isupper() and len(token) > 1:
                categories["names"] += 1
            elif token.isalpha():
                categories["content_words"] += 1

        # Normalize
        total = sum(categories.values())
        if total > 0:
            categories = {k: v / total for k, v in categories.items()}

        return categories

    def _determine_primary_category(
        self, scores: dict[str, float]
    ) -> tuple[ExpertCategory, float]:
        """Determine primary category from scores."""
        if not scores:
            return ExpertCategory.UNKNOWN, 0.0

        # Map score keys to ExpertCategory
        category_map = {
            "code": ExpertCategory.CODE,
            "math": ExpertCategory.MATH,
            "punctuation": ExpertCategory.PUNCTUATION,
            "whitespace": ExpertCategory.WHITESPACE,
            "function_words": ExpertCategory.FUNCTION_WORDS,
            "content_words": ExpertCategory.CONTENT_WORDS,
            "names": ExpertCategory.NAMES,
            "numbers": ExpertCategory.MATH,
        }

        # Find highest score
        top_key = max(scores.keys(), key=lambda k: scores[k])
        top_score = scores[top_key]

        # Check if it's dominant enough
        if top_score < 0.2:
            return ExpertCategory.UNKNOWN, top_score

        return category_map.get(top_key, ExpertCategory.UNKNOWN), top_score

    def _analyze_positional_bias(self, positions: list[int]) -> str:
        """Analyze if expert has positional bias."""
        if not positions:
            return "none"

        # Get distribution
        avg_pos = sum(positions) / len(positions)
        max_pos = max(positions) if positions else 1

        if max_pos == 0:
            return "uniform"

        # Normalize
        norm_avg = avg_pos / max_pos

        if norm_avg < 0.3:
            return "early"
        elif norm_avg > 0.7:
            return "late"
        elif 0.4 <= norm_avg <= 0.6:
            return "middle"
        else:
            return "uniform"

    def _analyze_context_sensitivity(
        self, context_data: list[tuple[int, list[int]]]
    ) -> float:
        """Analyze how much routing depends on context."""
        if len(context_data) < 10:
            return 0.0

        # Group by token (ignoring context)
        from collections import defaultdict
        token_contexts: dict[int, list[tuple[int, ...]]] = defaultdict(list)

        for pos, context in context_data:
            # We need the token at this position - approximate from context
            if context:
                token_contexts[tuple(context)].append((pos,))

        # If same token appears in very different contexts, high sensitivity
        # This is a simplified measure
        unique_contexts = len(token_contexts)
        if unique_contexts == 0:
            return 0.0

        # Normalize by total data points
        return min(1.0, unique_contexts / len(context_data))

    def _detect_semantic_clusters(
        self,
        token_ids: list[int],
        top_tokens: list[tuple[str, int]],
    ) -> list[str]:
        """Detect semantic clusters in expert's tokens."""
        clusters = []

        # Check for specific patterns
        top_token_strs = {t.lower() for t, _ in top_tokens[:30]}

        # Python-specific
        python_keywords = {"def", "class", "import", "return", "self", "none", "true", "false"}
        if len(top_token_strs & python_keywords) >= 3:
            clusters.append("python_keywords")

        # JavaScript-specific
        js_keywords = {"function", "const", "let", "var", "this", "null", "undefined"}
        if len(top_token_strs & js_keywords) >= 3:
            clusters.append("javascript_keywords")

        # JSON/data structures
        json_markers = {"{", "}", "[", "]", ":", ",", '"', "'"}
        if len(top_token_strs & json_markers) >= 4:
            clusters.append("json_syntax")

        # SQL
        sql_keywords = {"select", "from", "where", "and", "or", "order", "by", "join"}
        if len(top_token_strs & sql_keywords) >= 3:
            clusters.append("sql_keywords")

        # Mathematical
        math_ops = {"+", "-", "*", "/", "=", "<", ">", "^", "(", ")"}
        if len(top_token_strs & math_ops) >= 4:
            clusters.append("math_operators")

        # Numeric
        numeric_count = sum(1 for t, _ in top_tokens[:20] if self._is_number(t))
        if numeric_count >= 5:
            clusters.append("numeric_values")

        # Punctuation heavy
        punct_count = sum(1 for t, _ in top_tokens[:20] if len(t) == 1 and not t.isalnum())
        if punct_count >= 8:
            clusters.append("punctuation_heavy")

        # Whitespace/formatting
        ws_tokens = {"\n", "\t", " ", "\\n", "\\t", "  "}
        if len(top_token_strs & ws_tokens) >= 2:
            clusters.append("whitespace_formatting")

        return clusters

    def _group_by_category(
        self, identities: dict[int, ExpertIdentity]
    ) -> dict[str, list[int]]:
        """Group experts by their primary category."""
        groups: dict[str, list[int]] = {}

        for expert_idx, identity in identities.items():
            cat = identity.primary_category.value
            if cat not in groups:
                groups[cat] = []
            if identity.category_confidence >= 0.2:  # Only if reasonably confident
                groups[cat].append(expert_idx)

        return groups

    def _find_redundant_pairs(
        self, identities: dict[int, ExpertIdentity]
    ) -> list[tuple[int, int, float]]:
        """Find pairs of experts with high overlap."""
        pairs = []

        expert_list = list(identities.keys())
        for i, exp1 in enumerate(expert_list):
            for exp2 in expert_list[i+1:]:
                similarity = self._compute_expert_similarity(
                    identities[exp1], identities[exp2]
                )
                if similarity > 0.7:  # High similarity threshold
                    pairs.append((exp1, exp2, similarity))

        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def _compute_expert_similarity(
        self, id1: ExpertIdentity, id2: ExpertIdentity
    ) -> float:
        """Compute similarity between two expert identities."""
        # Compare category scores
        score_sim = 0.0
        all_cats = set(id1.category_scores.keys()) | set(id2.category_scores.keys())
        if all_cats:
            for cat in all_cats:
                s1 = id1.category_scores.get(cat, 0)
                s2 = id2.category_scores.get(cat, 0)
                score_sim += min(s1, s2)

        # Compare top tokens
        top1 = {t for t, _ in id1.top_tokens[:10]}
        top2 = {t for t, _ in id2.top_tokens[:10]}
        if top1 or top2:
            token_sim = len(top1 & top2) / len(top1 | top2)
        else:
            token_sim = 0

        return 0.5 * score_sim + 0.5 * token_sim

    def _find_specialists(
        self, identities: dict[int, ExpertIdentity]
    ) -> list[int]:
        """Find highly specialized experts (low token entropy)."""
        specialists = []

        # Get entropy distribution
        entropies = [id.token_entropy for id in identities.values() if id.total_activations > 0]
        if not entropies:
            return []

        threshold = sum(entropies) / len(entropies) * 0.5  # Below average

        for expert_idx, identity in identities.items():
            if identity.token_entropy < threshold and identity.total_activations > 10:
                specialists.append(expert_idx)

        return specialists

    def _find_generalists(
        self, identities: dict[int, ExpertIdentity]
    ) -> list[int]:
        """Find generalist experts (high token entropy)."""
        generalists = []

        entropies = [id.token_entropy for id in identities.values() if id.total_activations > 0]
        if not entropies:
            return []

        threshold = sum(entropies) / len(entropies) * 1.5  # Above average

        for expert_idx, identity in identities.items():
            if identity.token_entropy > threshold and identity.total_activations > 10:
                generalists.append(expert_idx)

        return generalists

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID."""
        try:
            return self.tokenizer.decode([token_id])
        except Exception:
            return f"<{token_id}>"

    def _is_number(self, s: str) -> bool:
        """Check if string is a number."""
        try:
            float(s.replace(",", ""))
            return True
        except (ValueError, AttributeError):
            return False


def identify_experts(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    test_prompts: list[str] | None = None,
) -> ExpertIdentificationResult:
    """
    Convenience function to identify all experts in a layer.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        layer_idx: Layer to analyze
        test_prompts: Optional custom test prompts

    Returns:
        ExpertIdentificationResult with all expert identities

    Example:
        >>> result = identify_experts(model, tokenizer, layer_idx=12)
        >>> print(result.summary())
        >>> for exp_id, identity in result.expert_identities.items():
        ...     print(identity.summary())
    """
    identifier = ExpertIdentifier(model, tokenizer)
    return identifier.identify_all_experts(layer_idx, test_prompts)


def print_expert_identities(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    test_prompts: list[str] | None = None,
) -> None:
    """
    Print expert identification results.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        layer_idx: Layer to analyze
        test_prompts: Optional custom test prompts
    """
    result = identify_experts(model, tokenizer, layer_idx, test_prompts)
    result.print_all_identities()


# =============================================================================
# Expert Compression
# =============================================================================


@dataclass
class ExpertMergeResult:
    """Result of merging experts."""

    source_experts: list[int]
    target_expert: int
    similarity: float
    weight_blend: str  # "average", "weighted", "dominant"


@dataclass
class CompressionPlan:
    """Plan for compressing MoE experts."""

    original_num_experts: int
    target_num_experts: int
    merges: list[ExpertMergeResult]
    pruned_experts: list[int]
    kept_experts: list[int]
    estimated_memory_reduction: float  # 0-1
    estimated_quality_impact: str  # "minimal", "moderate", "significant"

    def summary(self) -> str:
        """Return compression plan summary."""
        lines = [
            f"Compression Plan: {self.original_num_experts} → {self.target_num_experts} experts",
            f"Memory reduction: ~{self.estimated_memory_reduction:.0%}",
            f"Quality impact: {self.estimated_quality_impact}",
            "",
            f"Kept experts ({len(self.kept_experts)}): {self.kept_experts}",
            f"Pruned experts ({len(self.pruned_experts)}): {self.pruned_experts}",
        ]

        if self.merges:
            lines.append("")
            lines.append(f"Merges ({len(self.merges)}):")
            for merge in self.merges:
                lines.append(
                    f"  {merge.source_experts} → Expert {merge.target_expert} "
                    f"(sim={merge.similarity:.1%}, {merge.weight_blend})"
                )

        return "\n".join(lines)


@dataclass
class CompressedMoEConfig:
    """Configuration for a compressed MoE layer."""

    layer_idx: int
    original_num_experts: int
    compressed_num_experts: int

    # Mapping from old expert indices to new
    expert_mapping: dict[int, int]  # old_idx -> new_idx (-1 = pruned)

    # For merged experts, which originals were combined
    merged_from: dict[int, list[int]]  # new_idx -> [old_idx, ...]

    # Router weight adjustments
    router_remap: mx.array | None = None  # Remapped router weights


class ExpertCompressor:
    """
    Compress MoE models by merging or pruning redundant experts.

    Uses expert identification to find:
    1. Highly similar experts (merge candidates)
    2. Low-utilization experts (prune candidates)
    3. Essential specialists (must keep)

    Compression strategies:
    - **merge**: Combine similar experts by averaging weights
    - **prune**: Remove low-utilization experts, reroute to similar ones
    - **reduce_k**: Reduce number of active experts per token

    Example:
        >>> compressor = ExpertCompressor(model, tokenizer)
        >>> plan = compressor.plan_compression(
        ...     layer_idx=12,
        ...     target_experts=16,  # 32 → 16
        ...     strategy="merge",
        ... )
        >>> print(plan.summary())
        >>>
        >>> # Apply compression
        >>> compressed_model = compressor.apply_compression(plan)
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initialize compressor.

        Args:
            model: The MoE model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.identifier = ExpertIdentifier(model, tokenizer)

    def analyze_compression_potential(
        self,
        layer_idx: int,
        test_prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze how much a layer can be compressed.

        Args:
            layer_idx: Layer to analyze
            test_prompts: Optional test prompts

        Returns:
            Analysis dict with compression recommendations
        """
        # Get expert identities
        result = self.identifier.identify_all_experts(layer_idx, test_prompts)

        # Compute pairwise similarities
        similarities = []
        experts = list(result.expert_identities.keys())
        for i, e1 in enumerate(experts):
            for e2 in experts[i + 1:]:
                sim = self.identifier._compute_expert_similarity(
                    result.expert_identities[e1],
                    result.expert_identities[e2],
                )
                similarities.append((e1, e2, sim))

        # Find merge candidates (similarity > 0.6)
        merge_candidates = [(e1, e2, s) for e1, e2, s in similarities if s > 0.6]
        merge_candidates.sort(key=lambda x: x[2], reverse=True)

        # Find prune candidates (low activation, not specialist)
        activations = {
            e: id.total_activations
            for e, id in result.expert_identities.items()
        }
        avg_activation = sum(activations.values()) / len(activations) if activations else 0

        prune_candidates = [
            e for e, act in activations.items()
            if act < avg_activation * 0.3  # Less than 30% of average
            and e not in result.specialist_experts
        ]

        # Estimate compression potential
        mergeable_groups = self._find_merge_groups(merge_candidates)
        potential_reduction = len(prune_candidates) + sum(
            len(g) - 1 for g in mergeable_groups
        )

        return {
            "layer_idx": layer_idx,
            "num_experts": result.num_experts,
            "merge_candidates": merge_candidates[:10],
            "prune_candidates": prune_candidates,
            "specialist_experts": result.specialist_experts,
            "generalist_experts": result.generalist_experts,
            "mergeable_groups": mergeable_groups,
            "potential_reduction": potential_reduction,
            "max_compression_ratio": 1 - (potential_reduction / result.num_experts),
            "recommended_target": max(
                result.num_experts - potential_reduction,
                result.num_experts // 2,  # At most 50% reduction
                len(result.specialist_experts) + 2,  # Keep specialists + buffer
            ),
        }

    def _find_merge_groups(
        self,
        similarities: list[tuple[int, int, float]],
        threshold: float = 0.6,
    ) -> list[list[int]]:
        """Find groups of similar experts that can be merged."""
        # Union-find to group similar experts
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for e1, e2, sim in similarities:
            if sim >= threshold:
                union(e1, e2)

        # Group by root
        groups: dict[int, list[int]] = {}
        for e in parent:
            root = find(e)
            if root not in groups:
                groups[root] = []
            groups[root].append(e)

        # Only return groups with 2+ members
        return [sorted(g) for g in groups.values() if len(g) >= 2]

    def plan_compression(
        self,
        layer_idx: int,
        target_experts: int | None = None,
        strategy: str = "balanced",
        similarity_threshold: float = 0.6,
        test_prompts: list[str] | None = None,
    ) -> CompressionPlan:
        """
        Create a compression plan for an MoE layer.

        Args:
            layer_idx: Layer to compress
            target_experts: Target number of experts (None = auto)
            strategy: "merge", "prune", or "balanced"
            similarity_threshold: Threshold for merging (0.5-0.8)
            test_prompts: Optional test prompts

        Returns:
            CompressionPlan
        """
        analysis = self.analyze_compression_potential(layer_idx, test_prompts)

        if target_experts is None:
            target_experts = analysis["recommended_target"]

        original = analysis["num_experts"]
        reduction_needed = original - target_experts

        if reduction_needed <= 0:
            return CompressionPlan(
                original_num_experts=original,
                target_num_experts=original,
                merges=[],
                pruned_experts=[],
                kept_experts=list(range(original)),
                estimated_memory_reduction=0.0,
                estimated_quality_impact="none",
            )

        merges = []
        pruned = []
        kept = set(range(original))

        # Protect specialists
        protected = set(analysis["specialist_experts"])

        if strategy in ("prune", "balanced"):
            # Prune low-utilization non-specialists first
            for e in analysis["prune_candidates"]:
                if len(pruned) >= reduction_needed:
                    break
                if e not in protected:
                    pruned.append(e)
                    kept.discard(e)

        if strategy in ("merge", "balanced") and len(pruned) < reduction_needed:
            # Merge similar experts
            for group in analysis["mergeable_groups"]:
                if len(pruned) + len(merges) >= reduction_needed:
                    break

                # Keep the most-used expert, merge others into it
                group_activations = [
                    (e, self.identifier.identify_expert(layer_idx, e).total_activations)
                    for e in group if e not in protected and e in kept
                ]

                if len(group_activations) < 2:
                    continue

                group_activations.sort(key=lambda x: x[1], reverse=True)
                target = group_activations[0][0]
                sources = [e for e, _ in group_activations[1:]]

                merges.append(ExpertMergeResult(
                    source_experts=sources,
                    target_expert=target,
                    similarity=0.7,  # Approximate
                    weight_blend="weighted",
                ))

                for s in sources:
                    kept.discard(s)

        # Estimate impact
        memory_reduction = 1 - (len(kept) / original)

        if memory_reduction < 0.2:
            quality_impact = "minimal"
        elif memory_reduction < 0.4:
            quality_impact = "moderate"
        else:
            quality_impact = "significant"

        return CompressionPlan(
            original_num_experts=original,
            target_num_experts=len(kept),
            merges=merges,
            pruned_experts=pruned,
            kept_experts=sorted(kept),
            estimated_memory_reduction=memory_reduction,
            estimated_quality_impact=quality_impact,
        )

    def apply_compression(
        self,
        plan: CompressionPlan,
        layer_idx: int,
        inplace: bool = False,
    ) -> CompressedMoEConfig:
        """
        Apply a compression plan to create compressed layer config.

        Note: This creates the configuration for compression. Actual weight
        manipulation requires model-specific handling.

        Args:
            plan: Compression plan to apply
            layer_idx: Layer index
            inplace: Whether to modify model in place (not recommended)

        Returns:
            CompressedMoEConfig with mapping information
        """
        # Build expert mapping
        expert_mapping = {}
        new_idx = 0

        for old_idx in range(plan.original_num_experts):
            if old_idx in plan.kept_experts:
                expert_mapping[old_idx] = new_idx
                new_idx += 1
            else:
                expert_mapping[old_idx] = -1  # Pruned

        # Track merges
        merged_from: dict[int, list[int]] = {}
        for merge in plan.merges:
            target_new_idx = expert_mapping[merge.target_expert]
            merged_from[target_new_idx] = [merge.target_expert] + merge.source_experts

        return CompressedMoEConfig(
            layer_idx=layer_idx,
            original_num_experts=plan.original_num_experts,
            compressed_num_experts=plan.target_num_experts,
            expert_mapping=expert_mapping,
            merged_from=merged_from,
        )

    def create_compressed_router(
        self,
        plan: CompressionPlan,
        layer_idx: int,
    ) -> mx.array | None:
        """
        Create router weights for compressed model.

        For pruned/merged experts, redistributes routing to remaining experts.

        Args:
            plan: Compression plan
            layer_idx: Layer index

        Returns:
            New router weight matrix or None if not applicable
        """
        layers = _get_layers(self.model)
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "router"):
            return None

        router = layer.mlp.router
        if not hasattr(router, "weight"):
            return None

        # Original router: [num_experts, hidden_size]
        original_weights = router.weight

        # Create new router with only kept experts
        kept_indices = mx.array(plan.kept_experts)
        new_weights = mx.take(original_weights, kept_indices, axis=0)

        # For merged experts, average their router rows
        for merge in plan.merges:
            target_new_idx = plan.kept_experts.index(merge.target_expert)
            source_indices = [merge.target_expert] + merge.source_experts

            # Average router weights
            merged_weight = mx.mean(
                mx.take(original_weights, mx.array(source_indices), axis=0),
                axis=0,
                keepdims=True,
            )
            new_weights = mx.concatenate([
                new_weights[:target_new_idx],
                merged_weight,
                new_weights[target_new_idx + 1:],
            ], axis=0)

        return new_weights


def plan_expert_compression(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    target_experts: int | None = None,
    strategy: str = "balanced",
) -> CompressionPlan:
    """
    Convenience function to plan expert compression.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        layer_idx: Layer to compress
        target_experts: Target number of experts
        strategy: "merge", "prune", or "balanced"

    Returns:
        CompressionPlan

    Example:
        >>> plan = plan_expert_compression(model, tokenizer, layer_idx=12, target_experts=16)
        >>> print(plan.summary())
        >>> # Compression Plan: 32 → 16 experts
        >>> # Memory reduction: ~50%
        >>> # Quality impact: moderate
    """
    compressor = ExpertCompressor(model, tokenizer)
    return compressor.plan_compression(layer_idx, target_experts, strategy)


def analyze_compression(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
) -> dict[str, Any]:
    """
    Analyze compression potential for an MoE layer.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        layer_idx: Layer to analyze

    Returns:
        Analysis dict with recommendations

    Example:
        >>> analysis = analyze_compression(model, tokenizer, layer_idx=12)
        >>> print(f"Can reduce to {analysis['recommended_target']} experts")
        >>> print(f"Mergeable groups: {analysis['mergeable_groups']}")
    """
    compressor = ExpertCompressor(model, tokenizer)
    return compressor.analyze_compression_potential(layer_idx)
