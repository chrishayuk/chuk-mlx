"""
Hook infrastructure for capturing intermediate model states.

MLX doesn't have PyTorch's register_forward_hook, so we use a wrapper
approach that intercepts layer outputs during forward pass.

Example:
    >>> from chuk_lazarus.introspection import ModelHooks, CaptureConfig, LayerSelection
    >>>
    >>> hooks = ModelHooks(model)
    >>> hooks.configure(CaptureConfig(
    ...     layers=[0, 4, 8, 12],
    ...     capture_attention_weights=True,
    ... ))
    >>>
    >>> # Run inference
    >>> output = hooks.forward(input_ids)
    >>>
    >>> # Access captured states
    >>> layer_4_hidden = hooks.state.hidden_states[4]
    >>> layer_4_attn = hooks.state.attention_weights[4]
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field


class LayerSelection(str, Enum):
    """Selection strategy for which layers to capture."""

    ALL = "all"
    """Capture all layers."""


class PositionSelection(str, Enum):
    """Selection strategy for which sequence positions to capture."""

    ALL = "all"
    """Capture all positions."""

    LAST = "last"
    """Capture only the last position (memory efficient)."""


class CaptureConfig(BaseModel):
    """Configuration for what to capture during forward pass."""

    layers: list[int] | LayerSelection = Field(
        default=LayerSelection.ALL,
        description="Which layers to capture. Use LayerSelection.ALL for all layers or a list of indices.",
    )

    capture_hidden_states: bool = Field(
        default=True,
        description="Capture hidden states after each layer.",
    )

    capture_attention_weights: bool = Field(
        default=False,
        description="Capture attention weight matrices. Memory intensive.",
    )

    capture_attention_output: bool = Field(
        default=False,
        description="Capture attention output before residual add.",
    )

    capture_ffn_output: bool = Field(
        default=False,
        description="Capture FFN output before residual add.",
    )

    capture_pre_norm: bool = Field(
        default=False,
        description="Capture states before layer norm (raw residual stream).",
    )

    positions: list[int] | PositionSelection = Field(
        default=PositionSelection.LAST,
        description="Which sequence positions to capture.",
    )

    detach: bool = Field(
        default=True,
        description="Whether to detach captured tensors from computation graph.",
    )

    model_config = ConfigDict(use_enum_values=False)


class CapturedState(BaseModel):
    """Container for captured intermediate states."""

    hidden_states: dict[int, Any] = Field(
        default_factory=dict,
        description="Hidden states after each layer. Shape: [batch, seq, hidden]",
    )

    attention_weights: dict[int, Any] = Field(
        default_factory=dict,
        description="Attention weights per layer. Shape: [batch, heads, seq, seq]",
    )

    attention_outputs: dict[int, Any] = Field(
        default_factory=dict,
        description="Attention outputs before residual. Shape: [batch, seq, hidden]",
    )

    ffn_outputs: dict[int, Any] = Field(
        default_factory=dict,
        description="FFN outputs before residual. Shape: [batch, seq, hidden]",
    )

    pre_norm_states: dict[int, Any] = Field(
        default_factory=dict,
        description="States before layer norm. Shape: [batch, seq, hidden]",
    )

    input_ids: Any | None = Field(
        default=None,
        description="Original input token IDs.",
    )

    embeddings: Any | None = Field(
        default=None,
        description="Token embeddings before first layer.",
    )

    final_hidden: Any | None = Field(
        default=None,
        description="Final hidden state after all layers.",
    )

    logits: Any | None = Field(
        default=None,
        description="Output logits if computed.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def clear(self) -> None:
        """Clear all captured states to free memory."""
        self.hidden_states.clear()
        self.attention_weights.clear()
        self.attention_outputs.clear()
        self.ffn_outputs.clear()
        self.pre_norm_states.clear()
        self.input_ids = None
        self.embeddings = None
        self.final_hidden = None
        self.logits = None

    @property
    def num_layers_captured(self) -> int:
        """Number of layers with captured hidden states."""
        return len(self.hidden_states)

    @property
    def captured_layers(self) -> list[int]:
        """Sorted list of layer indices that were captured."""
        return sorted(self.hidden_states.keys())

    def get_hidden_at_position(self, layer_idx: int, position: int = -1) -> mx.array | None:
        """
        Get hidden state for a specific layer and sequence position.

        Args:
            layer_idx: Layer index
            position: Sequence position (-1 for last)

        Returns:
            Hidden state vector of shape [hidden_size] or None if not captured
        """
        if layer_idx not in self.hidden_states:
            return None
        hidden = self.hidden_states[layer_idx]
        # Handle both [seq, hidden] and [batch, seq, hidden]
        if hidden.ndim == 2:
            return hidden[position]
        return hidden[0, position]


class ModelHooks:
    """
    Hook manager for capturing intermediate model states.

    This class wraps a model's forward pass to capture intermediate
    activations, attention weights, and other internal states without
    modifying the model code.

    Example:
        >>> hooks = ModelHooks(model)
        >>> hooks.configure(CaptureConfig(
        ...     layers=[0, 4, 8],
        ...     capture_attention_weights=True,
        ... ))
        >>>
        >>> # Wrapped forward pass captures states
        >>> logits = hooks.forward(input_ids)
        >>>
        >>> # Access captured data
        >>> print(hooks.state.hidden_states[4].shape)
        >>> print(hooks.state.attention_weights[4].shape)
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_scale: float | None = None,
        model_config: Any | None = None,
    ):
        """
        Initialize hooks for a model.

        Args:
            model: The model to hook into. Should have a .model.layers attribute.
            embedding_scale: Optional scale factor for embeddings.
                Some models (e.g., Gemma) scale embeddings by sqrt(hidden_size).
                If not provided, will try to detect from model_config or model properties.
            model_config: Optional model config (e.g., GemmaConfig, LlamaConfig).
                Used to auto-detect embedding_scale and other model properties.
        """
        self.model = model
        self.model_config = model_config
        self.config = CaptureConfig()
        self.state = CapturedState()
        self._original_forwards: dict[int, Callable] = {}
        self._hooked = False
        self._embedding_scale_override = embedding_scale

    def configure(self, config: CaptureConfig) -> ModelHooks:
        """
        Configure what to capture.

        Args:
            config: Capture configuration

        Returns:
            Self for chaining
        """
        self.config = config
        return self

    def capture_layers(
        self,
        layers: list[int] | LayerSelection = LayerSelection.ALL,
        capture_attention: bool = False,
    ) -> ModelHooks:
        """
        Convenience method to configure layer capture.

        Args:
            layers: Which layers to capture, or LayerSelection.ALL
            capture_attention: Whether to capture attention weights

        Returns:
            Self for chaining
        """
        self.config.layers = layers
        self.config.capture_attention_weights = capture_attention
        return self

    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if a layer should be captured based on config."""
        if self.config.layers == LayerSelection.ALL:
            return True
        if isinstance(self.config.layers, list):
            return layer_idx in self.config.layers
        return False

    def _maybe_slice_positions(self, tensor: mx.array) -> mx.array:
        """Slice tensor to only keep configured positions."""
        if self.config.positions == PositionSelection.ALL:
            return tensor
        if self.config.positions == PositionSelection.LAST:
            # Keep only last position
            if tensor.ndim == 2:  # [seq, hidden]
                return tensor[-1:]
            elif tensor.ndim == 3:  # [batch, seq, hidden]
                return tensor[:, -1:]
            elif tensor.ndim == 4:  # [batch, heads, seq, seq] for attention
                return tensor[:, :, -1:, :]
        # Explicit position list
        if isinstance(self.config.positions, list):
            positions = self.config.positions
            if tensor.ndim == 2:
                return tensor[positions]
            elif tensor.ndim == 3:
                return tensor[:, positions, :]
            elif tensor.ndim == 4:
                return tensor[:, :, positions, :]
        return tensor

    def _get_layers(self) -> list[nn.Module]:
        """Get the list of transformer layers from the model."""
        # Try common attribute names
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return list(self.model.transformer.h)
            if hasattr(self.model.transformer, "layers"):
                return list(self.model.transformer.layers)
        raise ValueError(
            "Cannot find layers in model. Expected model.model.layers, "
            "model.layers, or model.transformer.h"
        )

    def _get_embed_tokens(self) -> nn.Module | None:
        """Get the embedding layer from the model."""
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                return self.model.transformer.wte
        return None

    def _get_final_norm(self) -> nn.Module | None:
        """Get the final layer norm from the model."""
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "norm"):
                return inner.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self) -> Callable[[mx.array], mx.array] | None:
        """Get the LM head or tied embedding projection from the model."""
        # Check for explicit lm_head first
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            return self.model.lm_head

        # Check for tied embeddings (explicit flag)
        if hasattr(self.model, "tie_word_embeddings") and self.model.tie_word_embeddings:
            embed = self._get_embed_tokens()
            if embed is not None and hasattr(embed, "as_linear"):
                return embed.as_linear

        # Fallback: if no lm_head, try to use embedding as_linear (common in mlx-lm models)
        # This handles models that use tied embeddings without setting the flag
        embed = self._get_embed_tokens()
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear

        return None

    def _get_embedding_scale(self) -> float | None:
        """
        Get embedding scale factor if model uses one.

        Checks in order:
        1. Explicit override provided at init
        2. model_config.embedding_scale (from model families)
        3. Model backbone embedding_scale property
        4. Attached _embedding_scale_for_hooks (for external models)

        Note: External models (e.g., from mlx_lm) may not expose embedding_scale,
        which can cause incorrect logit lens results for models like Gemma.
        Use the model_config or embedding_scale parameter when creating ModelHooks.
        """
        # Use override if provided
        if self._embedding_scale_override is not None:
            return self._embedding_scale_override

        # Check model_config (from model families registry)
        if self.model_config is not None and hasattr(self.model_config, "embedding_scale"):
            scale = self.model_config.embedding_scale
            if scale is not None:
                return scale

        # Check for property on the model backbone (our native models)
        inner = getattr(self.model, "model", self.model)
        if hasattr(inner, "embedding_scale"):
            return inner.embedding_scale

        # Check for attached scale (set by analyzer for quantized models)
        if hasattr(self.model, "_embedding_scale_for_hooks"):
            return self.model._embedding_scale_for_hooks

        return None

    def forward(
        self,
        input_ids: mx.array,
        cache: Any | None = None,
        return_logits: bool = True,
    ) -> mx.array | None:
        """
        Run forward pass while capturing intermediate states.

        This manually unrolls the forward pass to capture states at each layer.

        Args:
            input_ids: Input token IDs [batch, seq] or [seq]
            cache: Optional KV cache
            return_logits: Whether to compute and return logits

        Returns:
            Logits if return_logits=True, else None
        """
        self.state.clear()

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        self.state.input_ids = input_ids

        # Get model components
        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        # Embeddings
        if embed is not None:
            # Call the embedding layer - works for both nn.Embedding and wrapped embeddings
            h = embed(input_ids)
            # Apply embedding scale if model uses one (e.g., Gemma)
            embed_scale = self._get_embedding_scale()
            if embed_scale is not None:
                h = h * embed_scale
            self.state.embeddings = h
        else:
            raise ValueError("Cannot find embedding layer")

        # Create causal mask (critical for correct attention)
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Process each layer
        for layer_idx, layer in enumerate(layers):
            should_capture = self._should_capture_layer(layer_idx)

            if should_capture and self.config.capture_pre_norm:
                self.state.pre_norm_states[layer_idx] = self._maybe_slice_positions(h)

            # Run layer forward
            # Layers can return:
            # - BlockOutput (our models) with .hidden_states
            # - Tuple (hidden_states,) or (hidden_states, cache)
            # - Just the tensor directly
            # Try with mask first (for attention layers), fall back without (for SSM layers)
            try:
                layer_out = layer(h, mask=mask, cache=cache)
            except TypeError:
                # Layer doesn't accept mask (e.g., Mamba SSM layers)
                layer_out = layer(h, cache=cache)

            # Extract hidden state from layer output
            if hasattr(layer_out, "hidden_states"):
                # BlockOutput from our models
                h = layer_out.hidden_states
                # Check for attention weights
                if (
                    should_capture
                    and self.config.capture_attention_weights
                    and hasattr(layer_out, "attention_weights")
                    and layer_out.attention_weights is not None
                ):
                    self.state.attention_weights[layer_idx] = self._maybe_slice_positions(
                        layer_out.attention_weights
                    )
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
                # Check for attention weights in output
                if should_capture and self.config.capture_attention_weights and len(layer_out) > 2:
                    self.state.attention_weights[layer_idx] = self._maybe_slice_positions(
                        layer_out[2]
                    )
            else:
                h = layer_out

            # Capture hidden state after layer
            if should_capture and self.config.capture_hidden_states:
                captured = self._maybe_slice_positions(h)
                if self.config.detach:
                    captured = mx.stop_gradient(captured)
                self.state.hidden_states[layer_idx] = captured

        # Final norm
        if final_norm is not None:
            h = final_norm(h)

        self.state.final_hidden = h

        # LM head
        if return_logits and lm_head is not None:
            head_out = lm_head(h)
            # Handle HeadOutput from our models
            if hasattr(head_out, "logits"):
                logits = head_out.logits
            else:
                logits = head_out
            self.state.logits = logits
            return logits

        return None

    def forward_to_layer(
        self,
        input_ids: mx.array,
        target_layer: int,
    ) -> mx.array:
        """
        Run forward pass up to a specific layer and return hidden state.

        Useful for probing experiments where you only need activations
        up to a certain depth.

        Args:
            input_ids: Input token IDs
            target_layer: Stop after this layer (0-indexed)

        Returns:
            Hidden state after target_layer
        """
        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        layers = self._get_layers()
        embed = self._get_embed_tokens()

        if embed is not None:
            h = embed(input_ids)
        else:
            raise ValueError("Cannot find embedding layer")

        for layer_idx, layer in enumerate(layers):
            if layer_idx > target_layer:
                break

            layer_out = layer(h)
            # Extract hidden state from layer output
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        return h

    def get_layer_logits(
        self,
        layer_idx: int,
        normalize: bool = True,
    ) -> mx.array | None:
        """
        Project hidden state from a specific layer to vocabulary logits.

        This is the core of "logit lens" - seeing what the model would
        predict if we stopped at an intermediate layer.

        Args:
            layer_idx: Which layer's hidden state to project
            normalize: Whether to apply final layer norm first

        Returns:
            Logits of shape [batch, seq, vocab] or None if layer not captured
        """
        if layer_idx not in self.state.hidden_states:
            return None

        h = self.state.hidden_states[layer_idx]

        # Apply final norm if requested
        if normalize:
            final_norm = self._get_final_norm()
            if final_norm is not None:
                h = final_norm(h)

        # Project to vocab
        lm_head = self._get_lm_head()
        if lm_head is not None:
            head_out = lm_head(h)
            # Handle HeadOutput from our models
            if hasattr(head_out, "logits"):
                return head_out.logits
            return head_out

        return None

    def __repr__(self) -> str:
        """String representation."""
        num_layers = len(self._get_layers()) if hasattr(self, "model") else "?"
        return (
            f"ModelHooks(layers={self.config.layers}, "
            f"captured={self.state.num_layers_captured}/{num_layers})"
        )
