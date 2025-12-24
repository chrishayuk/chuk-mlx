"""
Transformer backbone.

Standard decoder-only transformer architecture used by Llama, Mistral, etc.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..blocks import TransformerBlock
from ..components.embeddings import create_token_embedding
from ..components.normalization import LayerNorm, RMSNorm
from ..core.config import ModelConfig
from ..core.enums import FFNType
from .base import Backbone, BackboneOutput


class TransformerBackbone(Backbone):
    """
    Transformer backbone (decoder-only).

    Consists of:
    - Token embeddings (RoPE is applied per-layer)
    - Stack of transformer blocks
    - Final layer normalization

    Args:
        vocab_size: Vocabulary size
        hidden_size: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        intermediate_size: FFN intermediate dimension
        ffn_type: Type of FFN (MLP, SwiGLU, GEGLU)
        norm_type: Normalization type
        norm_eps: Normalization epsilon
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum sequence length
        sliding_window: Optional sliding window size
        tie_word_embeddings: Share input/output embeddings

    Example:
        >>> backbone = TransformerBackbone(
        ...     vocab_size=32000,
        ...     hidden_size=4096,
        ...     num_layers=32,
        ...     num_heads=32,
        ...     num_kv_heads=8,
        ... )
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = backbone(input_ids)
        >>> output.last_hidden_state.shape
        (1, 5, 4096)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        ffn_type: FFNType = FFNType.SWIGLU,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        sliding_window: int | None = None,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads or num_heads
        self._intermediate_size = intermediate_size or hidden_size * 4
        self.tie_word_embeddings = tie_word_embeddings

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=self._num_kv_heads,
                intermediate_size=self._intermediate_size,
                ffn_type=ffn_type,
                norm_type=norm_type,
                norm_eps=norm_eps,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                sliding_window=sliding_window,
            )
            for _ in range(num_layers)
        ]

        # Final normalization
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(hidden_size, eps=norm_eps)
        else:
            self.norm = LayerNorm(hidden_size, eps=norm_eps)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask
            cache: Optional list of KV caches
            output_hidden_states: Return all layer hidden states

        Returns:
            BackboneOutput with hidden states and cache
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            # Standard causal mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)
        else:
            mask = attention_mask

        # Track hidden states if requested
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # Initialize cache list
        new_cache = []

        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            output = layer(hidden_states, mask=mask, cache=layer_cache)
            hidden_states = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> list[tuple[mx.array, mx.array]]:
        """Initialize KV cache for all layers."""
        return [layer.init_cache(batch_size, max_seq_len) for layer in self.layers]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings

    @classmethod
    def from_config(cls, config: ModelConfig) -> TransformerBackbone:
        """Create from ModelConfig."""
        # Determine FFN type
        if config.hidden_act in ("silu", "swish"):
            ffn_type = FFNType.SWIGLU
        elif config.hidden_act == "gelu":
            ffn_type = FFNType.GEGLU
        else:
            ffn_type = FFNType.MLP

        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            ffn_type=ffn_type,
            norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            sliding_window=config.sliding_window,
            tie_word_embeddings=config.tie_word_embeddings,
        )


def create_transformer_backbone(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    intermediate_size: int | None = None,
) -> TransformerBackbone:
    """Factory function for TransformerBackbone."""
    return TransformerBackbone(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
    )
