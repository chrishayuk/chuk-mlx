"""
Transformer block.

Standard transformer block: attention + FFN with pre-norm.
Supports different attention types (MHA, GQA, sliding window).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..components.attention import GroupedQueryAttention, MultiHeadAttention, SlidingWindowAttention
from ..components.ffn import GEGLU, MLP, SwiGLU
from ..components.normalization import LayerNorm, RMSNorm
from ..core.config import ModelConfig
from ..core.enums import AttentionType, BlockType, FFNType
from .base import Block, BlockOutput


class TransformerBlock(Block):
    """
    Standard transformer block with pre-normalization.

    Architecture:
        x -> norm1 -> attention -> + residual -> norm2 -> ffn -> + residual -> output

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA, None = MHA)
        intermediate_size: FFN intermediate dimension
        attention_type: Type of attention (MHA, GQA, sliding)
        ffn_type: Type of FFN (MLP, SwiGLU, GEGLU)
        norm_type: Normalization type (rmsnorm, layernorm)
        norm_eps: Epsilon for normalization
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum sequence length for RoPE
        sliding_window: Window size for sliding window attention
        attention_dropout: Dropout for attention
        hidden_dropout: Dropout for FFN

    Example:
        >>> block = TransformerBlock(
        ...     hidden_size=4096,
        ...     num_heads=32,
        ...     num_kv_heads=8,  # GQA
        ...     intermediate_size=14336,
        ...     ffn_type=FFNType.SWIGLU,
        ... )
        >>> x = mx.random.normal((2, 100, 4096))
        >>> output = block(x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        attention_type: AttentionType = AttentionType.GROUPED_QUERY,
        ffn_type: FFNType = FFNType.SWIGLU,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        sliding_window: int | None = None,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads or num_heads
        self._intermediate_size = intermediate_size or hidden_size * 4

        # Determine head dimension
        head_dim = hidden_size // num_heads

        # Pre-attention norm
        if norm_type == "rmsnorm":
            self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        else:
            self.input_layernorm = LayerNorm(hidden_size, eps=norm_eps)
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=norm_eps)

        # Attention - create config
        from ..core.config import AttentionConfig, PositionConfig, RoPEConfig

        rope_config = RoPEConfig(
            theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        position_config = PositionConfig(
            max_position_embeddings=max_position_embeddings,
            rope=rope_config,
        )
        attention_config = AttentionConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=self._num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            sliding_window_size=sliding_window,
            position=position_config,
        )

        if sliding_window is not None:
            self.self_attn = SlidingWindowAttention(attention_config)
        elif self._num_kv_heads < num_heads:
            self.self_attn = GroupedQueryAttention(attention_config)
        else:
            self.self_attn = MultiHeadAttention(attention_config)

        # FFN - create config
        from ..core.config import FFNConfig

        ffn_config = FFNConfig(
            hidden_size=hidden_size,
            intermediate_size=self._intermediate_size,
            ffn_type=ffn_type,
            dropout=hidden_dropout,
        )

        if ffn_type == FFNType.SWIGLU:
            self.mlp = SwiGLU(ffn_config)
        elif ffn_type == FFNType.GEGLU:
            self.mlp = GEGLU(ffn_config)
        else:
            self.mlp = MLP(ffn_config)

        # Optional dropout
        self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else None

    @property
    def block_type(self) -> BlockType:
        return BlockType.TRANSFORMER

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> BlockOutput:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            mask: Optional attention mask
            cache: Optional (key, value) cache for inference

        Returns:
            BlockOutput with hidden states and updated cache
        """
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        if self.dropout is not None:
            x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Initialize KV cache."""
        head_dim = self._hidden_size // self._num_heads
        # (batch, num_kv_heads, max_seq_len, head_dim)
        k_cache = mx.zeros((batch_size, self._num_kv_heads, max_seq_len, head_dim))
        v_cache = mx.zeros((batch_size, self._num_kv_heads, max_seq_len, head_dim))
        return k_cache, v_cache

    @classmethod
    def from_config(cls, config: ModelConfig, layer_idx: int = 0) -> TransformerBlock:
        """
        Create TransformerBlock from ModelConfig.

        Args:
            config: Model configuration
            layer_idx: Layer index (for future per-layer configs)

        Returns:
            TransformerBlock instance
        """
        # Determine attention type
        if config.sliding_window:
            attention_type = AttentionType.SLIDING_WINDOW
        elif config.num_key_value_heads and config.num_key_value_heads < config.num_attention_heads:
            attention_type = AttentionType.GROUPED_QUERY
        else:
            attention_type = AttentionType.MULTI_HEAD

        # Determine FFN type
        if config.hidden_act in ("silu", "swish"):
            ffn_type = FFNType.SWIGLU
        elif config.hidden_act == "gelu":
            ffn_type = FFNType.GEGLU
        else:
            ffn_type = FFNType.MLP

        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            attention_type=attention_type,
            ffn_type=ffn_type,
            norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            sliding_window=config.sliding_window,
        )


def create_transformer_block(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    intermediate_size: int | None = None,
    ffn_type: FFNType = FFNType.SWIGLU,
    norm_eps: float = 1e-5,
) -> TransformerBlock:
    """
    Factory function for TransformerBlock.

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (None = MHA)
        intermediate_size: FFN intermediate dimension
        ffn_type: Type of FFN
        norm_eps: Normalization epsilon

    Returns:
        TransformerBlock instance
    """
    return TransformerBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        ffn_type=ffn_type,
        norm_eps=norm_eps,
    )
