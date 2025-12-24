"""
Grouped Query Attention (GQA).

GQA uses fewer KV heads than query heads, with groups of query heads
sharing the same KV head. This reduces memory and compute while
maintaining quality.

Used by: Llama 2 70B, Llama 3, Mistral 7B, etc.

Reference: https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import AttentionConfig
from .base import AttentionBase


class GroupedQueryAttention(AttentionBase):
    """
    Grouped Query Attention.

    Multiple query heads share each KV head, reducing memory usage.

    Args:
        config: Attention configuration with num_key_value_heads < num_attention_heads

    Example:
        >>> config = AttentionConfig(
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,  # 4 query heads per KV head
        ...     hidden_size=4096,
        ... )
        >>> attn = GroupedQueryAttention(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output, cache = attn(x, mask=create_causal_mask(10))
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        # Number of query heads per KV head
        self.n_rep = self.num_heads // self.num_kv_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.use_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.use_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.use_bias,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.use_bias,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Compute grouped query attention.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            mask: Attention mask (additive)
            cache: Optional KV cache

        Returns:
            output: Shape (batch, seq_len, hidden_size)
            cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Reshape K, V: (batch, seq_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Get cache offset for RoPE
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]

        # Apply RoPE
        q, k = self._apply_rope(q, k, offset=offset)

        # Update cache
        k, v, new_cache = self._update_cache(k, v, cache)

        # Repeat KV heads to match query heads
        # This is more efficient than doing it in the attention kernel
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Compute attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, new_cache


def create_grouped_query_attention(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None = None,
    bias: bool = False,
    rope_theta: float = 10000.0,
    max_position_embeddings: int = 4096,
) -> GroupedQueryAttention:
    """
    Factory function for GQA.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (must divide num_heads)
        head_dim: Dimension per head
        bias: Use bias in projections
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum sequence length

    Returns:
        GroupedQueryAttention instance
    """
    from ...core.config import PositionConfig, RoPEConfig

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    rope_config = RoPEConfig(
        theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
    )
    position_config = PositionConfig(rope=rope_config)

    config = AttentionConfig(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        attention_bias=bias,
        position=position_config,
    )
    return GroupedQueryAttention(config)
