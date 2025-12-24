"""
Sliding Window Attention.

Limits attention to a local window around each position, reducing
memory complexity from O(n^2) to O(n * window_size).

Used by: Mistral, Mixtral, Longformer (with global tokens)

Reference: https://arxiv.org/abs/2310.06825
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import AttentionConfig
from .base import AttentionBase, create_sliding_window_mask


class SlidingWindowAttention(AttentionBase):
    """
    Sliding Window Attention.

    Each position only attends to positions within a local window.
    This is combined with GQA for efficiency (as in Mistral).

    Args:
        config: Attention configuration with sliding_window_size set

    Example:
        >>> config = AttentionConfig(
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     hidden_size=4096,
        ...     sliding_window_size=4096,
        ... )
        >>> attn = SlidingWindowAttention(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output, cache = attn(x)
    """

    def __init__(self, config: AttentionConfig):
        super().__init__(config)

        self.window_size = config.sliding_window_size
        if self.window_size is None:
            raise ValueError("sliding_window_size must be set for SlidingWindowAttention")

        # Number of query heads per KV head (for GQA)
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

        # Mask cache
        self._mask_cache: dict[int, mx.array] = {}

    def _get_mask(self, seq_len: int, dtype) -> mx.array:
        """Get or create sliding window mask."""
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = create_sliding_window_mask(seq_len, self.window_size, dtype)
        return self._mask_cache[seq_len]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Compute sliding window attention.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            mask: Optional additional mask (combined with window mask)
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

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
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

        # For cached generation, we need to handle the sliding window differently
        kv_seq_len = k.shape[2]

        if cache is not None and kv_seq_len > self.window_size:
            # Only keep the last window_size positions in cache
            k = k[:, :, -self.window_size :, :]
            v = v[:, :, -self.window_size :, :]
            new_cache = (k, v)

        # Repeat KV heads
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Create sliding window mask
        if cache is None:
            # Training: use full sliding window mask
            window_mask = self._get_mask(seq_len, x.dtype)
        else:
            # Generation: simple causal mask (window handled by cache truncation)
            window_mask = None

        # Combine with provided mask
        if mask is not None and window_mask is not None:
            final_mask = mask + window_mask
        elif window_mask is not None:
            final_mask = window_mask
        else:
            final_mask = mask

        # Compute attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=final_mask)

        # Reshape output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, new_cache


def create_sliding_window_attention(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    window_size: int,
    head_dim: int | None = None,
    bias: bool = False,
    rope_theta: float = 10000.0,
    max_position_embeddings: int = 32768,
) -> SlidingWindowAttention:
    """
    Factory function for sliding window attention.

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        window_size: Size of attention window
        head_dim: Dimension per head
        bias: Use bias in projections
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum sequence length

    Returns:
        SlidingWindowAttention instance
    """
    from ...core.config import AttentionType, PositionConfig, RoPEConfig

    rope_config = RoPEConfig(
        theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
    )
    position_config = PositionConfig(rope=rope_config)

    config = AttentionConfig(
        attention_type=AttentionType.SLIDING_WINDOW,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        attention_bias=bias,
        sliding_window_size=window_size,
        position=position_config,
    )
    return SlidingWindowAttention(config)
