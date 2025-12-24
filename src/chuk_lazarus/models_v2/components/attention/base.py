"""
Base attention utilities.

Provides common functionality for all attention implementations:
- Causal mask creation
- KV cache management
- Attention output reshaping
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...core.config import AttentionConfig


class AttentionBase(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.

    Subclasses implement specific attention patterns (MHA, GQA, etc.).

    Args:
        config: Attention configuration
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        # Bias setting
        self.use_bias = config.attention_bias

        # RoPE setup (if using rotary embeddings)
        self.rope = None
        if config.position.position_type.value == "rope":
            from ..embeddings.rope import RoPE

            self.rope = RoPE(config.position.rope, dims=self.head_dim)

    @abstractmethod
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Compute attention.

        Args:
            x: Input hidden states, shape (batch, seq_len, hidden_size)
            mask: Attention mask (additive, -inf for masked positions)
            cache: Optional (key_cache, value_cache) tuple

        Returns:
            output: Attention output, shape (batch, seq_len, hidden_size)
            cache: Updated (key_cache, value_cache) or None
        """
        ...

    def _apply_rope(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q: Queries, shape (batch, num_heads, seq_len, head_dim)
            k: Keys, shape (batch, num_kv_heads, seq_len, head_dim)
            offset: Position offset for KV cache

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        return q, k

    def _update_cache(
        self,
        k: mx.array,
        v: mx.array,
        cache: tuple[mx.array, mx.array] | None,
    ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
        """
        Update KV cache with new keys and values.

        Args:
            k: New keys, shape (batch, num_kv_heads, seq_len, head_dim)
            v: New values, shape (batch, num_kv_heads, seq_len, head_dim)
            cache: Existing cache or None

        Returns:
            Tuple of (full_k, full_v, updated_cache)
        """
        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        return k, v, (k, v)

    def _repeat_kv(self, x: mx.array, n_rep: int) -> mx.array:
        """
        Repeat KV heads to match query heads (for GQA/MQA).

        Args:
            x: KV tensor, shape (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of times to repeat each KV head

        Returns:
            Repeated tensor, shape (batch, num_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return x

        batch, num_kv_heads, seq_len, head_dim = x.shape

        # Repeat along head dimension
        x = mx.expand_dims(x, axis=2)  # (batch, num_kv_heads, 1, seq_len, head_dim)
        x = mx.repeat(x, n_rep, axis=2)  # (batch, num_kv_heads, n_rep, seq_len, head_dim)
        x = x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

        return x


def create_causal_mask(seq_len: int, dtype: Any = None) -> mx.array:
    """
    Create a causal (autoregressive) attention mask.

    Args:
        seq_len: Sequence length
        dtype: Data type for the mask

    Returns:
        Causal mask, shape (seq_len, seq_len)
        Upper triangular is -inf, diagonal and below is 0.
    """
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    if dtype is not None:
        mask = mask.astype(dtype)
    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    dtype: Any = None,
) -> mx.array:
    """
    Create a sliding window attention mask.

    Each position can only attend to positions within the window.

    Args:
        seq_len: Sequence length
        window_size: Size of the attention window
        dtype: Data type for the mask

    Returns:
        Sliding window mask, shape (seq_len, seq_len)
    """
    # Start with causal mask
    mask = create_causal_mask(seq_len, dtype)

    # Create window mask
    positions = mx.arange(seq_len)
    row_indices = positions[:, None]
    col_indices = positions[None, :]

    # Mask positions outside window (before row - window_size)
    window_mask = col_indices < (row_indices - window_size + 1)

    # Apply window mask
    neg_inf = mx.array(float("-inf"))
    if dtype is not None:
        neg_inf = neg_inf.astype(dtype)

    mask = mx.where(window_mask, neg_inf, mask)

    return mask
