"""
Attention with Linear Biases (ALiBi).

ALiBi adds position-dependent biases to attention scores instead of
using position embeddings. This allows for better length extrapolation.

Reference: https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class ALiBi(nn.Module):
    """
    ALiBi position bias generator.

    Generates position-dependent biases to add to attention scores.
    The biases decay linearly with distance, with different slopes per head.

    Args:
        num_heads: Number of attention heads

    Example:
        >>> alibi = ALiBi(num_heads=32)
        >>> bias = alibi(seq_len=100)  # (1, 32, 100, 100)
        >>> attention_scores = attention_scores + bias
    """

    def __init__(self, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        # Compute slopes for each head
        slopes = self._compute_slopes(num_heads)
        self._slopes = mx.array(slopes).reshape(1, num_heads, 1, 1)

    def _compute_slopes(self, num_heads: int) -> list[float]:
        """
        Compute ALiBi slopes for each head.

        Slopes follow a geometric sequence based on head count.

        Args:
            num_heads: Number of attention heads

        Returns:
            List of slopes, one per head
        """

        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)

        # For non-power-of-2, interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)

        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2][: num_heads - closest_power_of_2]

        return slopes + extra_slopes

    def __call__(self, seq_len: int) -> mx.array:
        """
        Generate ALiBi bias tensor.

        Args:
            seq_len: Sequence length

        Returns:
            Bias tensor, shape (1, num_heads, seq_len, seq_len)
        """
        # Create position difference matrix
        positions = mx.arange(seq_len)
        # (seq_len, seq_len) - relative positions
        relative_positions = positions[:, None] - positions[None, :]

        # Apply slopes: (1, num_heads, seq_len, seq_len)
        bias = self._slopes * relative_positions[None, None, :, :]

        return bias

    def get_bias_for_cache(
        self,
        query_len: int,
        key_len: int,
    ) -> mx.array:
        """
        Generate ALiBi bias for cached inference.

        During generation, query_len=1 and key_len grows.

        Args:
            query_len: Number of query positions
            key_len: Number of key positions

        Returns:
            Bias tensor, shape (1, num_heads, query_len, key_len)
        """
        # Query positions start from key_len - query_len
        query_positions = mx.arange(key_len - query_len, key_len)
        key_positions = mx.arange(key_len)

        # Relative positions: (query_len, key_len)
        relative_positions = query_positions[:, None] - key_positions[None, :]

        # Apply slopes: (1, num_heads, query_len, key_len)
        bias = self._slopes * relative_positions[None, None, :, :]

        return bias


def compute_alibi_bias(
    num_heads: int,
    seq_len: int,
) -> mx.array:
    """
    Compute ALiBi bias tensor (functional API).

    Args:
        num_heads: Number of attention heads
        seq_len: Sequence length

    Returns:
        Bias tensor, shape (1, num_heads, seq_len, seq_len)
    """
    alibi = ALiBi(num_heads)
    return alibi(seq_len)


def compute_alibi_slopes(num_heads: int) -> mx.array:
    """
    Compute just the ALiBi slopes.

    Args:
        num_heads: Number of attention heads

    Returns:
        Slopes tensor, shape (num_heads,)
    """
    alibi = ALiBi(num_heads)
    return alibi._slopes.squeeze()
