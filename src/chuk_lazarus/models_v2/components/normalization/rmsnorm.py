"""
RMSNorm (Root Mean Square Layer Normalization).

Simpler and faster than LayerNorm - doesn't subtract mean.
Used by Llama, Mistral, and most modern LLMs.

Reference: https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import NormConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes by RMS without centering (no mean subtraction).
    Faster than LayerNorm with similar performance.

    Args:
        dims: Normalization dimension (hidden_size)
        eps: Small constant for numerical stability

    Example:
        >>> norm = RMSNorm(dims=4096, eps=1e-6)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = norm(x)  # (2, 10, 4096)
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()

        self.dims = dims
        self.eps = eps

        # Learnable scale parameter
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply RMS normalization.

        Args:
            x: Input, shape (..., dims)

        Returns:
            Normalized output, same shape as input
        """
        # Compute RMS
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale
        return self.weight * (x / rms)

    @classmethod
    def from_config(cls, config: NormConfig) -> RMSNorm:
        """
        Create from NormConfig.

        Note: NormConfig doesn't have dims, so this is a convenience
        for when dims is passed separately.
        """
        raise NotImplementedError("Use RMSNorm(dims, eps) directly")


def create_rmsnorm(dims: int, eps: float = 1e-6) -> RMSNorm:
    """
    Factory function for RMSNorm.

    Args:
        dims: Normalization dimension
        eps: Epsilon for numerical stability

    Returns:
        RMSNorm instance
    """
    return RMSNorm(dims=dims, eps=eps)
