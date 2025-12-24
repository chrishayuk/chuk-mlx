"""
Normalization variants for specific architectures.

GemmaNorm: RMSNorm with +1 offset (used by Gemma)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GemmaNorm(nn.Module):
    """
    Gemma-style RMSNorm.

    Same as RMSNorm but adds 1 to the weight before applying.
    This is a quirk of the Gemma architecture.

    Args:
        dims: Normalization dimension (hidden_size)
        eps: Small constant for numerical stability

    Example:
        >>> norm = GemmaNorm(dims=4096, eps=1e-6)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = norm(x)  # (2, 10, 4096)
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()

        self.dims = dims
        self.eps = eps

        # Learnable scale parameter (will have 1 added)
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply Gemma-style RMS normalization.

        Args:
            x: Input, shape (..., dims)

        Returns:
            Normalized output, same shape as input
        """
        # Compute RMS
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale with +1 offset
        return (1 + self.weight) * (x / rms)


def create_gemma_norm(dims: int, eps: float = 1e-6) -> GemmaNorm:
    """
    Factory function for GemmaNorm.

    Args:
        dims: Normalization dimension
        eps: Epsilon for numerical stability

    Returns:
        GemmaNorm instance
    """
    return GemmaNorm(dims=dims, eps=eps)
