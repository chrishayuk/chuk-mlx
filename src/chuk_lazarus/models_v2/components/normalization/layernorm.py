"""
Layer Normalization.

Standard layer normalization with mean centering and variance normalization.
Used by BERT, GPT-2, and some other models.

Reference: https://arxiv.org/abs/1607.06450
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes by subtracting mean and dividing by standard deviation.

    Args:
        dims: Normalization dimension (hidden_size)
        eps: Small constant for numerical stability
        affine: Whether to use learnable scale and shift

    Example:
        >>> norm = LayerNorm(dims=4096, eps=1e-5)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = norm(x)  # (2, 10, 4096)
    """

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply layer normalization.

        Args:
            x: Input, shape (..., dims)

        Returns:
            Normalized output, same shape as input
        """
        # Compute mean and variance
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply affine transformation
        if self.affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm


def create_layernorm(
    dims: int,
    eps: float = 1e-5,
    affine: bool = True,
) -> LayerNorm:
    """
    Factory function for LayerNorm.

    Args:
        dims: Normalization dimension
        eps: Epsilon for numerical stability
        affine: Use learnable parameters

    Returns:
        LayerNorm instance
    """
    return LayerNorm(dims=dims, eps=eps, affine=affine)
