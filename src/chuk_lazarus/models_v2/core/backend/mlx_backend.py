"""
MLX backend implementation for Apple Silicon.

Optimized for Metal Performance Shaders on Apple hardware.
"""

from __future__ import annotations

from typing import Any

from .base import Backend
from .types import BackendType


class MLXBackend(Backend):
    """MLX backend implementation for Apple Silicon."""

    def __init__(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn

            self._mx = mx
            self._nn = nn
        except ImportError as e:
            raise ImportError(
                "MLX is required for MLXBackend. Install with: pip install mlx"
            ) from e

    @property
    def name(self) -> BackendType:
        return BackendType.MLX

    @property
    def device(self) -> str:
        return "mps"  # MLX runs on Metal

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._mx.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._mx.ones(shape, dtype=dtype)

    def randn(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._mx.random.normal(shape, dtype=dtype)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return self._mx.arange(start, end, step, dtype=dtype)

    def from_numpy(self, array: Any) -> Any:
        return self._mx.array(array)

    def to_numpy(self, tensor: Any) -> Any:
        import numpy as np

        return np.array(tensor)

    def matmul(self, a: Any, b: Any) -> Any:
        return self._mx.matmul(a, b)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return self._mx.softmax(x, axis=axis)

    def relu(self, x: Any) -> Any:
        return self._mx.maximum(x, 0)

    def silu(self, x: Any) -> Any:
        return x * self._mx.sigmoid(x)

    def gelu(self, x: Any) -> Any:
        return self._nn.gelu(x)

    def tanh(self, x: Any) -> Any:
        return self._mx.tanh(x)

    def sigmoid(self, x: Any) -> Any:
        return self._mx.sigmoid(x)

    def layer_norm(self, x: Any, weight: Any, bias: Any | None, eps: float) -> Any:
        mean = self._mx.mean(x, axis=-1, keepdims=True)
        var = self._mx.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / self._mx.sqrt(var + eps)
        result = weight * normalized
        if bias is not None:
            result = result + bias
        return result

    def rms_norm(self, x: Any, weight: Any, eps: float) -> Any:
        rms = self._mx.sqrt(self._mx.mean(x * x, axis=-1, keepdims=True) + eps)
        return weight * (x / rms)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def transpose(self, x: Any, axes: tuple[int, ...]) -> Any:
        return x.transpose(axes)

    def concatenate(self, tensors: list[Any], axis: int = 0) -> Any:
        return self._mx.concatenate(tensors, axis=axis)

    def split(self, x: Any, num_splits: int, axis: int = 0) -> list[Any]:
        return self._mx.split(x, num_splits, axis=axis)

    def scaled_dot_product_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Any | None = None,
        scale: float | None = None,
    ) -> Any:
        # MLX requires scale to be a float
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)
        return self._mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)

    def create_causal_mask(self, seq_len: int, dtype: Any = None) -> Any:
        return self._nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

    def stop_gradient(self, x: Any) -> Any:
        return self._mx.stop_gradient(x)

    def eval(self, *tensors: Any) -> None:
        self._mx.eval(*tensors)
