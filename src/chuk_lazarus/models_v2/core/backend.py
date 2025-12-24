"""
Backend abstraction for compute frameworks.

Provides a unified interface that works across:
- MLX (Apple Silicon)
- PyTorch (CUDA, CPU)
- JAX (TPU, GPU)

The framework is backend-agnostic by design. Components use
abstract tensor types and the backend provides concrete implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Generic tensor type - actual type depends on backend
Tensor = TypeVar("Tensor")


class BackendType(str, Enum):
    """Supported compute backends."""

    MLX = "mlx"
    TORCH = "torch"
    JAX = "jax"
    NUMPY = "numpy"  # For testing/CPU-only

    def __str__(self) -> str:
        return self.value


class Backend(ABC):
    """
    Abstract backend interface.

    Provides unified tensor operations across frameworks.
    """

    @property
    @abstractmethod
    def name(self) -> BackendType:
        """Backend identifier."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Current device (e.g., 'gpu', 'cpu', 'mps')."""
        ...

    # Tensor creation
    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Tensor:
        """Create tensor of zeros."""
        ...

    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Tensor:
        """Create tensor of ones."""
        ...

    @abstractmethod
    def randn(self, shape: tuple[int, ...], dtype: Any = None) -> Tensor:
        """Create tensor with random normal values."""
        ...

    @abstractmethod
    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Tensor:
        """Create range tensor."""
        ...

    @abstractmethod
    def from_numpy(self, array: Any) -> Tensor:
        """Convert numpy array to tensor."""
        ...

    @abstractmethod
    def to_numpy(self, tensor: Tensor) -> Any:
        """Convert tensor to numpy array."""
        ...

    # Tensor operations
    @abstractmethod
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication."""
        ...

    @abstractmethod
    def softmax(self, x: Tensor, axis: int = -1) -> Tensor:
        """Softmax activation."""
        ...

    @abstractmethod
    def relu(self, x: Tensor) -> Tensor:
        """ReLU activation."""
        ...

    @abstractmethod
    def silu(self, x: Tensor) -> Tensor:
        """SiLU/Swish activation."""
        ...

    @abstractmethod
    def gelu(self, x: Tensor) -> Tensor:
        """GELU activation."""
        ...

    @abstractmethod
    def tanh(self, x: Tensor) -> Tensor:
        """Tanh activation."""
        ...

    @abstractmethod
    def sigmoid(self, x: Tensor) -> Tensor:
        """Sigmoid activation."""
        ...

    @abstractmethod
    def layer_norm(self, x: Tensor, weight: Tensor, bias: Tensor | None, eps: float) -> Tensor:
        """Layer normalization."""
        ...

    @abstractmethod
    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        """RMS normalization."""
        ...

    # Shape operations
    @abstractmethod
    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Reshape tensor."""
        ...

    @abstractmethod
    def transpose(self, x: Tensor, axes: tuple[int, ...]) -> Tensor:
        """Transpose tensor."""
        ...

    @abstractmethod
    def concatenate(self, tensors: list[Tensor], axis: int = 0) -> Tensor:
        """Concatenate tensors."""
        ...

    @abstractmethod
    def split(self, x: Tensor, num_splits: int, axis: int = 0) -> list[Tensor]:
        """Split tensor."""
        ...

    # Attention operations
    @abstractmethod
    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        scale: float | None = None,
    ) -> Tensor:
        """Scaled dot-product attention."""
        ...

    @abstractmethod
    def create_causal_mask(self, seq_len: int, dtype: Any = None) -> Tensor:
        """Create causal attention mask."""
        ...

    # Gradient operations
    @abstractmethod
    def stop_gradient(self, x: Tensor) -> Tensor:
        """Stop gradient propagation."""
        ...

    # Evaluation
    @abstractmethod
    def eval(self, *tensors: Tensor) -> None:
        """Force evaluation of lazy tensors (for MLX)."""
        ...


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


class TorchBackend(Backend):
    """PyTorch backend implementation."""

    def __init__(self, device: str = "cuda"):
        try:
            import torch

            self._torch = torch
            self._device = device if torch.cuda.is_available() else "cpu"
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TorchBackend. Install with: pip install torch"
            ) from e

    @property
    def name(self) -> BackendType:
        return BackendType.TORCH

    @property
    def device(self) -> str:
        return self._device

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.zeros(shape, dtype=dtype, device=self._device)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.ones(shape, dtype=dtype, device=self._device)

    def randn(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        return self._torch.randn(shape, dtype=dtype, device=self._device)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return self._torch.arange(start, end, step, dtype=dtype, device=self._device)

    def from_numpy(self, array: Any) -> Any:
        return self._torch.from_numpy(array).to(self._device)

    def to_numpy(self, tensor: Any) -> Any:
        return tensor.cpu().numpy()

    def matmul(self, a: Any, b: Any) -> Any:
        return self._torch.matmul(a, b)

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return self._torch.softmax(x, dim=axis)

    def relu(self, x: Any) -> Any:
        return self._torch.relu(x)

    def silu(self, x: Any) -> Any:
        return self._torch.nn.functional.silu(x)

    def gelu(self, x: Any) -> Any:
        return self._torch.nn.functional.gelu(x)

    def tanh(self, x: Any) -> Any:
        return self._torch.tanh(x)

    def sigmoid(self, x: Any) -> Any:
        return self._torch.sigmoid(x)

    def layer_norm(self, x: Any, weight: Any, bias: Any | None, eps: float) -> Any:
        return self._torch.nn.functional.layer_norm(
            x, weight.shape, weight=weight, bias=bias, eps=eps
        )

    def rms_norm(self, x: Any, weight: Any, eps: float) -> Any:
        rms = self._torch.sqrt(self._torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return weight * (x / rms)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        return x.reshape(shape)

    def transpose(self, x: Any, axes: tuple[int, ...]) -> Any:
        return x.permute(axes)

    def concatenate(self, tensors: list[Any], axis: int = 0) -> Any:
        return self._torch.cat(tensors, dim=axis)

    def split(self, x: Any, num_splits: int, axis: int = 0) -> list[Any]:
        return list(self._torch.chunk(x, num_splits, dim=axis))

    def scaled_dot_product_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        mask: Any | None = None,
        scale: float | None = None,
    ) -> Any:
        return self._torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scale
        )

    def create_causal_mask(self, seq_len: int, dtype: Any = None) -> Any:
        mask = self._torch.triu(
            self._torch.ones(seq_len, seq_len, device=self._device),
            diagonal=1,
        )
        return mask.masked_fill(mask == 1, float("-inf"))

    def stop_gradient(self, x: Any) -> Any:
        return x.detach()

    def eval(self, *tensors: Any) -> None:
        # PyTorch is eager, no-op
        pass


# Backend registry
_current_backend: Backend | None = None


def get_backend() -> Backend:
    """Get the current backend."""
    global _current_backend
    if _current_backend is None:
        # Default to MLX on macOS, PyTorch otherwise
        import platform

        if platform.system() == "Darwin":
            try:
                _current_backend = MLXBackend()
            except ImportError:
                _current_backend = TorchBackend()
        else:
            _current_backend = TorchBackend()

    return _current_backend


def set_backend(backend: Backend | BackendType | str) -> None:
    """Set the current backend."""
    global _current_backend

    if isinstance(backend, Backend):
        _current_backend = backend
    elif isinstance(backend, (BackendType, str)):
        backend_type = BackendType(str(backend))
        if backend_type == BackendType.MLX:
            _current_backend = MLXBackend()
        elif backend_type == BackendType.TORCH:
            _current_backend = TorchBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
    else:
        raise TypeError(f"Expected Backend, BackendType, or str, got {type(backend)}")


def reset_backend() -> None:
    """Reset to default backend."""
    global _current_backend
    _current_backend = None
