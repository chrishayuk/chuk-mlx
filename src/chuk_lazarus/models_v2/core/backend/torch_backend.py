"""
PyTorch backend implementation.

Supports CUDA, CPU, and MPS (Apple Silicon via PyTorch).
"""

from __future__ import annotations

from typing import Any

from .base import Backend
from .types import BackendType


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
