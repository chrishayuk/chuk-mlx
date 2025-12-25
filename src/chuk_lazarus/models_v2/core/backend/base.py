"""
Abstract backend interface.

Provides unified tensor operations across frameworks.
All concrete backends (MLX, PyTorch, JAX) implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import BackendType, Tensor


class Backend(ABC):
    """
    Abstract backend interface.

    Provides unified tensor operations across frameworks.
    Async-native design - all implementations should be non-blocking where possible.
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

    # === Tensor Creation ===

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

    # === Tensor Operations ===

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

    # === Shape Operations ===

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

    # === Attention Operations ===

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

    # === Gradient Operations ===

    @abstractmethod
    def stop_gradient(self, x: Tensor) -> Tensor:
        """Stop gradient propagation."""
        ...

    # === Evaluation ===

    @abstractmethod
    def eval(self, *tensors: Tensor) -> None:
        """Force evaluation of lazy tensors (for MLX)."""
        ...
