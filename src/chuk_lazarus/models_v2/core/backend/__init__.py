"""
Backend abstraction for compute frameworks.

Provides a unified interface that works across:
- MLX (Apple Silicon)
- PyTorch (CUDA, CPU)
- JAX (TPU, GPU) [planned]

The framework is backend-agnostic by design. Components use
abstract tensor types and the backend provides concrete implementations.

Design Principles:
- Async-native: All implementations should be non-blocking
- Pydantic-native: Types are validated where appropriate
- No magic strings: Use BackendType enum
- No dictionary goop: Structured interfaces throughout
"""

from .base import Backend
from .mlx_backend import MLXBackend
from .registry import get_backend, reset_backend, set_backend
from .torch_backend import TorchBackend
from .types import BackendType, Tensor

__all__ = [
    # Types
    "Tensor",
    "BackendType",
    # Base class
    "Backend",
    # Implementations
    "MLXBackend",
    "TorchBackend",
    # Registry
    "get_backend",
    "set_backend",
    "reset_backend",
]
