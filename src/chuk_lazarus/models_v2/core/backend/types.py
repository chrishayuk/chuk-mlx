"""
Backend types and enums.

No magic strings - use enums for type safety.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# Generic tensor type alias - actual type depends on backend
# Using Any since the concrete type varies by backend (mx.array, torch.Tensor, etc.)
Tensor = Any


class BackendType(str, Enum):
    """Supported compute backends."""

    MLX = "mlx"
    TORCH = "torch"
    JAX = "jax"
    NUMPY = "numpy"  # For testing/CPU-only

    def __str__(self) -> str:
        return self.value
