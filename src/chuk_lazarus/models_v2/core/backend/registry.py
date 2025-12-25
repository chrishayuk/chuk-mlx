"""
Backend registry and management.

Provides global backend instance management with auto-detection.
"""

from __future__ import annotations

import logging

from .base import Backend
from .types import BackendType

logger = logging.getLogger(__name__)

# Global backend registry
_current_backend: Backend | None = None


def get_backend() -> Backend:
    """
    Get the current backend.

    Auto-detects the best backend based on platform:
    - macOS: MLX (Apple Silicon optimized)
    - Other: PyTorch (CUDA/CPU)
    """
    global _current_backend
    if _current_backend is None:
        # Default to MLX on macOS, PyTorch otherwise
        import platform

        if platform.system() == "Darwin":
            try:
                from .mlx_backend import MLXBackend

                _current_backend = MLXBackend()
                logger.debug("Auto-selected MLX backend for macOS")
            except ImportError:
                from .torch_backend import TorchBackend

                _current_backend = TorchBackend()
                logger.debug("Falling back to PyTorch backend (MLX not available)")
        else:
            from .torch_backend import TorchBackend

            _current_backend = TorchBackend()
            logger.debug("Auto-selected PyTorch backend")

    return _current_backend


def set_backend(backend: Backend | BackendType | str) -> None:
    """
    Set the current backend.

    Args:
        backend: Backend instance, BackendType enum, or string name
    """
    global _current_backend

    if isinstance(backend, Backend):
        _current_backend = backend
    elif isinstance(backend, (BackendType, str)):
        backend_type = BackendType(str(backend))
        if backend_type == BackendType.MLX:
            from .mlx_backend import MLXBackend

            _current_backend = MLXBackend()
        elif backend_type == BackendType.TORCH:
            from .torch_backend import TorchBackend

            _current_backend = TorchBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
    else:
        raise TypeError(f"Expected Backend, BackendType, or str, got {type(backend)}")

    logger.debug(f"Set backend to {_current_backend.name}")


def reset_backend() -> None:
    """Reset to default backend (auto-detection on next get_backend() call)."""
    global _current_backend
    _current_backend = None
    logger.debug("Reset backend to auto-detection")
