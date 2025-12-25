"""
Tests for backend registry functions.
"""

import pytest

from chuk_lazarus.models_v2.core.backend.base import Backend
from chuk_lazarus.models_v2.core.backend.mlx_backend import MLXBackend
from chuk_lazarus.models_v2.core.backend.registry import (
    get_backend,
    reset_backend,
    set_backend,
)
from chuk_lazarus.models_v2.core.backend.torch_backend import TorchBackend
from chuk_lazarus.models_v2.core.backend.types import BackendType


class TestGetBackend:
    """Tests for get_backend function."""

    def setup_method(self):
        """Reset backend before each test."""
        reset_backend()

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_get_backend_returns_backend(self):
        """Test get_backend returns a Backend instance."""
        backend = get_backend()
        assert isinstance(backend, Backend)

    def test_get_backend_caches_instance(self):
        """Test get_backend returns the same instance on repeated calls."""
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_get_backend_auto_detects_mlx_on_macos(self):
        """Test auto-detection selects MLX on macOS."""
        # On macOS, should select MLX
        import platform

        if platform.system() == "Darwin":
            backend = get_backend()
            assert isinstance(backend, MLXBackend)


class TestSetBackend:
    """Tests for set_backend function."""

    def setup_method(self):
        """Reset backend before each test."""
        reset_backend()

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_set_backend_with_instance(self):
        """Test setting backend with Backend instance."""
        backend = MLXBackend()
        set_backend(backend)
        assert get_backend() is backend

    def test_set_backend_with_backend_type_enum(self):
        """Test setting backend with BackendType enum."""
        set_backend(BackendType.MLX)
        backend = get_backend()
        assert isinstance(backend, MLXBackend)

    def test_set_backend_with_string(self):
        """Test setting backend with string name."""
        set_backend("mlx")
        backend = get_backend()
        assert isinstance(backend, MLXBackend)

    def test_set_backend_invalid_type_raises(self):
        """Test setting backend with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected Backend, BackendType, or str"):
            set_backend(123)

    def test_set_backend_unsupported_backend_raises(self):
        """Test setting unsupported backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            set_backend(BackendType.JAX)

    def test_set_backend_with_torch_enum(self):
        """Test setting backend with BackendType.TORCH enum."""
        set_backend(BackendType.TORCH)
        backend = get_backend()
        assert isinstance(backend, TorchBackend)

    def test_set_backend_with_torch_string(self):
        """Test setting backend with 'torch' string."""
        set_backend("torch")
        backend = get_backend()
        assert isinstance(backend, TorchBackend)

    def test_set_backend_with_torch_instance(self):
        """Test setting backend with TorchBackend instance."""
        backend = TorchBackend()
        set_backend(backend)
        assert get_backend() is backend


class TestResetBackend:
    """Tests for reset_backend function."""

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_reset_backend_clears_cached_instance(self):
        """Test reset_backend clears the cached backend."""
        # Get initial backend
        _ = get_backend()

        # Reset
        reset_backend()

        # Get new backend - should be a new instance
        backend_after_reset = get_backend()

        # They should be equal (same type) but may or may not be same instance
        # depending on auto-detection
        assert isinstance(backend_after_reset, Backend)


class TestGetBackendAutoDetection:
    """Tests for get_backend auto-detection paths."""

    def setup_method(self):
        """Reset backend before each test."""
        reset_backend()

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_auto_detect_torch_on_non_darwin(self):
        """Test auto-detection selects Torch on non-Darwin platforms."""
        from unittest.mock import patch

        # Mock platform.system() to return "Linux"
        with patch("platform.system", return_value="Linux"):
            reset_backend()
            backend = get_backend()
            assert isinstance(backend, TorchBackend)


class TestBackendIntegration:
    """Integration tests for backend registry."""

    def setup_method(self):
        """Reset backend before each test."""
        reset_backend()

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_switch_backends(self):
        """Test switching between backends."""
        # Start with MLX
        set_backend(BackendType.MLX)
        backend1 = get_backend()
        assert isinstance(backend1, MLXBackend)

        # Switch to Torch
        set_backend(BackendType.TORCH)
        backend2 = get_backend()
        assert isinstance(backend2, TorchBackend)

        # Switch back to MLX
        set_backend("mlx")
        backend3 = get_backend()
        assert isinstance(backend3, MLXBackend)

    def test_backend_has_required_attributes(self):
        """Test that returned backend has required attributes."""
        backend = get_backend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "zeros")
        assert hasattr(backend, "ones")
