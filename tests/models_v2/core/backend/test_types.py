"""
Tests for backend types and enums.
"""

import pytest

from chuk_lazarus.models_v2.core.backend import BackendType


class TestBackendType:
    """Tests for BackendType enum."""

    def test_values(self):
        """Test enum values."""
        assert BackendType.MLX.value == "mlx"
        assert BackendType.TORCH.value == "torch"
        assert BackendType.JAX.value == "jax"
        assert BackendType.NUMPY.value == "numpy"

    def test_str(self):
        """Test string conversion."""
        assert str(BackendType.MLX) == "mlx"
        assert str(BackendType.TORCH) == "torch"

    def test_from_string(self):
        """Test creating from string."""
        assert BackendType("mlx") == BackendType.MLX
        assert BackendType("torch") == BackendType.TORCH

    def test_invalid_string(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            BackendType("invalid")

    def test_all_backends_defined(self):
        """Test all expected backends are defined."""
        backends = list(BackendType)
        assert len(backends) == 4
        assert BackendType.MLX in backends
        assert BackendType.TORCH in backends
        assert BackendType.JAX in backends
        assert BackendType.NUMPY in backends
