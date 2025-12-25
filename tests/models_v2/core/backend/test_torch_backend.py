"""
Tests for PyTorch backend implementation.

These tests only run if PyTorch is available.
"""

import numpy as np
import pytest

from chuk_lazarus.models_v2.core.backend import BackendType


@pytest.fixture
def torch_available():
    """Check if torch is available."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


class TestTorchBackendCreation:
    """Tests for TorchBackend creation."""

    def test_torch_backend_exists(self):
        """Test TorchBackend class exists."""
        from chuk_lazarus.models_v2.core.backend import TorchBackend

        assert TorchBackend is not None

    def test_torch_backend_creation(self, torch_available):
        """Test TorchBackend creation."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        assert backend.name == BackendType.TORCH


class TestTorchBackendTensorCreation:
    """Tests for TorchBackend tensor creation."""

    def test_zeros(self, torch_available):
        """Test TorchBackend zeros."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        tensor = backend.zeros((2, 3))
        assert tensor.shape == (2, 3)

    def test_ones(self, torch_available):
        """Test TorchBackend ones."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        tensor = backend.ones((3, 4))
        assert tensor.shape == (3, 4)

    def test_randn(self, torch_available):
        """Test TorchBackend randn."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        tensor = backend.randn((2, 5))
        assert tensor.shape == (2, 5)

    def test_arange(self, torch_available):
        """Test TorchBackend arange."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        tensor = backend.arange(0, 10, 2)
        result = backend.to_numpy(tensor)
        assert np.allclose(result, np.arange(0, 10, 2))

    def test_from_numpy(self, torch_available):
        """Test TorchBackend from_numpy."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = backend.from_numpy(arr)
        assert tensor.shape == (2, 2)


class TestTorchBackendOperations:
    """Tests for TorchBackend operations."""

    def test_matmul(self, torch_available):
        """Test TorchBackend matmul."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        a = backend.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = backend.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(c), expected)

    def test_softmax(self, torch_available):
        """Test TorchBackend softmax."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        result = backend.softmax(x, axis=-1)
        result_np = backend.to_numpy(result)
        assert np.allclose(result_np.sum(), 1.0, atol=1e-5)


class TestTorchBackendActivations:
    """Tests for TorchBackend activations."""

    def test_activations(self, torch_available):
        """Test TorchBackend activations."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.from_numpy(np.array([0.0, 1.0, 2.0], dtype=np.float32))

        assert backend.relu(x).shape == (3,)
        assert backend.silu(x).shape == (3,)
        assert backend.gelu(x).shape == (3,)
        assert backend.tanh(x).shape == (3,)
        assert backend.sigmoid(x).shape == (3,)


class TestTorchBackendNormalization:
    """Tests for TorchBackend normalization."""

    def test_norms(self, torch_available):
        """Test TorchBackend normalization."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))
        bias = backend.zeros((4,))

        result = backend.layer_norm(x, weight, bias, eps=1e-5)
        assert result.shape == (1, 4)

        result = backend.rms_norm(x, weight, eps=1e-5)
        assert result.shape == (1, 4)


class TestTorchBackendShapeOperations:
    """Tests for TorchBackend shape operations."""

    def test_reshape_transpose(self, torch_available):
        """Test TorchBackend reshape and transpose."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

        reshaped = backend.reshape(x, (6, 4))
        assert reshaped.shape == (6, 4)

        transposed = backend.transpose(x, (0, 2, 1))
        assert transposed.shape == (2, 4, 3)

    def test_concat_split(self, torch_available):
        """Test TorchBackend concatenate and split."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))

        concatenated = backend.concatenate([a, b], axis=0)
        assert concatenated.shape == (4, 3)

        x = backend.from_numpy(np.arange(12, dtype=np.float32).reshape(4, 3))
        parts = backend.split(x, 2, axis=0)
        assert len(parts) == 2
        assert parts[0].shape == (2, 3)


class TestTorchBackendAttention:
    """Tests for TorchBackend attention."""

    def test_attention(self, torch_available):
        """Test TorchBackend attention."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        batch, heads, seq, dim = 1, 2, 4, 8
        q = backend.randn((batch, heads, seq, dim))
        k = backend.randn((batch, heads, seq, dim))
        v = backend.randn((batch, heads, seq, dim))

        result = backend.scaled_dot_product_attention(q, k, v)
        assert result.shape == (batch, heads, seq, dim)

    def test_causal_mask(self, torch_available):
        """Test TorchBackend causal mask."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        mask = backend.create_causal_mask(5)
        assert mask.shape == (5, 5)


class TestTorchBackendGradients:
    """Tests for TorchBackend gradient operations."""

    def test_stop_gradient(self, torch_available):
        """Test TorchBackend stop_gradient."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.randn((2, 3))
        result = backend.stop_gradient(x)
        assert result.shape == (2, 3)

    def test_eval(self, torch_available):
        """Test TorchBackend eval (no-op for eager)."""
        if not torch_available:
            pytest.skip("PyTorch not installed")

        from chuk_lazarus.models_v2.core.backend import TorchBackend

        backend = TorchBackend()
        x = backend.randn((2, 3))
        backend.eval(x)  # Should not raise
