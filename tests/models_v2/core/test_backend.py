"""
Tests for backend abstraction module.

Tests MLXBackend, TorchBackend (mocked), and backend management functions.
"""

import numpy as np
import pytest

from chuk_lazarus.models_v2.core.backend import (
    Backend,
    BackendType,
    MLXBackend,
    get_backend,
    reset_backend,
    set_backend,
)


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


class TestMLXBackend:
    """Tests for MLXBackend."""

    def test_creation(self):
        """Test MLXBackend creation."""
        backend = MLXBackend()
        assert backend.name == BackendType.MLX
        assert backend.device == "mps"

    def test_zeros(self):
        """Test zeros creation."""
        backend = MLXBackend()
        tensor = backend.zeros((2, 3))
        assert tensor.shape == (2, 3)
        assert np.allclose(backend.to_numpy(tensor), np.zeros((2, 3)))

    def test_ones(self):
        """Test ones creation."""
        backend = MLXBackend()
        tensor = backend.ones((3, 4))
        assert tensor.shape == (3, 4)
        assert np.allclose(backend.to_numpy(tensor), np.ones((3, 4)))

    def test_randn(self):
        """Test random normal creation."""
        backend = MLXBackend()
        tensor = backend.randn((2, 5))
        assert tensor.shape == (2, 5)

    def test_arange(self):
        """Test arange creation."""
        backend = MLXBackend()
        tensor = backend.arange(0, 10, 2)
        result = backend.to_numpy(tensor)
        assert np.allclose(result, np.arange(0, 10, 2))

    def test_from_numpy(self):
        """Test numpy to tensor conversion."""
        backend = MLXBackend()
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = backend.from_numpy(arr)
        assert tensor.shape == (2, 2)
        assert np.allclose(backend.to_numpy(tensor), arr)

    def test_matmul(self):
        """Test matrix multiplication."""
        backend = MLXBackend()
        a = backend.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = backend.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(backend.to_numpy(c), expected)

    def test_softmax(self):
        """Test softmax."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        result = backend.softmax(x, axis=-1)
        result_np = backend.to_numpy(result)
        assert np.allclose(result_np.sum(), 1.0, atol=1e-5)

    def test_relu(self):
        """Test ReLU activation."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        result = backend.relu(x)
        expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert np.allclose(backend.to_numpy(result), expected)

    def test_silu(self):
        """Test SiLU activation."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        result = backend.silu(x)
        assert result.shape == (3,)

    def test_gelu(self):
        """Test GELU activation."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        result = backend.gelu(x)
        assert result.shape == (3,)

    def test_tanh(self):
        """Test tanh activation."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        result = backend.tanh(x)
        expected = np.tanh(np.array([0.0, 1.0, -1.0]))
        assert np.allclose(backend.to_numpy(result), expected, atol=1e-5)

    def test_sigmoid(self):
        """Test sigmoid activation."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        result = backend.sigmoid(x)
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0])))
        assert np.allclose(backend.to_numpy(result), expected, atol=1e-5)

    def test_layer_norm(self):
        """Test layer normalization."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))
        bias = backend.zeros((4,))
        result = backend.layer_norm(x, weight, bias, eps=1e-5)
        result_np = backend.to_numpy(result)
        # Check that output is normalized (mean ~0, std ~1)
        assert np.abs(result_np.mean()) < 1e-5
        assert np.abs(result_np.std() - 1.0) < 0.1

    def test_layer_norm_no_bias(self):
        """Test layer normalization without bias."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))
        result = backend.layer_norm(x, weight, None, eps=1e-5)
        assert result.shape == (1, 4)

    def test_rms_norm(self):
        """Test RMS normalization."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        weight = backend.ones((4,))
        result = backend.rms_norm(x, weight, eps=1e-5)
        assert result.shape == (1, 4)

    def test_reshape(self):
        """Test reshape."""
        backend = MLXBackend()
        x = backend.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        result = backend.reshape(x, (3, 2))
        assert result.shape == (3, 2)

    def test_transpose(self):
        """Test transpose."""
        backend = MLXBackend()
        x = backend.from_numpy(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        result = backend.transpose(x, (0, 2, 1))
        assert result.shape == (2, 4, 3)

    def test_concatenate(self):
        """Test concatenate."""
        backend = MLXBackend()
        a = backend.ones((2, 3))
        b = backend.zeros((2, 3))
        result = backend.concatenate([a, b], axis=0)
        assert result.shape == (4, 3)

    def test_split(self):
        """Test split."""
        backend = MLXBackend()
        x = backend.from_numpy(np.arange(12, dtype=np.float32).reshape(4, 3))
        parts = backend.split(x, 2, axis=0)
        assert len(parts) == 2
        assert parts[0].shape == (2, 3)

    def test_scaled_dot_product_attention(self):
        """Test SDPA."""
        backend = MLXBackend()
        batch, heads, seq, dim = 1, 2, 4, 8
        q = backend.randn((batch, heads, seq, dim))
        k = backend.randn((batch, heads, seq, dim))
        v = backend.randn((batch, heads, seq, dim))
        result = backend.scaled_dot_product_attention(q, k, v)
        assert result.shape == (batch, heads, seq, dim)

    def test_scaled_dot_product_attention_with_mask(self):
        """Test SDPA with mask."""
        backend = MLXBackend()
        batch, heads, seq, dim = 1, 2, 4, 8
        q = backend.randn((batch, heads, seq, dim))
        k = backend.randn((batch, heads, seq, dim))
        v = backend.randn((batch, heads, seq, dim))
        mask = backend.create_causal_mask(seq)
        result = backend.scaled_dot_product_attention(q, k, v, mask=mask)
        assert result.shape == (batch, heads, seq, dim)

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        backend = MLXBackend()
        mask = backend.create_causal_mask(5)
        # Should be lower triangular (masked upper)
        assert mask.shape == (5, 5)

    def test_stop_gradient(self):
        """Test stop gradient."""
        backend = MLXBackend()
        x = backend.randn((2, 3))
        result = backend.stop_gradient(x)
        assert result.shape == (2, 3)

    def test_eval(self):
        """Test eval (force computation)."""
        backend = MLXBackend()
        x = backend.randn((2, 3))
        backend.eval(x)  # Should not raise


class TestBackendManagement:
    """Tests for backend management functions."""

    def test_get_backend_default(self):
        """Test getting default backend."""
        reset_backend()
        backend = get_backend()
        assert isinstance(backend, Backend)
        # On macOS, should default to MLX
        import platform

        if platform.system() == "Darwin":
            assert backend.name == BackendType.MLX

    def test_set_backend_instance(self):
        """Test setting backend with instance."""
        reset_backend()
        mlx_backend = MLXBackend()
        set_backend(mlx_backend)
        assert get_backend() is mlx_backend

    def test_set_backend_string(self):
        """Test setting backend with string."""
        reset_backend()
        set_backend("mlx")
        assert get_backend().name == BackendType.MLX

    def test_set_backend_enum(self):
        """Test setting backend with enum."""
        reset_backend()
        set_backend(BackendType.MLX)
        assert get_backend().name == BackendType.MLX

    def test_set_backend_invalid_type(self):
        """Test setting backend with invalid type."""
        reset_backend()
        with pytest.raises(TypeError):
            set_backend(123)

    def test_reset_backend(self):
        """Test resetting backend."""
        reset_backend()
        # Get a backend
        _ = get_backend()
        # Reset
        reset_backend()
        # Get again - should get a new instance
        backend2 = get_backend()
        # They may be the same type but should be fresh instances
        assert isinstance(backend2, Backend)


class TestBackendAbstract:
    """Tests for Backend abstract base class."""

    def test_cannot_instantiate(self):
        """Test that Backend cannot be directly instantiated."""
        with pytest.raises(TypeError):
            Backend()
