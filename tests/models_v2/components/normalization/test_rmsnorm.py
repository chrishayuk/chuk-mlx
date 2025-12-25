"""
Tests for RMSNorm.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.normalization import RMSNorm
from chuk_lazarus.models_v2.components.normalization.rmsnorm import create_rmsnorm


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_basic_creation(self):
        """Test basic creation."""
        norm = RMSNorm(dims=256)
        assert norm.dims == 256

    def test_forward(self):
        """Test forward pass."""
        norm = RMSNorm(dims=256)
        x = mx.random.normal((2, 10, 256))
        output = norm(x)
        assert output.shape == (2, 10, 256)

    def test_rms_normalization(self):
        """Test that RMS normalization is applied correctly."""
        norm = RMSNorm(dims=64, eps=1e-6)

        # Override weight to be ones for testing
        norm.weight = mx.ones((64,))

        x = mx.ones((1, 1, 64)) * 2.0  # All twos

        output = norm(x)

        # RMS of all 2s is 2, so output should be 2/2 = 1
        assert mx.allclose(output, mx.ones_like(output), atol=1e-4)

    def test_custom_eps(self):
        """Test custom epsilon."""
        norm = RMSNorm(dims=64, eps=1e-8)
        assert norm.eps == 1e-8

    def test_factory_function(self):
        """Test factory function."""
        norm = create_rmsnorm(dims=512, eps=1e-6)
        assert isinstance(norm, RMSNorm)
        assert norm.dims == 512

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        norm = RMSNorm(dims=128)
        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 128))
            output = norm(x)
            assert output.shape == (batch_size, 10, 128)

    def test_different_seq_lengths(self):
        """Test with different sequence lengths."""
        norm = RMSNorm(dims=128)
        for seq_len in [1, 5, 10, 50, 100]:
            x = mx.random.normal((2, seq_len, 128))
            output = norm(x)
            assert output.shape == (2, seq_len, 128)


class TestRMSNormGradients:
    """Tests for gradient flow through RMSNorm."""

    def test_rmsnorm_gradients(self):
        """Test gradients flow through RMSNorm."""
        norm = RMSNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(norm, loss_fn)
        loss, grads = loss_and_grad_fn(norm, x)
        assert loss.item() > 0


class TestRMSNormNumericalStability:
    """Tests for numerical stability of RMSNorm."""

    def test_rmsnorm_small_values(self):
        """Test RMSNorm with very small values."""
        norm = RMSNorm(dims=64, eps=1e-6)
        x = mx.random.normal((2, 10, 64)) * 1e-6
        output = norm(x)
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_rmsnorm_large_values(self):
        """Test RMSNorm with large values."""
        norm = RMSNorm(dims=64, eps=1e-6)
        x = mx.random.normal((2, 10, 64)) * 1e6
        output = norm(x)
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))
