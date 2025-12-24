"""
Tests for normalization components.

Tests LayerNorm, RMSNorm, and GemmaNorm.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.normalization import RMSNorm
from chuk_lazarus.models_v2.components.normalization.layernorm import (
    LayerNorm,
    create_layernorm,
)
from chuk_lazarus.models_v2.components.normalization.rmsnorm import create_rmsnorm
from chuk_lazarus.models_v2.components.normalization.variants import (
    GemmaNorm,
    create_gemma_norm,
)


class TestLayerNorm:
    """Tests for LayerNorm."""

    def test_basic_creation(self):
        """Test basic creation."""
        norm = LayerNorm(dims=256)
        assert norm.dims == 256
        assert norm.affine is True

    def test_forward(self):
        """Test forward pass."""
        norm = LayerNorm(dims=256)
        x = mx.random.normal((2, 10, 256))
        output = norm(x)
        assert output.shape == (2, 10, 256)

    def test_normalized_mean_and_var(self):
        """Test that output has approximately zero mean and unit variance."""
        norm = LayerNorm(dims=64, affine=False)
        x = mx.random.normal((4, 20, 64)) * 5 + 10  # Non-zero mean, non-unit var

        output = norm(x)

        # Check mean is close to 0
        mean = mx.mean(output, axis=-1)
        assert mx.all(mx.abs(mean) < 0.01)

        # Check variance is close to 1
        var = mx.var(output, axis=-1)
        assert mx.all(mx.abs(var - 1.0) < 0.1)

    def test_without_affine(self):
        """Test LayerNorm without affine parameters."""
        norm = LayerNorm(dims=128, affine=False)
        assert norm.weight is None
        assert norm.bias is None

        x = mx.random.normal((2, 10, 128))
        output = norm(x)
        assert output.shape == (2, 10, 128)

    def test_with_affine(self):
        """Test LayerNorm with affine parameters."""
        norm = LayerNorm(dims=128, affine=True)
        assert norm.weight is not None
        assert norm.bias is not None

        x = mx.random.normal((2, 10, 128))
        output = norm(x)
        assert output.shape == (2, 10, 128)

    def test_custom_eps(self):
        """Test custom epsilon."""
        norm = LayerNorm(dims=64, eps=1e-6)
        assert norm.eps == 1e-6

    def test_factory_function(self):
        """Test factory function."""
        norm = create_layernorm(dims=512, eps=1e-5, affine=True)
        assert isinstance(norm, LayerNorm)
        assert norm.dims == 512


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


class TestGemmaNorm:
    """Tests for GemmaNorm (RMSNorm with +1 offset)."""

    def test_basic_creation(self):
        """Test basic creation."""
        norm = GemmaNorm(dims=256)
        assert norm.dims == 256

    def test_forward(self):
        """Test forward pass."""
        norm = GemmaNorm(dims=256)
        x = mx.random.normal((2, 10, 256))
        output = norm(x)
        assert output.shape == (2, 10, 256)

    def test_weight_offset(self):
        """Test that weight has +1 offset applied."""
        norm = GemmaNorm(dims=64)

        # Default weight is zeros, so effective weight is 1
        x = mx.ones((1, 1, 64)) * 2.0
        output = norm(x)

        # With weight=0, effective weight is 1, so output = x / rms
        # RMS of all 2s is 2, so output should be (1+0) * 2/2 = 1
        assert mx.allclose(output, mx.ones_like(output), atol=1e-4)

    def test_custom_eps(self):
        """Test custom epsilon."""
        norm = GemmaNorm(dims=64, eps=1e-8)
        assert norm.eps == 1e-8

    def test_factory_function(self):
        """Test factory function."""
        norm = create_gemma_norm(dims=512, eps=1e-6)
        assert isinstance(norm, GemmaNorm)
        assert norm.dims == 512


class TestNormalizationGradients:
    """Tests for gradient flow through normalization layers."""

    def test_layernorm_gradients(self):
        """Test gradients flow through LayerNorm."""
        norm = LayerNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(norm, x)
        assert loss.item() > 0

    def test_rmsnorm_gradients(self):
        """Test gradients flow through RMSNorm."""
        norm = RMSNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(norm, x)
        assert loss.item() > 0

    def test_gemmanorm_gradients(self):
        """Test gradients flow through GemmaNorm."""
        norm = GemmaNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(norm, x)
        assert loss.item() > 0


class TestNormalizationNumericalStability:
    """Tests for numerical stability of normalization layers."""

    def test_layernorm_small_values(self):
        """Test LayerNorm with very small values."""
        norm = LayerNorm(dims=64, eps=1e-5)
        x = mx.random.normal((2, 10, 64)) * 1e-6
        output = norm(x)
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_layernorm_large_values(self):
        """Test LayerNorm with large values."""
        norm = LayerNorm(dims=64, eps=1e-5)
        x = mx.random.normal((2, 10, 64)) * 1e6
        output = norm(x)
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

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
