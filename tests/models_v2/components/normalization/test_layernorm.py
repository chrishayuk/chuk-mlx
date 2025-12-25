"""
Tests for LayerNorm.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.normalization.layernorm import (
    LayerNorm,
    create_layernorm,
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


class TestLayerNormGradients:
    """Tests for gradient flow through LayerNorm."""

    def test_layernorm_gradients(self):
        """Test gradients flow through LayerNorm."""
        norm = LayerNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(norm, x)
        assert loss.item() > 0


class TestLayerNormNumericalStability:
    """Tests for numerical stability of LayerNorm."""

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
