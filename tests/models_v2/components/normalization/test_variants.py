"""
Tests for normalization variants (GemmaNorm).
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.normalization.variants import (
    GemmaNorm,
    create_gemma_norm,
)


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


class TestGemmaNormGradients:
    """Tests for gradient flow through GemmaNorm."""

    def test_gemmanorm_gradients(self):
        """Test gradients flow through GemmaNorm."""
        norm = GemmaNorm(dims=64)
        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(norm, loss_fn)
        loss, grads = loss_and_grad_fn(norm, x)
        assert loss.item() > 0
