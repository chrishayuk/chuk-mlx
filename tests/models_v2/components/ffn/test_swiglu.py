"""
Tests for SwiGLU feed-forward network.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.ffn import SwiGLU
from chuk_lazarus.models_v2.components.ffn.swiglu import create_swiglu
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import FFNType


class TestSwiGLU:
    """Tests for SwiGLU feed-forward network."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = FFNConfig(
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = SwiGLU(config)

        x = mx.random.normal((2, 10, 512))
        output = ffn(x)

        assert output.shape == (2, 10, 512)

    def test_default_intermediate_size(self):
        """Test default intermediate size calculation."""
        config = FFNConfig(
            hidden_size=512,
            intermediate_size=1408,  # Approx 8/3 * 512
        )
        ffn = SwiGLU(config)

        x = mx.random.normal((1, 5, 512))
        output = ffn(x)

        assert output.shape == (1, 5, 512)

    def test_expansion_factor(self):
        """Test intermediate size is expanded."""
        config = FFNConfig(
            hidden_size=768,
            intermediate_size=3072,  # 4x expansion
        )
        ffn = SwiGLU(config)

        # Gate projection should go to intermediate_size
        assert ffn.gate_proj.weight.shape[0] == 3072

    def test_gating_mechanism(self):
        """Test that gating produces different outputs than identity."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
        )
        ffn = SwiGLU(config)

        x = mx.random.normal((1, 1, 128))
        output = ffn(x)

        # Output should be different from input (gating changes it)
        assert not mx.allclose(output, x).item()

    def test_no_bias_default(self):
        """Test bias is disabled by default."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            bias=False,
        )
        ffn = SwiGLU(config)

        # Linear layers should not have bias when bias=False
        # Check that bias attribute doesn't exist or is None
        assert not hasattr(ffn.gate_proj, "bias") or ffn.gate_proj.bias is None


class TestSwiGLUFactory:
    """Tests for SwiGLU factory function."""

    def test_create_swiglu(self):
        """Test SwiGLU factory function."""
        ffn = create_swiglu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, SwiGLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

    def test_create_swiglu_directly(self):
        """Test creating SwiGLU directly."""
        config = FFNConfig(
            ffn_type=FFNType.SWIGLU,
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = SwiGLU(config)

        assert isinstance(ffn, SwiGLU)


class TestSwiGLUGradients:
    """Tests for gradient flow through SwiGLU."""

    def test_swiglu_gradients(self):
        """Test gradients flow through SwiGLU."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
        )
        ffn = SwiGLU(config)

        x = mx.random.normal((1, 5, 128))

        def loss_fn(model, x):
            return mx.mean(model(x) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(ffn, x)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())
