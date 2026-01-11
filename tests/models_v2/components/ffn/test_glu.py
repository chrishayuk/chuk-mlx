"""
Tests for GLU (Gated Linear Unit) feed-forward network.

Tests the unified GLU class which supports different activations:
- SwiGLU (SILU activation)
- GEGLU (GELU activation)
- Other gated variants
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.ffn import (
    GEGLU,
    GLU,
    SwiGLU,
    create_geglu,
    create_glu,
    create_swiglu,
)
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType


class TestGLU:
    """Tests for GLU feed-forward network."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = FFNConfig(
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = GLU(config)

        x = mx.random.normal((2, 10, 512))
        output = ffn(x)

        assert output.shape == (2, 10, 512)

    def test_default_silu_activation(self):
        """Test default activation is SILU (SwiGLU)."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.SILU,
        )
        ffn = GLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)

    def test_gelu_activation(self):
        """Test GELU activation (GEGLU)."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.GELU,
        )
        ffn = GLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)

    def test_relu_activation(self):
        """Test RELU activation (ReGLU)."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.RELU,
        )
        ffn = GLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)

    def test_expansion_factor(self):
        """Test intermediate size is expanded."""
        config = FFNConfig(
            hidden_size=768,
            intermediate_size=3072,  # 4x expansion
        )
        ffn = GLU(config)

        # Gate projection should go to intermediate_size
        assert ffn.gate_proj.weight.shape[0] == 3072

    def test_three_projections(self):
        """Test GLU has gate, up, and down projections."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = GLU(config)

        assert hasattr(ffn, "gate_proj")
        assert hasattr(ffn, "up_proj")
        assert hasattr(ffn, "down_proj")

    def test_gating_mechanism(self):
        """Test that gating produces different outputs than identity."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
        )
        ffn = GLU(config)

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
        ffn = GLU(config)

        # Linear layers should not have bias when bias=False
        assert not hasattr(ffn.gate_proj, "bias") or ffn.gate_proj.bias is None

    def test_with_bias(self):
        """Test GLU with bias enabled."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            bias=True,
        )
        ffn = GLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)


class TestGLUAliases:
    """Test that SwiGLU and GEGLU are aliases for GLU."""

    def test_swiglu_is_glu(self):
        """Test SwiGLU is an alias for GLU."""
        assert SwiGLU is GLU

    def test_geglu_is_glu(self):
        """Test GEGLU is an alias for GLU."""
        assert GEGLU is GLU


class TestGLUFactory:
    """Tests for GLU factory functions."""

    def test_create_glu(self):
        """Test GLU factory function."""
        ffn = create_glu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, GLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

    def test_create_glu_with_activation(self):
        """Test GLU factory with custom activation."""
        ffn = create_glu(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.GELU,
        )
        assert isinstance(ffn, GLU)

    def test_create_swiglu(self):
        """Test SwiGLU factory function."""
        ffn = create_swiglu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, GLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

    def test_create_geglu(self):
        """Test GEGLU factory function."""
        ffn = create_geglu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, GLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)


class TestGLUGradients:
    """Tests for gradient flow through GLU."""

    def test_glu_gradients(self):
        """Test gradients flow through GLU."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
        )
        ffn = GLU(config)

        x = mx.random.normal((1, 5, 128))

        def loss_fn(model, x):
            return mx.mean(model(x) ** 2)

        loss_and_grad_fn = nn.value_and_grad(ffn, loss_fn)
        loss, grads = loss_and_grad_fn(ffn, x)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_gradients_with_different_activations(self):
        """Test gradients flow with different activations."""
        for activation in [
            ActivationType.SILU,
            ActivationType.GELU,
            ActivationType.RELU,
        ]:
            config = FFNConfig(
                hidden_size=64,
                intermediate_size=128,
                activation=activation,
            )
            ffn = GLU(config)

            x = mx.random.normal((1, 3, 64))

            def loss_fn(model, x):
                return mx.mean(model(x) ** 2)

            loss_and_grad_fn = nn.value_and_grad(ffn, loss_fn)
            loss, grads = loss_and_grad_fn(ffn, x)

            assert loss.item() > 0


class TestGLUBatchHandling:
    """Tests for batch handling in GLU."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = GLU(config)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 256))
            output = ffn(x)
            assert output.shape == (batch_size, 10, 256)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = GLU(config)

        for seq_len in [1, 5, 10, 20]:
            x = mx.random.normal((2, seq_len, 256))
            output = ffn(x)
            assert output.shape == (2, seq_len, 256)
