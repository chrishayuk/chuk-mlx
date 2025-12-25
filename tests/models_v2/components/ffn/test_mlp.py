"""
Tests for standard MLP.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.ffn import MLP
from chuk_lazarus.models_v2.components.ffn.mlp import create_mlp
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType, FFNType


class TestMLP:
    """Tests for standard MLP."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=512,
            intermediate_size=2048,
        )
        ffn = MLP(config)

        x = mx.random.normal((2, 10, 512))
        output = ffn(x)

        assert output.shape == (2, 10, 512)

    def test_with_bias(self):
        """Test MLP with bias enabled."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=256,
            intermediate_size=1024,
            bias=True,
        )
        ffn = MLP(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)

    def test_relu_activation(self):
        """Test MLP with ReLU activation."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=128,
            intermediate_size=512,
            activation=ActivationType.RELU,
        )
        ffn = MLP(config)

        x = mx.random.normal((1, 5, 128))
        output = ffn(x)

        assert output.shape == (1, 5, 128)


class TestMLPFactory:
    """Tests for MLP factory function."""

    def test_create_mlp(self):
        """Test MLP factory function."""
        ffn = create_mlp(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, MLP)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

    def test_create_mlp_directly(self):
        """Test creating MLP directly."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=512,
            intermediate_size=2048,
        )
        ffn = MLP(config)

        assert isinstance(ffn, MLP)


class TestMLPGradients:
    """Tests for gradient flow through MLP."""

    def test_mlp_gradients(self):
        """Test gradients flow through MLP."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=128,
            intermediate_size=512,
        )
        ffn = MLP(config)

        x = mx.random.normal((1, 5, 128))

        def loss_fn(model, x):
            return mx.mean(model(x) ** 2)

        loss_and_grad_fn = nn.value_and_grad(ffn, loss_fn)
        loss, grads = loss_and_grad_fn(ffn, x)

        assert loss.item() > 0
