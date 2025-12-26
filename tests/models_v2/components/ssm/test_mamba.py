"""
Tests for Mamba layer and MambaBlock.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.ssm import (
    Mamba,
    MambaBlock,
)
from chuk_lazarus.models_v2.core.config import SSMConfig


class TestMamba:
    """Tests for Mamba layer (wrapper around SelectiveSSM)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        mamba = Mamba(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = mx.random.normal((2, 10, 256))
        output, cache = mamba(x)

        assert output.shape == (2, 10, 256)

    def test_from_config(self):
        """Test creating from SSMConfig."""
        config = SSMConfig(
            hidden_size=512,
            state_size=32,
            conv_kernel_size=4,
            expand_factor=2,
        )
        mamba = Mamba.from_config(config, d_model=512)

        x = mx.random.normal((1, 5, 512))
        output, _ = mamba(x)

        assert output.shape == (1, 5, 512)


class TestMambaBlock:
    """Tests for MambaBlock (Mamba + residual + norm)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MambaBlock(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = mx.random.normal((2, 10, 256))
        output, cache = block(x)

        assert output.shape == (2, 10, 256)

    def test_residual_connection(self):
        """Test residual connection is applied."""
        block = MambaBlock(
            d_model=128,
            d_state=16,
        )

        x = mx.zeros((1, 5, 128))
        output, _ = block(x)

        # With zero input, output should be close to zero due to residual
        assert mx.mean(mx.abs(output)).item() < 1.0

    def test_init_cache(self):
        """Test cache initialization."""
        block = MambaBlock(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        batch_size = 4
        cache = block.init_cache(batch_size)

        conv_state, ssm_state = cache
        # MambaBlock has .mamba attribute which wraps SelectiveSSM
        d_inner = block.mamba.d_model * block.mamba.expand
        d_conv = block.mamba.d_conv
        d_state = block.mamba.d_state

        assert conv_state.shape == (batch_size, d_inner, d_conv - 1)
        assert ssm_state.shape == (batch_size, d_inner, d_state)


class TestMambaBlockGradients:
    """Tests for gradient flow through MambaBlock."""

    def test_mamba_block_gradients(self):
        """Test gradients flow through MambaBlock."""
        block = MambaBlock(
            d_model=64,
            d_state=8,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(block, loss_fn)
        loss, grads = loss_and_grad_fn(block, x)

        assert loss.item() > 0
