"""
Tests for SSM components.

Tests SelectiveSSM, Mamba layer, and MambaBlock.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.ssm import (
    Mamba,
    MambaBlock,
    SelectiveSSM,
    selective_scan,
    selective_scan_step,
)
from chuk_lazarus.models_v2.core.config import SSMConfig


class TestSelectiveSSM:
    """Tests for SelectiveSSM."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        ssm = SelectiveSSM(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = mx.random.normal((2, 10, 256))
        output, cache = ssm(x)

        assert output.shape == (2, 10, 256)
        assert cache is None  # No cache in training mode

    def test_inner_dimension(self):
        """Test inner dimension expansion."""
        ssm = SelectiveSSM(
            d_model=256,
            d_state=16,
            expand=2,
        )

        assert ssm.d_inner == 512  # 256 * 2

    def test_dt_rank_auto(self):
        """Test automatic dt_rank calculation."""
        ssm = SelectiveSSM(
            d_model=768,
            dt_rank="auto",
        )

        expected_rank = max(1, 768 // 16)
        assert ssm.dt_rank == expected_rank

    def test_dt_rank_explicit(self):
        """Test explicit dt_rank."""
        ssm = SelectiveSSM(
            d_model=256,
            dt_rank=32,
        )

        assert ssm.dt_rank == 32

    def test_with_cache(self):
        """Test SSM with cache for inference."""
        ssm = SelectiveSSM(
            d_model=128,
            d_state=16,
            d_conv=4,
        )

        batch_size = 2
        d_inner = ssm.d_inner

        # Create initial cache
        conv_state = mx.zeros((batch_size, ssm.d_conv - 1, d_inner))
        ssm_state = mx.zeros((batch_size, d_inner, ssm.d_state))
        cache = (conv_state, ssm_state)

        x = mx.random.normal((batch_size, 1, 128))
        output, new_cache = ssm(x, cache=cache)

        assert output.shape == (batch_size, 1, 128)
        assert new_cache is not None
        assert len(new_cache) == 2  # (conv_state, ssm_state)

    def test_different_state_sizes(self):
        """Test different state sizes."""
        for d_state in [8, 16, 32, 64]:
            ssm = SelectiveSSM(
                d_model=128,
                d_state=d_state,
            )

            x = mx.random.normal((1, 5, 128))
            output, _ = ssm(x)

            assert output.shape == (1, 5, 128)

    def test_different_conv_widths(self):
        """Test different convolution widths."""
        for d_conv in [2, 4, 8]:
            ssm = SelectiveSSM(
                d_model=128,
                d_conv=d_conv,
            )

            x = mx.random.normal((1, 10, 128))
            output, _ = ssm(x)

            assert output.shape == (1, 10, 128)


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


class TestSelectiveScan:
    """Tests for selective_scan function."""

    def test_basic_scan(self):
        """Test basic selective scan."""
        batch, seq_len, d_inner = 2, 10, 64
        d_state = 16

        x = mx.random.normal((batch, seq_len, d_inner))
        dt = mx.abs(mx.random.normal((batch, seq_len, d_inner))) + 0.01
        A = -mx.abs(mx.random.normal((d_inner, d_state)))
        B = mx.random.normal((batch, seq_len, d_state))
        C = mx.random.normal((batch, seq_len, d_state))
        D = mx.ones((d_inner,))

        y = selective_scan(x, dt, A, B, C, D)

        assert y.shape == (batch, seq_len, d_inner)

    def test_skip_connection(self):
        """Test D adds skip connection."""
        batch, seq_len, d_inner = 1, 5, 32
        d_state = 8

        x = mx.random.normal((batch, seq_len, d_inner))
        dt = mx.abs(mx.random.normal((batch, seq_len, d_inner))) + 0.01
        A = -mx.abs(mx.random.normal((d_inner, d_state)))
        B = mx.random.normal((batch, seq_len, d_state))
        C = mx.random.normal((batch, seq_len, d_state))
        D = mx.ones((d_inner,))

        y = selective_scan(x, dt, A, B, C, D)

        # Output should be influenced by x through D
        assert y.shape == x.shape


class TestSelectiveScanStep:
    """Tests for selective_scan_step function."""

    def test_single_step(self):
        """Test single step of selective scan."""
        batch, d_inner = 2, 64
        d_state = 16

        x = mx.random.normal((batch, 1, d_inner))
        dt = mx.abs(mx.random.normal((batch, 1, d_inner))) + 0.01
        A = -mx.abs(mx.random.normal((d_inner, d_state)))
        B = mx.random.normal((batch, 1, d_state))
        C = mx.random.normal((batch, 1, d_state))
        D = mx.ones((d_inner,))
        ssm_state = mx.zeros((batch, d_inner, d_state))

        y, new_state = selective_scan_step(x, dt, A, B, C, D, ssm_state)

        assert y.shape == (batch, 1, d_inner)
        assert new_state.shape == (batch, d_inner, d_state)

    def test_state_update(self):
        """Test state is updated."""
        batch, d_inner = 1, 32
        d_state = 8

        x = mx.random.normal((batch, 1, d_inner))
        dt = mx.abs(mx.random.normal((batch, 1, d_inner))) + 0.01
        A = -mx.abs(mx.random.normal((d_inner, d_state)))
        B = mx.random.normal((batch, 1, d_state))
        C = mx.random.normal((batch, 1, d_state))
        D = mx.ones((d_inner,))
        ssm_state = mx.zeros((batch, d_inner, d_state))

        _, new_state = selective_scan_step(x, dt, A, B, C, D, ssm_state)

        # State should be updated (not all zeros anymore)
        assert mx.mean(mx.abs(new_state)).item() > 0


class TestSSMGradients:
    """Tests for gradient flow through SSM."""

    def test_ssm_gradients(self):
        """Test gradients flow through SelectiveSSM."""
        ssm = SelectiveSSM(
            d_model=64,
            d_state=8,
            d_conv=4,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(ssm, x)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

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

        loss, grads = mx.value_and_grad(loss_fn)(block, x)

        assert loss.item() > 0


class TestSSMBatchHandling:
    """Tests for batch dimension handling."""

    def test_different_batch_sizes(self):
        """Test SSM handles different batch sizes."""
        ssm = SelectiveSSM(
            d_model=128,
            d_state=16,
        )

        for batch_size in [1, 2, 4]:
            x = mx.random.normal((batch_size, 5, 128))
            output, _ = ssm(x)
            assert output.shape == (batch_size, 5, 128)

    def test_different_sequence_lengths(self):
        """Test SSM handles different sequence lengths."""
        ssm = SelectiveSSM(
            d_model=128,
            d_state=16,
        )

        for seq_len in [1, 5, 10, 50]:
            x = mx.random.normal((2, seq_len, 128))
            output, _ = ssm(x)
            assert output.shape == (2, seq_len, 128)

    def test_single_token(self):
        """Test SSM handles single token input."""
        ssm = SelectiveSSM(
            d_model=128,
            d_state=16,
        )

        x = mx.random.normal((1, 1, 128))
        output, _ = ssm(x)
        assert output.shape == (1, 1, 128)
