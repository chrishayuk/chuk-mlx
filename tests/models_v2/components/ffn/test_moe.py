"""
Tests for Mixture of Experts (MoE).
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.components.ffn import SwiGLU
from chuk_lazarus.models_v2.components.ffn.moe import (
    MoE,
    MoERouter,
    create_moe,
)
from chuk_lazarus.models_v2.core.config import FFNConfig


class TestMoERouter:
    """Tests for MoE router."""

    def test_basic_creation(self):
        """Test basic router creation."""
        router = MoERouter(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2,
        )
        assert router.num_experts == 8
        assert router.num_experts_per_tok == 2

    def test_forward(self):
        """Test router forward pass."""
        router = MoERouter(
            hidden_size=256,
            num_experts=8,
            num_experts_per_tok=2,
        )
        x = mx.random.normal((2, 10, 256))
        weights, indices = router(x)

        assert weights.shape == (2, 10, 2)  # top-k weights
        assert indices.shape == (2, 10, 2)  # top-k indices

    def test_weights_normalized(self):
        """Test that routing weights sum to approximately 1."""
        router = MoERouter(
            hidden_size=128,
            num_experts=4,
            num_experts_per_tok=2,
        )
        x = mx.random.normal((2, 10, 128))
        weights, _ = router(x)

        # Weights should sum to approximately 1 for each token
        weight_sums = mx.sum(weights, axis=-1)
        assert mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-4)

    def test_indices_valid(self):
        """Test that indices are valid expert indices."""
        num_experts = 8
        router = MoERouter(
            hidden_size=128,
            num_experts=num_experts,
            num_experts_per_tok=2,
        )
        x = mx.random.normal((2, 10, 128))
        _, indices = router(x)

        # All indices should be in valid range
        assert mx.all(indices >= 0)
        assert mx.all(indices < num_experts)


class TestMoE:
    """Tests for Mixture of Experts."""

    def test_basic_creation(self):
        """Test basic MoE creation."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
        )
        moe = MoE(config)
        assert moe.num_experts == 4
        assert moe.num_experts_per_tok == 2

    def test_forward(self):
        """Test MoE forward pass."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=2,
        )
        moe = MoE(config)
        x = mx.random.normal((2, 10, 256))
        output = moe(x)

        assert output.shape == (2, 10, 256)

    def test_single_token(self):
        """Test MoE with single token."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
        )
        moe = MoE(config)
        x = mx.random.normal((1, 1, 128))
        output = moe(x)

        assert output.shape == (1, 1, 128)

    def test_different_batch_sizes(self):
        """Test MoE with different batch sizes."""
        config = FFNConfig(
            hidden_size=128,
            intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
        )
        moe = MoE(config)

        for batch_size in [1, 2, 4]:
            x = mx.random.normal((batch_size, 5, 128))
            output = moe(x)
            assert output.shape == (batch_size, 5, 128)

    def test_missing_num_experts_raises(self):
        """Test that missing num_experts raises error."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            num_experts=None,
            num_experts_per_tok=2,
        )
        with pytest.raises(ValueError, match="num_experts must be set"):
            MoE(config)

    def test_missing_num_experts_per_tok_raises(self):
        """Test that missing num_experts_per_tok raises error."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_experts_per_tok=None,
        )
        with pytest.raises(ValueError, match="num_experts_per_tok must be set"):
            MoE(config)

    def test_factory_function(self):
        """Test MoE factory function."""
        moe = create_moe(
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
        )
        assert isinstance(moe, MoE)
        assert moe.num_experts == 8


class TestFFNBatchHandling:
    """Tests for batch dimension handling."""

    def test_different_batch_sizes(self):
        """Test FFN handles different batch sizes."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = SwiGLU(config)

        for batch_size in [1, 2, 4, 8]:
            x = mx.random.normal((batch_size, 10, 256))
            output = ffn(x)
            assert output.shape == (batch_size, 10, 256)

    def test_different_sequence_lengths(self):
        """Test FFN handles different sequence lengths."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = SwiGLU(config)

        for seq_len in [1, 10, 100, 512]:
            x = mx.random.normal((2, seq_len, 256))
            output = ffn(x)
            assert output.shape == (2, seq_len, 256)

    def test_single_token(self):
        """Test FFN handles single token input."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
        )
        ffn = SwiGLU(config)

        x = mx.random.normal((1, 1, 256))
        output = ffn(x)
        assert output.shape == (1, 1, 256)
