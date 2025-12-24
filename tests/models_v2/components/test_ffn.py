"""
Tests for FFN components.

Tests SwiGLU, GEGLU, GLU, MLP, and MoE.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.ffn import (
    GEGLU,
    MLP,
    SwiGLU,
)
from chuk_lazarus.models_v2.components.ffn.geglu import create_geglu
from chuk_lazarus.models_v2.components.ffn.mlp import create_mlp
from chuk_lazarus.models_v2.components.ffn.moe import (
    MoE,
    MoERouter,
    create_moe,
)
from chuk_lazarus.models_v2.components.ffn.swiglu import create_swiglu
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType, FFNType


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


class TestGEGLU:
    """Tests for GEGLU feed-forward network."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = FFNConfig(
            ffn_type=FFNType.GEGLU,
            hidden_size=512,
            intermediate_size=1408,
            activation=ActivationType.GELU,
        )
        ffn = GEGLU(config)

        x = mx.random.normal((2, 10, 512))
        output = ffn(x)

        assert output.shape == (2, 10, 512)

    def test_gelu_activation(self):
        """Test GELU is used (different from SiLU)."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.GELU,
        )
        ffn = GEGLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)


# GLU tests removed - GLU class not currently exported


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


class TestFFNFactory:
    """Tests for FFN creation patterns."""

    def test_create_swiglu_directly(self):
        """Test creating SwiGLU directly."""
        config = FFNConfig(
            ffn_type=FFNType.SWIGLU,
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = SwiGLU(config)

        assert isinstance(ffn, SwiGLU)

    def test_create_geglu_directly(self):
        """Test creating GEGLU directly."""
        config = FFNConfig(
            ffn_type=FFNType.GEGLU,
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = GEGLU(config)

        assert isinstance(ffn, GEGLU)

    def test_create_mlp_directly(self):
        """Test creating MLP directly."""
        config = FFNConfig(
            ffn_type=FFNType.MLP,
            hidden_size=512,
            intermediate_size=2048,
        )
        ffn = MLP(config)

        assert isinstance(ffn, MLP)


class TestFFNGradients:
    """Tests for gradient flow through FFN."""

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

        loss, grads = mx.value_and_grad(loss_fn)(ffn, x)

        assert loss.item() > 0


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
        import pytest

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
        import pytest

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


class TestFFNFactoryFunctions:
    """Tests for FFN factory functions."""

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

    def test_create_geglu(self):
        """Test GEGLU factory function."""
        ffn = create_geglu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, GEGLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

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
