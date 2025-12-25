"""
Tests for Llama 4 MoE.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.llama4.config import Llama4TextConfig
from chuk_lazarus.models_v2.families.llama4.moe import (
    Llama4MLP,
    Llama4MoE,
    SwitchGLU,
    SwitchLinear,
    create_llama4_moe,
    swiglu,
)


class TestLlama4MLP:
    """Tests for Llama4MLP (shared expert)."""

    def test_creation(self):
        """Test MLP creation."""
        mlp = Llama4MLP(hidden_size=64, intermediate_size=128)

        assert mlp.hidden_size == 64
        assert mlp.intermediate_size == 128

    def test_forward_pass(self):
        """Test forward pass."""
        mlp = Llama4MLP(hidden_size=64, intermediate_size=128)

        x = mx.random.normal((2, 10, 64))
        output = mlp(x)

        assert output.shape == (2, 10, 64)

    def test_with_bias(self):
        """Test MLP with bias."""
        mlp = Llama4MLP(hidden_size=64, intermediate_size=128, bias=True)

        x = mx.random.normal((2, 5, 64))
        output = mlp(x)

        assert output.shape == (2, 5, 64)


class TestSwitchLinear:
    """Tests for SwitchLinear (expert selection layer)."""

    def test_creation(self):
        """Test SwitchLinear creation."""
        layer = SwitchLinear(input_dims=64, output_dims=128, num_experts=4)

        assert layer.input_dims == 64
        assert layer.output_dims == 128
        assert layer.num_experts == 4
        assert layer.weight.shape == (4, 128, 64)

    def test_creation_with_bias(self):
        """Test SwitchLinear with bias."""
        layer = SwitchLinear(input_dims=64, output_dims=128, num_experts=4, bias=True)

        assert "bias" in layer
        assert layer.bias.shape == (4, 128)

    def test_forward_pass(self):
        """Test forward pass."""
        layer = SwitchLinear(input_dims=64, output_dims=128, num_experts=4)

        # Input: (..., 1, 1, input_dims)
        x = mx.random.normal((10, 1, 1, 64))
        indices = mx.array([[0], [1], [2], [3], [0], [1], [2], [3], [0], [1]])

        output = layer(x, indices)

        # Output: (..., k, 1, output_dims)
        assert output.shape == (10, 1, 1, 128)

    def test_forward_with_bias(self):
        """Test forward with bias."""
        layer = SwitchLinear(input_dims=64, output_dims=128, num_experts=4, bias=True)

        x = mx.random.normal((5, 1, 1, 64))
        indices = mx.array([[0], [1], [2], [3], [0]])

        output = layer(x, indices)

        assert output.shape == (5, 1, 1, 128)


class TestSwiglu:
    """Tests for swiglu function."""

    def test_swiglu(self):
        """Test swiglu activation."""
        x = mx.random.normal((2, 5, 64))
        gate = mx.random.normal((2, 5, 64))

        output = swiglu(x, gate)

        assert output.shape == (2, 5, 64)


class TestSwitchGLU:
    """Tests for SwitchGLU (expert MLP with gather_mm)."""

    def test_creation(self):
        """Test SwitchGLU creation."""
        switch_glu = SwitchGLU(input_dims=64, hidden_dims=128, num_experts=4, bias=False)

        assert switch_glu.gate_proj is not None
        assert switch_glu.up_proj is not None
        assert switch_glu.down_proj is not None

    def test_forward_pass(self):
        """Test forward pass."""
        switch_glu = SwitchGLU(input_dims=64, hidden_dims=128, num_experts=4, bias=False)

        x = mx.random.normal((10, 64))  # (batch * seq, hidden_size)
        indices = mx.array([[0], [1], [2], [3], [0], [1], [2], [3], [0], [1]])

        output = switch_glu(x, indices)

        # Output: (batch * seq, k, hidden_size)
        assert output.shape == (10, 1, 64)


class TestLlama4MoE:
    """Tests for Llama4MoE."""

    def test_creation(self):
        """Test MoE creation."""
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        assert moe.hidden_size == 64
        assert moe.intermediate_size == 128
        assert moe.intermediate_size_mlp == 256
        assert moe.num_experts == 4
        assert moe.num_experts_per_tok == 1

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        x = mx.random.normal((2, 10, 64))
        output = moe(x)

        assert output.shape == (2, 10, 64)

    def test_shared_expert(self):
        """Test that shared expert is always active."""
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        # Shared expert should exist
        assert moe.shared_expert is not None

        x = mx.random.normal((2, 5, 64))
        output = moe(x)
        assert output.shape == (2, 5, 64)

    def test_router(self):
        """Test that router produces valid outputs."""
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        # Router should project to num_experts
        x = mx.random.normal((2, 5, 64))
        router_logits = moe.router(x)
        assert router_logits.shape == (2, 5, 4)


class TestCreateLlama4MoE:
    """Tests for create_llama4_moe factory function."""

    def test_create(self):
        """Test factory function."""
        config = Llama4TextConfig.tiny()
        moe = create_llama4_moe(config)

        assert isinstance(moe, Llama4MoE)


class TestLlama4MoEGradients:
    """Tests for gradient flow through MoE."""

    def test_forward_produces_output(self):
        """Test forward pass produces valid output (gradient tests skipped due to gather_mm limitation)."""
        # Note: Full gradient tests are skipped because MLX's gather_mm operation
        # (used in expert selection) does not support VJP with respect to indices.
        # This is a known limitation: "Cannot calculate VJP with respect to indices"
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        x = mx.random.normal((2, 5, 64))
        out = moe(x)

        assert out.shape == (2, 5, 64)
        # Verify output is finite
        assert mx.all(mx.isfinite(out))

    def test_shared_expert_gradients(self):
        """Test gradients flow through shared expert."""
        config = Llama4TextConfig.tiny()
        moe = Llama4MoE(config)

        x = mx.random.normal((2, 5, 64))

        def loss_fn(model, x):
            # Just use shared expert (no gather_mm)
            return mx.mean(model.shared_expert(x) ** 2)

        loss_and_grad_fn = nn.value_and_grad(moe, loss_fn)
        loss, grads = loss_and_grad_fn(moe, x)

        assert loss.item() > 0
