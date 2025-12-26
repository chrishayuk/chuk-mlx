"""
Tests for Llama 4 attention.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.llama4.attention import (
    Llama4Attention,
    Llama4FlexAttention,
    create_llama4_attention,
)
from chuk_lazarus.models_v2.families.llama4.config import Llama4TextConfig


class TestLlama4Attention:
    """Tests for Llama4Attention."""

    def test_creation(self):
        """Test attention creation."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config, layer_idx=0)

        assert attn.hidden_size == 64
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.head_dim == 16
        assert attn.n_rep == 2
        assert attn.use_qk_norm is True
        assert attn.scale == 16**-0.5

    def test_is_nope_layer(self):
        """Test NoPE layer detection."""
        config = Llama4TextConfig.tiny()
        config.no_rope_layers = [0, 2]

        # Layer 0 is NoPE
        attn0 = Llama4Attention(config, layer_idx=0)
        assert attn0.is_nope_layer is True
        assert attn0.rope is None

        # Layer 1 is RoPE
        attn1 = Llama4Attention(config, layer_idx=1)
        assert attn1.is_nope_layer is False
        assert attn1.rope is not None

        # Layer 2 is NoPE
        attn2 = Llama4Attention(config, layer_idx=2)
        assert attn2.is_nope_layer is True

    def test_no_nope_layers(self):
        """Test when no_rope_layers is None."""
        config = Llama4TextConfig.tiny()
        config.no_rope_layers = None

        attn = Llama4Attention(config, layer_idx=0)
        assert attn.is_nope_layer is False
        assert attn.rope is not None

    def test_qk_norm_disabled(self):
        """Test attention without QK normalization."""
        config = Llama4TextConfig.tiny()
        config.use_qk_norm = False

        attn = Llama4Attention(config)
        assert attn.use_qk_norm is False

    def test_temperature_tuning(self):
        """Test attention temperature tuning."""
        config = Llama4TextConfig.tiny()
        config.attn_temperature_tuning = True

        attn = Llama4Attention(config)
        assert attn.attn_temperature_tuning is True
        # Should have temperature parameter
        assert hasattr(attn, "temperature")

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config, layer_idx=1)  # RoPE layer

        x = mx.random.normal((2, 10, 64))
        output, cache = attn(x)

        assert output.shape == (2, 10, 64)
        assert cache is not None
        k, v = cache
        assert k.shape[0] == 2
        assert k.shape[2] == 10

    def test_forward_nope_layer(self):
        """Test forward pass for NoPE layer."""
        config = Llama4TextConfig.tiny()
        config.no_rope_layers = [0]

        attn = Llama4Attention(config, layer_idx=0)

        x = mx.random.normal((2, 10, 64))
        output, cache = attn(x)

        assert output.shape == (2, 10, 64)
        assert cache is not None

    def test_forward_with_mask(self):
        """Test forward with mask."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config, layer_idx=1)

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output, cache = attn(x, mask=mask)

        assert output.shape == (2, 10, 64)

    def test_forward_with_cache(self):
        """Test forward with KV cache."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config, layer_idx=1)

        # First pass
        x1 = mx.random.normal((2, 10, 64))
        _, cache = attn(x1)

        # Second pass with cache
        x2 = mx.random.normal((2, 1, 64))
        output, new_cache = attn(x2, cache=cache)

        assert output.shape == (2, 1, 64)
        k, v = new_cache
        assert k.shape[2] == 11  # 10 + 1

    def test_forward_with_qk_norm(self):
        """Test forward with QK normalization."""
        config = Llama4TextConfig.tiny()
        config.use_qk_norm = True

        attn = Llama4Attention(config, layer_idx=1)

        x = mx.random.normal((2, 5, 64))
        output, _ = attn(x)

        assert output.shape == (2, 5, 64)

    def test_forward_with_temperature_tuning(self):
        """Test forward with temperature tuning."""
        config = Llama4TextConfig.tiny()
        config.attn_temperature_tuning = True

        attn = Llama4Attention(config, layer_idx=1)

        x = mx.random.normal((2, 5, 64))
        output, _ = attn(x)

        assert output.shape == (2, 5, 64)

    def test_repeat_kv(self):
        """Test KV repeat method."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config)

        x = mx.random.normal((2, 2, 10, 16))

        # n_rep = 1, should return same
        result = attn._repeat_kv(x, n_rep=1)
        assert result.shape == x.shape

        # n_rep = 2
        result = attn._repeat_kv(x, n_rep=2)
        assert result.shape == (2, 4, 10, 16)


class TestLlama4FlexAttention:
    """Tests for Llama4FlexAttention."""

    def test_creation(self):
        """Test flex attention creation."""
        config = Llama4TextConfig.tiny()
        attn = Llama4FlexAttention(config, layer_idx=0)

        assert attn.floor_scale == 1

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        attn = Llama4FlexAttention(config, layer_idx=1)

        x = mx.random.normal((2, 10, 64))
        output, cache = attn(x)

        assert output.shape == (2, 10, 64)
        assert cache is not None


class TestCreateLlama4Attention:
    """Tests for create_llama4_attention factory function."""

    def test_create_default(self):
        """Test creating default attention."""
        config = Llama4TextConfig.tiny()
        attn = create_llama4_attention(config, layer_idx=0)

        assert isinstance(attn, Llama4Attention)
        assert not isinstance(attn, Llama4FlexAttention)

    def test_create_flex(self):
        """Test creating flex attention."""
        config = Llama4TextConfig.tiny()
        attn = create_llama4_attention(config, layer_idx=0, attention_type="flex")

        assert isinstance(attn, Llama4FlexAttention)


class TestLlama4AttentionGradients:
    """Tests for gradient flow through attention."""

    def test_gradients_flow(self):
        """Test gradients flow through attention."""
        config = Llama4TextConfig.tiny()
        attn = Llama4Attention(config, layer_idx=1)

        x = mx.random.normal((2, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(attn, loss_fn)
        loss, grads = loss_and_grad_fn(attn, x)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_gradients_with_qk_norm(self):
        """Test gradients with QK normalization."""
        config = Llama4TextConfig.tiny()
        config.use_qk_norm = True
        attn = Llama4Attention(config, layer_idx=1)

        x = mx.random.normal((2, 5, 64))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(attn, loss_fn)
        loss, grads = loss_and_grad_fn(attn, x)

        assert loss.item() > 0
