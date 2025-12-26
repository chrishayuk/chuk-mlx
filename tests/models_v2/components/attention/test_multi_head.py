"""
Tests for MultiHeadAttention.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.attention import MultiHeadAttention
from chuk_lazarus.models_v2.core.config import AttentionConfig
from chuk_lazarus.models_v2.core.enums import AttentionType


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = AttentionConfig(
            attention_type=AttentionType.MULTI_HEAD,
            num_attention_heads=8,
            hidden_size=512,
        )
        attn = MultiHeadAttention(config)

        x = mx.random.normal((2, 10, 512))
        output, cache = attn(x)

        assert output.shape == (2, 10, 512)
        assert cache is not None  # Should return KV cache

    def test_with_mask(self):
        """Test attention with mask."""
        config = AttentionConfig(
            num_attention_heads=8,
            hidden_size=512,
        )
        attn = MultiHeadAttention(config)

        x = mx.random.normal((2, 10, 512))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)

        output, cache = attn(x, mask=mask)

        assert output.shape == (2, 10, 512)

    def test_with_cache(self):
        """Test attention with KV cache."""
        config = AttentionConfig(
            num_attention_heads=8,
            hidden_size=512,
        )
        attn = MultiHeadAttention(config)

        # First pass to build cache
        x = mx.random.normal((2, 10, 512))
        _, cache = attn(x)

        # Second pass with cache
        x_new = mx.random.normal((2, 1, 512))
        output, new_cache = attn(x_new, cache=cache)

        assert output.shape == (2, 1, 512)
        # Cache shape: (batch, num_heads, seq_len, head_dim)
        # Cache should have grown along seq_len dimension (axis=2)
        assert new_cache[0].shape[2] == 11  # Original 10 + new 1

    def test_head_dim_computed(self):
        """Test head dimension is computed correctly."""
        config = AttentionConfig(
            num_attention_heads=16,
            hidden_size=1024,
        )
        attn = MultiHeadAttention(config)

        assert attn.head_dim == 64  # 1024 / 16

    def test_custom_head_dim(self):
        """Test custom head dimension."""
        config = AttentionConfig(
            num_attention_heads=8,
            hidden_size=512,
            head_dim=128,  # Custom
        )
        attn = MultiHeadAttention(config)

        assert attn.head_dim == 128


class TestMHAFactoryFunction:
    """Tests for create_multi_head_attention factory function."""

    def test_create_mha_basic(self):
        """Test basic create_multi_head_attention factory function."""
        from chuk_lazarus.models_v2.components.attention.multi_head import (
            create_multi_head_attention,
        )

        attn = create_multi_head_attention(
            hidden_size=512,
            num_heads=8,
        )

        assert isinstance(attn, MultiHeadAttention)
        assert attn.num_heads == 8
        assert attn.head_dim == 64  # 512 // 8

    def test_create_mha_with_custom_head_dim(self):
        """Test create_multi_head_attention with custom head dimension."""
        from chuk_lazarus.models_v2.components.attention.multi_head import (
            create_multi_head_attention,
        )

        attn = create_multi_head_attention(
            hidden_size=512,
            num_heads=8,
            head_dim=128,  # Custom
        )

        assert attn.head_dim == 128

    def test_create_mha_with_bias(self):
        """Test create_multi_head_attention with bias enabled."""
        from chuk_lazarus.models_v2.components.attention.multi_head import (
            create_multi_head_attention,
        )

        attn = create_multi_head_attention(
            hidden_size=256,
            num_heads=4,
            bias=True,
        )

        assert isinstance(attn, MultiHeadAttention)

    def test_create_mha_with_custom_rope(self):
        """Test create_multi_head_attention with custom RoPE parameters."""
        from chuk_lazarus.models_v2.components.attention.multi_head import (
            create_multi_head_attention,
        )

        attn = create_multi_head_attention(
            hidden_size=512,
            num_heads=8,
            rope_theta=50000.0,
            max_position_embeddings=8192,
        )

        assert isinstance(attn, MultiHeadAttention)

    def test_create_mha_forward_pass(self):
        """Test that created MHA works in forward pass."""
        from chuk_lazarus.models_v2.components.attention.multi_head import (
            create_multi_head_attention,
        )

        attn = create_multi_head_attention(
            hidden_size=256,
            num_heads=4,
        )

        x = mx.random.normal((2, 10, 256))
        output, cache = attn(x)

        assert output.shape == (2, 10, 256)
        assert cache is not None


class TestMHAGradients:
    """Tests for gradient flow through MHA."""

    def test_mha_gradients(self):
        """Test gradients flow through MHA."""
        config = AttentionConfig(
            num_attention_heads=4,
            hidden_size=128,
        )
        attn = MultiHeadAttention(config)

        x = mx.random.normal((1, 5, 128))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss_and_grad_fn = nn.value_and_grad(attn, loss_fn)
        loss, grads = loss_and_grad_fn(attn, x)

        assert loss.item() > 0
        # Check gradients exist
        assert any(g is not None for g in grads.values())
