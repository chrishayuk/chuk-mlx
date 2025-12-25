"""
Tests for GroupedQueryAttention (GQA).
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.components.attention import GroupedQueryAttention
from chuk_lazarus.models_v2.core.config import AttentionConfig
from chuk_lazarus.models_v2.core.enums import AttentionType


class TestGroupedQueryAttention:
    """Tests for GroupedQueryAttention (GQA)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = AttentionConfig(
            attention_type=AttentionType.GROUPED_QUERY,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            hidden_size=4096,
        )
        attn = GroupedQueryAttention(config)

        x = mx.random.normal((1, 5, 4096))
        output, cache = attn(x)

        assert output.shape == (1, 5, 4096)

    def test_kv_heads_different(self):
        """Test KV projection has fewer heads than Q."""
        config = AttentionConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        attn = GroupedQueryAttention(config)

        # KV projection should have fewer dimensions
        head_dim = 4096 // 32
        kv_dim = 8 * head_dim
        assert attn.k_proj.weight.shape[0] == kv_dim

    def test_mqa_single_kv_head(self):
        """Test multi-query attention (single KV head)."""
        config = AttentionConfig(
            num_attention_heads=32,
            num_key_value_heads=1,  # MQA
            hidden_size=4096,
        )
        attn = GroupedQueryAttention(config)

        x = mx.random.normal((1, 10, 4096))
        output, cache = attn(x)

        assert output.shape == (1, 10, 4096)

    def test_with_rope(self):
        """Test GQA with RoPE enabled."""
        config = AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=4,
            hidden_size=512,
        )
        attn = GroupedQueryAttention(config)

        x = mx.random.normal((2, 20, 512))
        output, _ = attn(x)

        assert output.shape == (2, 20, 512)

    def test_kv_cache_shapes(self):
        """Test KV cache has correct shapes for GQA."""
        config = AttentionConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        attn = GroupedQueryAttention(config)

        x = mx.random.normal((2, 10, 4096))
        _, cache = attn(x)

        k_cache, v_cache = cache
        head_dim = 4096 // 32
        # KV cache shape: (batch, num_kv_heads, seq_len, head_dim)
        assert k_cache.shape == (2, 8, 10, head_dim)
        assert v_cache.shape == (2, 8, 10, head_dim)


class TestGQAFactoryFunction:
    """Tests for create_grouped_query_attention factory function."""

    def test_create_gqa_basic(self):
        """Test basic create_grouped_query_attention factory function."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
        )

        assert isinstance(attn, GroupedQueryAttention)
        assert attn.num_heads == 32
        assert attn.num_kv_heads == 8

    def test_create_gqa_with_custom_head_dim(self):
        """Test create_grouped_query_attention with custom head dimension."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )

        assert attn.head_dim == 128

    def test_create_gqa_with_bias(self):
        """Test create_grouped_query_attention with bias enabled."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=2048,
            num_heads=16,
            num_kv_heads=4,
            bias=True,
        )

        assert isinstance(attn, GroupedQueryAttention)

    def test_create_gqa_with_custom_rope(self):
        """Test create_grouped_query_attention with custom RoPE parameters."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            rope_theta=100000.0,
            max_position_embeddings=32768,
        )

        assert isinstance(attn, GroupedQueryAttention)

    def test_create_gqa_mqa_single_kv_head(self):
        """Test create_grouped_query_attention with single KV head (MQA)."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=2048,
            num_heads=32,
            num_kv_heads=1,  # MQA
        )

        assert attn.num_kv_heads == 1

    def test_create_gqa_invalid_head_ratio(self):
        """Test create_grouped_query_attention raises error for invalid head ratio."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        with pytest.raises(ValueError, match="must be divisible"):
            create_grouped_query_attention(
                hidden_size=4096,
                num_heads=32,
                num_kv_heads=5,  # 32 not divisible by 5
            )

    def test_create_gqa_forward_pass(self):
        """Test that created GQA works in forward pass."""
        from chuk_lazarus.models_v2.components.attention.grouped_query import (
            create_grouped_query_attention,
        )

        attn = create_grouped_query_attention(
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
        )

        x = mx.random.normal((2, 20, 1024))
        output, cache = attn(x)

        assert output.shape == (2, 20, 1024)
        assert cache is not None


class TestGQAGradients:
    """Tests for gradient flow through GQA."""

    def test_gqa_gradients(self):
        """Test gradients flow through GQA."""
        config = AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_size=256,
        )
        attn = GroupedQueryAttention(config)

        x = mx.random.normal((1, 5, 256))

        def loss_fn(model, x):
            out, _ = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(attn, x)

        assert loss.item() > 0
