"""
Tests for attention components.

Tests MultiHeadAttention, GroupedQueryAttention, and SlidingWindowAttention.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    SlidingWindowAttention,
)
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


class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = AttentionConfig(
            attention_type=AttentionType.SLIDING_WINDOW,
            num_attention_heads=8,
            hidden_size=512,
            sliding_window_size=128,
        )
        attn = SlidingWindowAttention(config)

        x = mx.random.normal((2, 256, 512))  # Longer than window
        output, cache = attn(x)

        assert output.shape == (2, 256, 512)

    def test_window_size_respected(self):
        """Test that window size is stored correctly."""
        config = AttentionConfig(
            num_attention_heads=8,
            hidden_size=512,
            sliding_window_size=256,
        )
        attn = SlidingWindowAttention(config)

        assert attn.window_size == 256

    def test_short_sequence_no_truncation(self):
        """Test short sequences work without truncation."""
        config = AttentionConfig(
            num_attention_heads=8,
            hidden_size=512,
            sliding_window_size=1024,
        )
        attn = SlidingWindowAttention(config)

        x = mx.random.normal((2, 50, 512))  # Shorter than window
        output, _ = attn(x)

        assert output.shape == (2, 50, 512)

    def test_missing_window_size_raises(self):
        """Test that missing window_size raises error."""
        import pytest

        config = AttentionConfig(
            attention_type=AttentionType.SLIDING_WINDOW,
            num_attention_heads=8,
            hidden_size=512,
            sliding_window_size=None,  # Missing
        )
        with pytest.raises(ValueError, match="sliding_window_size must be set"):
            SlidingWindowAttention(config)

    def test_with_cache_generation(self):
        """Test sliding window attention with cache (generation mode)."""
        config = AttentionConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=256,
            sliding_window_size=32,
        )
        attn = SlidingWindowAttention(config)

        # First pass - build up cache
        x1 = mx.random.normal((1, 20, 256))
        output1, cache = attn(x1)

        # Second pass - with cache
        x2 = mx.random.normal((1, 1, 256))
        output2, cache2 = attn(x2, cache=cache)

        assert output2.shape == (1, 1, 256)
        assert cache2 is not None

    def test_cache_truncation_long_sequence(self):
        """Test cache is truncated for long sequences beyond window."""
        config = AttentionConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=256,
            sliding_window_size=16,  # Small window
        )
        attn = SlidingWindowAttention(config)

        # Build up cache longer than window
        x1 = mx.random.normal((1, 10, 256))
        _, cache1 = attn(x1)

        # Continue generation until cache exceeds window
        for _ in range(10):  # This should exceed window size
            x_next = mx.random.normal((1, 1, 256))
            _, cache1 = attn(x_next, cache=cache1)

        # Cache should be truncated to window size
        k_cache, v_cache = cache1
        assert k_cache.shape[2] <= 16  # Should be <= window_size

    def test_with_external_mask(self):
        """Test sliding window attention with external mask."""
        config = AttentionConfig(
            num_attention_heads=4,
            hidden_size=256,
            sliding_window_size=64,
        )
        attn = SlidingWindowAttention(config)

        x = mx.random.normal((1, 32, 256))
        # Create a simple mask
        mask = mx.zeros((32, 32))

        output, _ = attn(x, mask=mask)
        assert output.shape == (1, 32, 256)


class TestSlidingWindowFactory:
    """Tests for sliding window factory function."""

    def test_create_sliding_window_attention(self):
        """Test factory function."""
        from chuk_lazarus.models_v2.components.attention.sliding_window import (
            create_sliding_window_attention,
        )

        attn = create_sliding_window_attention(
            hidden_size=512,
            num_heads=8,
            num_kv_heads=8,
            window_size=128,
        )

        assert isinstance(attn, SlidingWindowAttention)
        assert attn.window_size == 128

    def test_create_with_custom_params(self):
        """Test factory with custom parameters."""
        from chuk_lazarus.models_v2.components.attention.sliding_window import (
            create_sliding_window_attention,
        )

        attn = create_sliding_window_attention(
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,  # GQA
            window_size=64,
            head_dim=64,
            bias=True,
            rope_theta=50000.0,
            max_position_embeddings=8192,
        )

        assert isinstance(attn, SlidingWindowAttention)
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2


class TestAttentionFactory:
    """Tests for attention factory pattern."""

    def test_create_mha_directly(self):
        """Test creating MultiHeadAttention directly."""
        config = AttentionConfig(
            attention_type=AttentionType.MULTI_HEAD,
            num_attention_heads=8,
            hidden_size=512,
        )
        attn = MultiHeadAttention(config)

        assert isinstance(attn, MultiHeadAttention)

    def test_create_gqa_directly(self):
        """Test creating GroupedQueryAttention directly."""
        config = AttentionConfig(
            attention_type=AttentionType.GROUPED_QUERY,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        attn = GroupedQueryAttention(config)

        assert isinstance(attn, GroupedQueryAttention)

    def test_create_sliding_window_directly(self):
        """Test creating SlidingWindowAttention directly."""
        config = AttentionConfig(
            attention_type=AttentionType.SLIDING_WINDOW,
            num_attention_heads=8,
            hidden_size=512,
            sliding_window_size=256,
        )
        attn = SlidingWindowAttention(config)

        assert isinstance(attn, SlidingWindowAttention)


class TestAttentionGradients:
    """Tests for gradient flow through attention."""

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

        loss, grads = mx.value_and_grad(loss_fn)(attn, x)

        assert loss.item() > 0
        # Check gradients exist
        assert any(g is not None for g in grads.values())

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
        import pytest

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
