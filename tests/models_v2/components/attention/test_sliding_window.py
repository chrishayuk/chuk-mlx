"""
Tests for SlidingWindowAttention.
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.components.attention import SlidingWindowAttention
from chuk_lazarus.models_v2.core.config import AttentionConfig
from chuk_lazarus.models_v2.core.enums import AttentionType


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
        from chuk_lazarus.models_v2.components.attention import MultiHeadAttention

        config = AttentionConfig(
            attention_type=AttentionType.MULTI_HEAD,
            num_attention_heads=8,
            hidden_size=512,
        )
        attn = MultiHeadAttention(config)

        assert isinstance(attn, MultiHeadAttention)

    def test_create_gqa_directly(self):
        """Test creating GroupedQueryAttention directly."""
        from chuk_lazarus.models_v2.components.attention import GroupedQueryAttention

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
