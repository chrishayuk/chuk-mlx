"""
Tests for protocol type definitions.
"""

import typing

from chuk_lazarus.models_v2.core.protocols import (
    AttentionCache,
    BlockCache,
    LayerCache,
)


class TestAttentionCache:
    """Tests for AttentionCache type."""

    def test_is_tuple_type(self):
        """Test AttentionCache is a tuple type alias."""
        # AttentionCache should be a type alias for tuple[Any, Any]
        origin = typing.get_origin(AttentionCache)
        assert origin is tuple

    def test_valid_cache(self):
        """Test creating a valid attention cache."""
        import mlx.core as mx

        keys = mx.zeros((1, 4, 8, 16))
        values = mx.zeros((1, 4, 8, 16))
        cache: AttentionCache = (keys, values)
        assert len(cache) == 2


class TestBlockCache:
    """Tests for BlockCache type."""

    def test_is_any_type(self):
        """Test BlockCache is Any type alias."""
        # BlockCache is Any (flexible block-specific cache)
        assert BlockCache is typing.Any


class TestLayerCache:
    """Tests for LayerCache type."""

    def test_is_list_type(self):
        """Test LayerCache is a list type alias."""
        # LayerCache should be a type alias for list[Any]
        origin = typing.get_origin(LayerCache)
        assert origin is list
