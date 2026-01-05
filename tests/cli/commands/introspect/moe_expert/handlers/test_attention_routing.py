"""Tests for attention_routing handler."""

from argparse import Namespace
from unittest.mock import patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing import (
    _parse_contexts,
    _parse_layers,
    handle_attention_routing,
)


class TestHandleAttentionRouting:
    """Tests for handle_attention_routing function."""

    def test_handle_attention_routing_calls_asyncio_run(self):
        """Test that handle_attention_routing calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_layers(self):
        """Test handle_attention_routing with layers parameter."""
        args = Namespace(model="test/model", layers="0,12,23")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_contexts(self):
        """Test handle_attention_routing with contexts parameter."""
        args = Namespace(model="test/model", contexts="def add,def hello")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_routing_with_token(self):
        """Test handle_attention_routing with token parameter."""
        args = Namespace(model="test/model", token="+")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()


class TestParseLayers:
    """Tests for _parse_layers function."""

    def test_parse_layers_default_three_layers(self):
        """Test default parsing returns early, middle, late layers."""
        moe_layers = (0, 5, 10, 15, 20)
        result = _parse_layers(None, moe_layers)
        assert result == [0, 10, 20]

    def test_parse_layers_default_two_layers(self):
        """Test default parsing with two MoE layers."""
        moe_layers = (0, 5)
        result = _parse_layers(None, moe_layers)
        assert result == [0, 5]

    def test_parse_layers_default_one_layer(self):
        """Test default parsing with one MoE layer."""
        moe_layers = (5,)
        result = _parse_layers(None, moe_layers)
        assert result == [5]

    def test_parse_layers_all(self):
        """Test parsing 'all' returns all MoE layers."""
        moe_layers = (0, 5, 10)
        result = _parse_layers("all", moe_layers)
        assert result == [0, 5, 10]

    def test_parse_layers_comma_separated(self):
        """Test parsing comma-separated layer indices."""
        moe_layers = (0, 5, 10, 15, 20)
        result = _parse_layers("0, 10, 20", moe_layers)
        assert result == [0, 10, 20]


class TestParseContexts:
    """Tests for _parse_contexts function."""

    def test_parse_contexts_default(self):
        """Test default contexts are returned."""
        result = _parse_contexts(None)
        # Should return default contexts with expected names
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_parse_contexts_custom(self):
        """Test parsing custom contexts."""
        result = _parse_contexts("hello world,foo bar")
        assert len(result) == 2
        # First word is used as name
        assert result[0] == ("hello", "hello world")
        assert result[1] == ("foo", "foo bar")

    def test_parse_contexts_single(self):
        """Test parsing single context."""
        result = _parse_contexts("calculate 2+3")
        assert len(result) == 1
        assert result[0] == ("calculate", "calculate 2+3")

    def test_parse_contexts_empty_items_filtered(self):
        """Test empty items are filtered out."""
        result = _parse_contexts("hello,   ,world")
        assert len(result) == 2
        assert result[0][1] == "hello"
        assert result[1][1] == "world"
