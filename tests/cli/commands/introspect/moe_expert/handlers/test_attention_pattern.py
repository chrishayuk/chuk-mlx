"""Tests for attention_pattern handler."""

from argparse import Namespace
from unittest.mock import patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern import (
    handle_attention_pattern,
)


class TestHandleAttentionPattern:
    """Tests for handle_attention_pattern function."""

    def test_handle_attention_pattern_calls_asyncio_run(self):
        """Test that handle_attention_pattern calls asyncio.run."""
        args = Namespace(model="test/model", prompt="test prompt")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_layer(self):
        """Test handle_attention_pattern with layer parameter."""
        args = Namespace(model="test/model", prompt="test prompt", layer=5)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_position(self):
        """Test handle_attention_pattern with position parameter."""
        args = Namespace(model="test/model", prompt="test", position=2)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_head(self):
        """Test handle_attention_pattern with head parameter."""
        args = Namespace(model="test/model", prompt="test", head=0)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_top_k(self):
        """Test handle_attention_pattern with top_k parameter."""
        args = Namespace(model="test/model", prompt="test", top_k=10)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()
