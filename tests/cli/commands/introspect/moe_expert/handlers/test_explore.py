"""Tests for explore handler."""

from argparse import Namespace
from unittest.mock import patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
    handle_explore,
)


class TestHandleExplore:
    """Tests for handle_explore function."""

    def test_handle_explore_calls_asyncio_run(self):
        """Test that handle_explore calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_explore_with_layer(self):
        """Test handle_explore with layer parameter."""
        args = Namespace(model="test/model", layer=5)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_explore_with_verbose(self):
        """Test handle_explore with verbose parameter."""
        args = Namespace(model="test/model", verbose=True)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()
