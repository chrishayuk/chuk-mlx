"""Tests for moe_expert package __init__.py."""

from argparse import Namespace
from unittest.mock import patch

from chuk_lazarus.cli.commands.introspect.moe_expert import (
    dispatch,
    introspect_moe_expert,
)


class TestIntrospectMoeExpert:
    """Tests for introspect_moe_expert entry point."""

    def test_introspect_moe_expert_calls_dispatch(self):
        """Test that introspect_moe_expert delegates to dispatch."""
        args = Namespace(action="chat", model="test/model")

        with patch("chuk_lazarus.cli.commands.introspect.moe_expert.dispatch") as mock_dispatch:
            introspect_moe_expert(args)
            mock_dispatch.assert_called_once_with(args)

    def test_dispatch_is_exported(self):
        """Test that dispatch is properly exported."""
        assert dispatch is not None
        assert callable(dispatch)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports are available."""
        from chuk_lazarus.cli.commands.introspect import moe_expert

        assert hasattr(moe_expert, "dispatch")
        assert hasattr(moe_expert, "introspect_moe_expert")
