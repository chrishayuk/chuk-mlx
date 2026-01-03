"""Tests for MoE expert CLI dispatcher."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher import (
    _get_handlers,
    dispatch,
)
from chuk_lazarus.introspection.moe.enums import MoEAction


class TestGetHandlers:
    """Tests for _get_handlers function."""

    def test_returns_dict(self):
        """Test that _get_handlers returns a dictionary."""
        handlers = _get_handlers()
        assert isinstance(handlers, dict)

    def test_all_actions_have_handlers(self):
        """Test that all MoEAction values have handlers."""
        handlers = _get_handlers()
        for action in MoEAction:
            assert action in handlers, f"Missing handler for {action.value}"

    def test_handlers_are_callable(self):
        """Test that all handlers are callable."""
        handlers = _get_handlers()
        for action, handler in handlers.items():
            assert callable(handler), f"Handler for {action.value} is not callable"

    def test_has_21_handlers(self):
        """Test that we have exactly 21 handlers."""
        handlers = _get_handlers()
        assert len(handlers) == 21


class TestDispatch:
    """Tests for dispatch function."""

    def test_dispatch_chat_action(self):
        """Test dispatching chat action."""
        args = Namespace(action="chat", model="test/model", expert=6, prompt="Test")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
        ) as mock_get_handlers:
            mock_handler = MagicMock()
            mock_get_handlers.return_value = {MoEAction.CHAT: mock_handler}

            dispatch(args)

            mock_handler.assert_called_once_with(args)

    def test_dispatch_compare_action(self):
        """Test dispatching compare action."""
        args = Namespace(action="compare", model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
        ) as mock_get_handlers:
            mock_handler = MagicMock()
            mock_get_handlers.return_value = {MoEAction.COMPARE: mock_handler}

            dispatch(args)

            mock_handler.assert_called_once_with(args)

    def test_dispatch_unknown_action_prints_error(self, capsys):
        """Test dispatching unknown action prints error."""
        args = Namespace(action="unknown_action", model="test/model")

        dispatch(args)

        captured = capsys.readouterr()
        assert "Unknown action: unknown_action" in captured.out
        assert "Available actions:" in captured.out

    def test_dispatch_default_action_is_chat(self):
        """Test that default action is chat when not specified."""
        args = Namespace(model="test/model")  # No action attribute

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
        ) as mock_get_handlers:
            mock_handler = MagicMock()
            mock_get_handlers.return_value = {MoEAction.CHAT: mock_handler}

            dispatch(args)

            mock_handler.assert_called_once_with(args)

    def test_dispatch_hyphenated_actions(self):
        """Test dispatching actions with hyphens in value."""
        hyphenated_actions = [
            ("control-tokens", MoEAction.CONTROL_TOKENS),
            ("context-test", MoEAction.CONTEXT_TEST),
            ("vocab-map", MoEAction.VOCAB_MAP),
            ("router-probe", MoEAction.ROUTER_PROBE),
            ("pattern-discovery", MoEAction.PATTERN_DISCOVERY),
            ("full-taxonomy", MoEAction.FULL_TAXONOMY),
            ("layer-sweep", MoEAction.LAYER_SWEEP),
        ]

        for action_str, action_enum in hyphenated_actions:
            args = Namespace(action=action_str, model="test/model")

            with patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
            ) as mock_get_handlers:
                mock_handler = MagicMock()
                mock_get_handlers.return_value = {action_enum: mock_handler}

                dispatch(args)

                mock_handler.assert_called_once_with(args)

    def test_dispatch_all_actions(self):
        """Test that all action strings can be dispatched."""
        for action in MoEAction:
            args = Namespace(action=action.value, model="test/model")

            with patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
            ) as mock_get_handlers:
                mock_handler = MagicMock()
                mock_get_handlers.return_value = {action: mock_handler}

                dispatch(args)

                mock_handler.assert_called_once_with(args)


class TestDispatchLogging:
    """Tests for dispatch logging behavior."""

    def test_dispatch_logs_debug_message(self, caplog):
        """Test that dispatch logs debug message."""
        import logging

        caplog.set_level(logging.DEBUG)
        args = Namespace(action="chat", model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.dispatcher._get_handlers"
        ) as mock_get_handlers:
            mock_handler = MagicMock()
            mock_get_handlers.return_value = {MoEAction.CHAT: mock_handler}

            dispatch(args)

            assert "Dispatching to handler for action: chat" in caplog.text
