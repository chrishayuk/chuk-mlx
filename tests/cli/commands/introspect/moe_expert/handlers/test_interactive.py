"""Tests for interactive handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive import (
    _async_interactive,
    handle_interactive,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import MoEModelInfo


class TestHandleInteractive:
    """Tests for handle_interactive function."""

    def test_handle_interactive_calls_asyncio_run(self):
        """Test that handle_interactive calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.asyncio"
        ) as mock_asyncio:
            handle_interactive(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncInteractive:
    """Tests for _async_interactive function."""

    @pytest.mark.asyncio
    async def test_interactive_startup_and_quit(self, capsys):
        """Test interactive mode starts and can quit."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["quit"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "INTERACTIVE EXPERT EXPLORER" in captured.out
            assert "Commands:" in captured.out
            assert "chat" in captured.out

    @pytest.mark.asyncio
    async def test_interactive_eof_exits(self, capsys):
        """Test interactive mode exits on EOF."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=EOFError),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Exiting" in captured.out

    @pytest.mark.asyncio
    async def test_interactive_empty_input_ignored(self, capsys):
        """Test that empty input is ignored."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["", "  ", "q"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            # Should complete without error
            captured = capsys.readouterr()
            assert "INTERACTIVE EXPERT EXPLORER" in captured.out

    @pytest.mark.asyncio
    async def test_interactive_chat_command(self, capsys):
        """Test chat command in interactive mode."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_result = AsyncMock()
        mock_result.response = "Hello there!"

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.chat_with_expert = AsyncMock(return_value=mock_result)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["chat 6 Hello world", "quit"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Expert 6:" in captured.out
            mock_router.chat_with_expert.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_compare_command(self, capsys):
        """Test compare command in interactive mode."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_expert_result = AsyncMock()
        mock_expert_result.expert_idx = 6
        mock_expert_result.response = "Response from expert 6"

        mock_result = AsyncMock()
        mock_result.expert_results = [mock_expert_result]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.compare_experts = AsyncMock(return_value=mock_result)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["compare 6,7 Hello", "exit"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Expert 6:" in captured.out
            mock_router.compare_experts.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_weights_command(self, capsys):
        """Test weights command in interactive mode."""
        from chuk_lazarus.introspection.moe.models import (
            LayerRouterWeights,
            RouterWeightCapture,
        )

        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="hello",
                        expert_indices=(6, 7),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["weights Hello", "q"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Layer 0:" in captured.out
            mock_router.capture_router_weights.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_unknown_command(self, capsys):
        """Test unknown command shows error."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["unknown_cmd", "quit"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Unknown command" in captured.out

    @pytest.mark.asyncio
    async def test_interactive_command_error_handling(self, capsys):
        """Test error handling in commands."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.chat_with_expert = AsyncMock(side_effect=ValueError("Test error"))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=["chat 6 Hello", "quit"]),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Error:" in captured.out

    @pytest.mark.asyncio
    async def test_interactive_keyboard_interrupt(self, capsys):
        """Test interactive mode exits on KeyboardInterrupt."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.interactive.ExpertRouter"
            ) as MockRouter,
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_interactive(args)

            captured = capsys.readouterr()
            assert "Exiting" in captured.out
