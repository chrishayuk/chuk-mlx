"""Tests for context_test handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test import (
    _async_context_test,
    handle_context_test,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleContextTest:
    """Tests for handle_context_test function."""

    def test_handle_context_test_calls_asyncio_run(self):
        """Test that handle_context_test calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.asyncio"
        ) as mock_asyncio:
            handle_context_test(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncContextTest:
    """Tests for _async_context_test function."""

    @pytest.mark.asyncio
    async def test_successful_context_test(self, capsys):
        """Test successful context test execution."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create mock test data
        mock_test_data = MagicMock()
        mock_test_data.target_token = "test"
        mock_test = MagicMock()
        mock_test.prompt = "This is a test"
        mock_test.context_type = "neutral"
        mock_test_data.tests = [mock_test]

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="test",
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
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.get_context_tests"
            ) as mock_get_tests,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_tests.return_value = mock_test_data

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out
            assert "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_context_test_shows_context_dependent_verdict(self, capsys):
        """Test context test shows context-dependent verdict when routing varies."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create mock test data with multiple tests
        mock_test_data = MagicMock()
        mock_test_data.target_token = "test"

        mock_test1 = MagicMock()
        mock_test1.prompt = "This is a test"
        mock_test1.context_type = "neutral"

        mock_test2 = MagicMock()
        mock_test2.prompt = "Another test context"
        mock_test2.context_type = "neutral"

        mock_test_data.tests = [mock_test1, mock_test2]

        # Return different expert indices for different prompts
        mock_weights1 = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="test",
                        expert_indices=(6, 7),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
        ]

        mock_weights2 = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="test",
                        expert_indices=(8, 9),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=[mock_weights1, mock_weights2])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.get_context_tests"
            ) as mock_get_tests,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_tests.return_value = mock_test_data

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "Verdict: Routing is CONTEXT-DEPENDENT" in captured.out
