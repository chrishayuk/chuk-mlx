"""Tests for control_tokens handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens import (
    _async_control_tokens,
    handle_control_tokens,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleControlTokens:
    """Tests for handle_control_tokens function."""

    def test_handle_control_tokens_calls_asyncio_run(self):
        """Test that handle_control_tokens calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens.asyncio"
        ) as mock_asyncio:
            handle_control_tokens(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncControlTokens:
    """Tests for _async_control_tokens function."""

    @pytest.mark.asyncio
    async def test_successful_control_tokens(self, capsys):
        """Test successful control tokens analysis."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="<s>",
                        expert_indices=(6, 7, 20),
                        weights=(0.5, 0.3, 0.2),
                    ),
                ),
            )
        ]

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_control_tokens(args)

            captured = capsys.readouterr()
            assert "CONTROL TOKEN EXPERT ANALYSIS" in captured.out
            assert "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_control_tokens_with_specific_layer(self, capsys):
        """Test control tokens with specific layer."""
        args = Namespace(model="test/model", layer=2)

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=2,
                positions=(
                    RouterWeightCapture(
                        layer_idx=2,
                        position_idx=0,
                        token="<s>",
                        expert_indices=(10,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_control_tokens(args)

            captured = capsys.readouterr()
            assert "layer 2" in captured.out

    @pytest.mark.asyncio
    async def test_control_tokens_skips_empty_encoded(self, capsys):
        """Test that tokens with empty encoding are skipped."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_tokenizer = MagicMock()
        # Return empty list for some tokens
        mock_tokenizer.encode.return_value = []

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_control_tokens(args)

            # Should complete without errors even though all tokens are skipped
            captured = capsys.readouterr()
            assert "CONTROL TOKEN EXPERT ANALYSIS" in captured.out

    @pytest.mark.asyncio
    async def test_control_tokens_handles_exceptions(self, capsys):
        """Test that exceptions during token processing are handled."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = Exception("Encoding error")

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.control_tokens.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_control_tokens(args)

            # Should complete without crashing
            captured = capsys.readouterr()
            assert "CONTROL TOKEN EXPERT ANALYSIS" in captured.out
