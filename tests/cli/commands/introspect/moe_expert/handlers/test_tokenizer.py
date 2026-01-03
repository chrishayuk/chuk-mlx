"""Tests for tokenizer handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer import (
    _async_tokenizer,
    handle_tokenizer,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleTokenizer:
    """Tests for handle_tokenizer function."""

    def test_handle_tokenizer_calls_asyncio_run(self):
        """Test that handle_tokenizer calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer.asyncio"
        ) as mock_asyncio:
            handle_tokenizer(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncTokenizer:
    """Tests for _async_tokenizer function."""

    @pytest.mark.asyncio
    async def test_successful_tokenizer_analysis(self, capsys):
        """Test successful tokenizer analysis."""
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
                        token="hello",
                        expert_indices=(6,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 100
        mock_tokenizer.decode.return_value = "hello"

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_tokenizer(args)

            captured = capsys.readouterr()
            assert "TOKENIZER-EXPERT ANALYSIS" in captured.out
            assert "Vocabulary size: 100" in captured.out

    @pytest.mark.asyncio
    async def test_tokenizer_with_custom_num_tokens(self, capsys):
        """Test tokenizer analysis with custom num_tokens."""
        args = Namespace(
            model="test/model",
            num_tokens=10,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
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
                        token="a",
                        expert_indices=(0,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.decode.return_value = "a"

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_tokenizer(args)

            # Should have analyzed only 10 tokens
            assert mock_router.capture_router_weights.call_count <= 10

    @pytest.mark.asyncio
    async def test_tokenizer_skips_whitespace_tokens(self, capsys):
        """Test that whitespace tokens are skipped."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 100
        # Return whitespace
        mock_tokenizer.decode.return_value = "   "

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_tokenizer(args)

            # Should complete without errors even though all tokens are whitespace
            captured = capsys.readouterr()
            assert "TOKENIZER-EXPERT ANALYSIS" in captured.out

    @pytest.mark.asyncio
    async def test_tokenizer_handles_exceptions(self, capsys):
        """Test that exceptions during token processing are handled."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 100
        mock_tokenizer.decode.side_effect = Exception("Decode error")

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.tokenizer.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_tokenizer(args)

            # Should complete without crashing
            captured = capsys.readouterr()
            assert "TOKENIZER-EXPERT ANALYSIS" in captured.out
