"""Tests for vocab_map handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map import (
    _async_vocab_map,
    handle_vocab_map,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleVocabMap:
    """Tests for handle_vocab_map function."""

    def test_handle_vocab_map_calls_asyncio_run(self):
        """Test that handle_vocab_map calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map.asyncio"
        ) as mock_asyncio:
            handle_vocab_map(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncVocabMap:
    """Tests for _async_vocab_map function."""

    @pytest.mark.asyncio
    async def test_successful_vocab_map(self, capsys):
        """Test successful vocabulary mapping."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=8,
            num_experts_per_tok=2,
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_vocab_map(args)

            captured = capsys.readouterr()
            assert "VOCABULARY-EXPERT MAPPING" in captured.out
            assert "100 tokens" in captured.out or "500 tokens" in captured.out

    @pytest.mark.asyncio
    async def test_vocab_map_with_custom_tokens(self, capsys):
        """Test vocabulary mapping with custom number of tokens."""
        args = Namespace(
            model="test/model",
            num_tokens=10,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_vocab_map(args)

            captured = capsys.readouterr()
            assert "10 tokens" in captured.out

    @pytest.mark.asyncio
    async def test_vocab_map_skips_empty_tokens(self, capsys):
        """Test that empty tokens are skipped."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 100
        # Return empty string
        mock_tokenizer.decode.return_value = ""

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.tokenizer = mock_tokenizer
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_vocab_map(args)

            # Should complete without errors even though all tokens are empty
            captured = capsys.readouterr()
            assert "VOCABULARY-EXPERT MAPPING" in captured.out

    @pytest.mark.asyncio
    async def test_vocab_map_handles_exceptions(self, capsys):
        """Test that exceptions during token processing are handled."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.vocab_map.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_vocab_map(args)

            # Should complete without crashing
            captured = capsys.readouterr()
            assert "VOCABULARY-EXPERT MAPPING" in captured.out
