"""Tests for collaboration handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.collaboration import (
    _async_collaboration,
    handle_collaboration,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertPair,
    MoEModelInfo,
)


class TestHandleCollaboration:
    """Tests for handle_collaboration function."""

    def test_handle_collaboration_calls_asyncio_run(self):
        """Test that handle_collaboration calls asyncio.run."""
        args = Namespace(
            model="test/model",
            prompt="Test",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.collaboration.asyncio"
        ) as mock_asyncio:
            handle_collaboration(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncCollaboration:
    """Tests for _async_collaboration function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model")

        await _async_collaboration(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p or --prompts is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_collaboration_single_prompt(self, capsys):
        """Test successful collaboration with single prompt."""
        args = Namespace(
            model="test/model",
            prompt="Test prompt",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            top_pairs=(
                ExpertPair(expert_a=6, expert_b=7, coactivation_count=25, coactivation_rate=0.25),
            ),
            generalist_experts=(6, 7),
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_analysis)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.collaboration.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_collaboration(args)

            captured = capsys.readouterr()
            assert "CO-ACTIVATION ANALYSIS" in captured.out
            assert "E6 + E7" in captured.out

    @pytest.mark.asyncio
    async def test_collaboration_with_multiple_prompts(self, capsys):
        """Test collaboration with multiple prompts."""
        args = Namespace(
            model="test/model",
            prompts="prompt1|prompt2|prompt3",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=150,
            top_pairs=(),
            generalist_experts=(),
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_analysis)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.collaboration.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_collaboration(args)

            captured = capsys.readouterr()
            assert "3 prompt(s)" in captured.out

            # Verify all 3 prompts were passed
            mock_router.analyze_coactivation.assert_called_once()
            call_args = mock_router.analyze_coactivation.call_args
            assert len(call_args[0][0]) == 3

    @pytest.mark.asyncio
    async def test_collaboration_with_prompts_from_file(self, capsys, tmp_path):
        """Test collaboration with prompts from file."""
        # Create a temp file with prompts
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("prompt1\nprompt2\nprompt3\n")

        args = Namespace(
            model="test/model",
            prompts=f"@{prompts_file}",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=150,
            top_pairs=(),
            generalist_experts=(),
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_analysis)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.collaboration.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_collaboration(args)

            # Verify all 3 prompts from file were passed
            mock_router.analyze_coactivation.assert_called_once()
            call_args = mock_router.analyze_coactivation.call_args
            assert len(call_args[0][0]) == 3
