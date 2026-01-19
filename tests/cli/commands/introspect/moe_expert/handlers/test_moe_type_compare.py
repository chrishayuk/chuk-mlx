"""Tests for moe_type_compare handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_compare import (
    _async_moe_type_compare,
    handle_moe_type_compare,
)
from chuk_lazarus.introspection.moe import MoEType
from chuk_lazarus.introspection.moe.moe_type import (
    MoETypeAnalysis,
    ProjectionRankAnalysis,
)


class TestHandleMoETypeCompare:
    """Tests for handle_moe_type_compare function."""

    def test_handle_moe_type_compare_calls_asyncio_run(self):
        """Test that handle_moe_type_compare calls asyncio.run."""
        args = Namespace(model="test/model", compare_model="other/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_compare.asyncio"
        ) as mock_asyncio:
            handle_moe_type_compare(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncMoETypeCompare:
    """Tests for _async_moe_type_compare function."""

    @pytest.fixture
    def mock_pseudo_analysis(self):
        """Create a mock pseudo-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="pseudo/model",
            layer_idx=0,
            num_experts=32,
            moe_type=MoEType.UPCYCLED,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(2880, 2880), max_rank=2880, effective_rank_95=1
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(2880, 2880), max_rank=2880, effective_rank_95=337
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2880, 2880), max_rank=2880, effective_rank_95=206
            ),
            mean_cosine_similarity=0.418,
            std_cosine_similarity=0.163,
        )

    @pytest.fixture
    def mock_native_analysis(self):
        """Create a mock native-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="native/model",
            layer_idx=0,
            num_experts=64,
            moe_type=MoEType.PRETRAINED_MOE,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(1024, 2048), max_rank=1024, effective_rank_95=755
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(1024, 2048), max_rank=1024, effective_rank_95=772
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2048, 1024), max_rank=1024, effective_rank_95=785
            ),
            mean_cosine_similarity=0.0,
            std_cosine_similarity=0.001,
        )

    @pytest.mark.asyncio
    async def test_successful_compare(self, capsys, mock_pseudo_analysis, mock_native_analysis):
        """Test successful compare execution."""
        args = Namespace(
            model="pseudo/model",
            compare_model="native/model",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_compare.MoETypeService"
        ) as MockService:
            # Return different results for each model
            async def analyze_side_effect(model_id):
                if "pseudo" in model_id:
                    return mock_pseudo_analysis
                return mock_native_analysis

            MockService.analyze = AsyncMock(side_effect=analyze_side_effect)

            await _async_moe_type_compare(args)

            captured = capsys.readouterr()
            assert "Comparing MoE types" in captured.out
            assert "pseudo/model" in captured.out
            assert "native/model" in captured.out
            assert "MOE TYPE COMPARISON" in captured.out

    @pytest.mark.asyncio
    async def test_compare_missing_compare_model(self, capsys):
        """Test compare with missing compare_model."""
        args = Namespace(
            model="test/model",
            compare_model=None,
        )

        await _async_moe_type_compare(args)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "--compare-model" in captured.out

    @pytest.mark.asyncio
    async def test_compare_missing_attribute(self, capsys):
        """Test compare handles missing compare_model attribute."""
        # Simulate args without compare_model attribute
        args = Namespace(model="test/model")

        await _async_moe_type_compare(args)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "--compare-model" in captured.out

    @pytest.mark.asyncio
    async def test_compare_concurrent_analysis(self, mock_pseudo_analysis, mock_native_analysis):
        """Test that both analyses run concurrently."""
        args = Namespace(
            model="pseudo/model",
            compare_model="native/model",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_compare.MoETypeService"
        ) as MockService:

            async def analyze_side_effect(model_id):
                if "pseudo" in model_id:
                    return mock_pseudo_analysis
                return mock_native_analysis

            MockService.analyze = AsyncMock(side_effect=analyze_side_effect)

            await _async_moe_type_compare(args)

            # Verify both models were analyzed
            assert MockService.analyze.call_count == 2
