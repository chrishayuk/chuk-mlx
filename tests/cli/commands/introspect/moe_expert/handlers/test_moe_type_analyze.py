"""Tests for moe_type_analyze handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze import (
    _async_moe_type_analyze,
    handle_moe_type_analyze,
)
from chuk_lazarus.introspection.moe import MoEType
from chuk_lazarus.introspection.moe.moe_type import (
    MoETypeAnalysis,
    ProjectionRankAnalysis,
)


class TestHandleMoETypeAnalyze:
    """Tests for handle_moe_type_analyze function."""

    def test_handle_moe_type_analyze_calls_asyncio_run(self):
        """Test that handle_moe_type_analyze calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze.asyncio"
        ) as mock_asyncio:
            handle_moe_type_analyze(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncMoETypeAnalyze:
    """Tests for _async_moe_type_analyze function."""

    @pytest.fixture
    def mock_analysis_result(self):
        """Create a mock analysis result."""
        return MoETypeAnalysis(
            model_id="test/model",
            layer_idx=0,
            num_experts=32,
            moe_type=MoEType.PSEUDO,
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

    @pytest.mark.asyncio
    async def test_successful_analyze(self, capsys, mock_analysis_result):
        """Test successful analyze execution."""
        args = Namespace(
            model="test/model",
            layer=None,
            output=None,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze.MoETypeService"
        ) as MockService:
            MockService.analyze = AsyncMock(return_value=mock_analysis_result)

            await _async_moe_type_analyze(args)

            captured = capsys.readouterr()
            assert "Analyzing MoE type" in captured.out
            assert "MOE TYPE ANALYSIS" in captured.out
            assert "PSEUDO-MOE" in captured.out

    @pytest.mark.asyncio
    async def test_analyze_with_layer(self, capsys, mock_analysis_result):
        """Test analyze with specific layer."""
        args = Namespace(
            model="test/model",
            layer=5,
            output=None,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze.MoETypeService"
        ) as MockService:
            MockService.analyze = AsyncMock(return_value=mock_analysis_result)

            await _async_moe_type_analyze(args)

            MockService.analyze.assert_called_once_with("test/model", layer=5)

    @pytest.mark.asyncio
    async def test_analyze_with_output_file(self, capsys, mock_analysis_result, tmp_path):
        """Test analyze with output file."""
        output_file = tmp_path / "output.json"
        args = Namespace(
            model="test/model",
            layer=None,
            output=str(output_file),
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze.MoETypeService"
        ) as MockService:
            MockService.analyze = AsyncMock(return_value=mock_analysis_result)

            await _async_moe_type_analyze(args)

            captured = capsys.readouterr()
            assert "Saved to:" in captured.out
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_analyze_missing_attributes(self, capsys, mock_analysis_result):
        """Test analyze handles missing optional attributes gracefully."""
        # Simulate args without layer/output attributes
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_type_analyze.MoETypeService"
        ) as MockService:
            MockService.analyze = AsyncMock(return_value=mock_analysis_result)

            # Should not raise even without layer/output attrs
            await _async_moe_type_analyze(args)

            MockService.analyze.assert_called_once_with("test/model", layer=None)
