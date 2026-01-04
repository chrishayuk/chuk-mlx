"""Tests for batching analyze command."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batching._types import AnalyzeConfig
from chuk_lazarus.cli.commands.data.batching.analyze import data_batching_analyze

LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
BUCKET_SPEC_PATCH = "chuk_lazarus.data.batching.BucketSpec"
REPORT_PATCH = "chuk_lazarus.data.batching.create_efficiency_report"


class TestDataBatchingAnalyze:
    """Tests for data_batching_analyze command."""

    @pytest.mark.asyncio
    async def test_analyze_efficiency(self, mock_length_cache):
        """Test analyzing batching efficiency."""
        config = AnalyzeConfig(
            cache=Path("/path/to/cache.db"),
            bucket_edges="128,256,512",
            overflow_max=1024,
            output=None,
        )

        mock_report = MagicMock()
        mock_report.to_ascii.return_value = "Efficiency Report ASCII"
        mock_report.model_dump.return_value = {"efficiency": 0.85}

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUCKET_SPEC_PATCH, create=True),
            patch(REPORT_PATCH, create=True, return_value=mock_report),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_analyze(config)

            assert "Efficiency Report ASCII" in result.report_ascii
            assert result.output_path is None

    @pytest.mark.asyncio
    async def test_analyze_with_output_file(self, tmp_path, mock_length_cache):
        """Test analyzing efficiency with JSON output."""
        output_file = tmp_path / "report.json"
        config = AnalyzeConfig(
            cache=Path("/path/to/cache.db"),
            bucket_edges="128,256",
            overflow_max=512,
            output=output_file,
        )

        mock_report = MagicMock()
        mock_report.to_ascii.return_value = "Report"
        mock_report.model_dump.return_value = {"efficiency": 0.90}

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUCKET_SPEC_PATCH, create=True),
            patch(REPORT_PATCH, create=True, return_value=mock_report),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_analyze(config)

            assert result.output_path == output_file
            assert output_file.exists()
            with open(output_file) as f:
                data = json.load(f)
                assert data["efficiency"] == 0.90
