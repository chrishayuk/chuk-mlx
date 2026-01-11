"""Tests for batching histogram command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batching._types import HistogramConfig
from chuk_lazarus.cli.commands.data.batching.histogram import data_batching_histogram

LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
HISTOGRAM_PATCH = "chuk_lazarus.data.batching.compute_length_histogram"


class TestDataBatchingHistogram:
    """Tests for data_batching_histogram command."""

    @pytest.mark.asyncio
    async def test_histogram_display(self, mock_length_cache):
        """Test displaying length histogram."""
        config = HistogramConfig(
            cache=Path("/path/to/cache.db"),
            bins=20,
            width=80,
        )

        mock_histogram = MagicMock()
        mock_histogram.to_ascii.return_value = "Histogram ASCII Art"
        mock_histogram.p25 = 10
        mock_histogram.p50 = 20
        mock_histogram.p75 = 30
        mock_histogram.p90 = 40
        mock_histogram.p95 = 45
        mock_histogram.p99 = 50

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(HISTOGRAM_PATCH, create=True, return_value=mock_histogram),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_histogram(config)

            assert result.histogram_ascii == "Histogram ASCII Art"
            assert result.p25 == 10
            assert result.p99 == 50

    @pytest.mark.asyncio
    async def test_histogram_with_custom_bins(self, mock_length_cache):
        """Test histogram with custom bin count."""
        config = HistogramConfig(
            cache=Path("/path/to/cache.db"),
            bins=50,
            width=100,
        )

        mock_histogram = MagicMock()
        mock_histogram.to_ascii.return_value = "Histogram"
        mock_histogram.p25 = 25
        mock_histogram.p50 = 50
        mock_histogram.p75 = 75
        mock_histogram.p90 = 90
        mock_histogram.p95 = 95
        mock_histogram.p99 = 99

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(HISTOGRAM_PATCH, create=True, return_value=mock_histogram) as mock_compute,
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            await data_batching_histogram(config)

            mock_compute.assert_called_once()
            call_args = mock_compute.call_args
            assert call_args.kwargs["num_bins"] == 50
