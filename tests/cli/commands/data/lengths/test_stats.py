"""Tests for lengths stats command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.lengths._types import (
    EmptyStatsResult,
    LengthStatsConfig,
    LengthStatsResult,
)
from chuk_lazarus.cli.commands.data.lengths.stats import data_lengths_stats

LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"


class TestDataLengthsStats:
    """Tests for data_lengths_stats command."""

    @pytest.mark.asyncio
    async def test_stats_with_populated_cache(self, mock_length_cache):
        """Test showing statistics for a populated cache."""
        config = LengthStatsConfig(cache=Path("/path/to/cache.db"))

        with patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls:
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_lengths_stats(config)

            assert isinstance(result, LengthStatsResult)
            assert result.total_samples == 5
            assert result.min_length == 10
            assert result.max_length == 30

    @pytest.mark.asyncio
    async def test_stats_with_empty_cache(self):
        """Test showing statistics for an empty cache."""
        config = LengthStatsConfig(cache=Path("/path/to/cache.db"))

        empty_cache = MagicMock()
        empty_cache.get_all.return_value = {}
        empty_cache.tokenizer_hash = "test_hash"

        with patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls:
            mock_cache_cls.load = AsyncMock(return_value=empty_cache)

            result = await data_lengths_stats(config)

            assert isinstance(result, EmptyStatsResult)
            assert result.to_display() == "Cache is empty"

    @pytest.mark.asyncio
    async def test_stats_percentiles(self, mock_length_cache):
        """Test that percentiles are calculated correctly."""
        config = LengthStatsConfig(cache=Path("/path/to/cache.db"))

        with patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls:
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_lengths_stats(config)

            assert result.p10 >= result.min_length
            assert result.p99 <= result.max_length
            assert result.p50 == result.median_length
