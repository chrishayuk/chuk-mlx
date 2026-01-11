"""Tests for batching suggest command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batching._types import (
    OptimizationGoalType,
    SuggestConfig,
)
from chuk_lazarus.cli.commands.data.batching.suggest import data_batching_suggest

LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
SUGGEST_PATCH = "chuk_lazarus.data.batching.suggest_bucket_edges"


class TestDataBatchingSuggest:
    """Tests for data_batching_suggest command."""

    @pytest.mark.asyncio
    async def test_suggest_minimize_waste(self, mock_length_cache):
        """Test suggesting bucket edges with minimize waste goal."""
        config = SuggestConfig(
            cache=Path("/path/to/cache.db"),
            num_buckets=5,
            goal=OptimizationGoalType.WASTE,
            max_length=2048,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="minimize_waste")
        mock_suggestion.edges = [128, 256, 512, 1024, 2048]
        mock_suggestion.overflow_max = 2048
        mock_suggestion.estimated_efficiency = 0.92
        mock_suggestion.rationale = "Optimized for minimal padding waste"

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(SUGGEST_PATCH, create=True, return_value=mock_suggestion),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_suggest(config)

            assert result.goal == "minimize_waste"
            assert result.edges == [128, 256, 512, 1024, 2048]
            assert result.estimated_efficiency == 0.92

    @pytest.mark.asyncio
    async def test_suggest_balance(self, mock_length_cache):
        """Test suggesting bucket edges with balance goal."""
        config = SuggestConfig(
            cache=Path("/path/to/cache.db"),
            num_buckets=4,
            goal=OptimizationGoalType.BALANCE,
            max_length=1024,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="balance_buckets")
        mock_suggestion.edges = [256, 512, 768, 1024]
        mock_suggestion.overflow_max = 1024
        mock_suggestion.estimated_efficiency = 0.88
        mock_suggestion.rationale = "Balanced distribution"

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(SUGGEST_PATCH, create=True, return_value=mock_suggestion),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_suggest(config)

            assert result.goal == "balance_buckets"

    @pytest.mark.asyncio
    async def test_suggest_memory(self, mock_length_cache):
        """Test suggesting bucket edges with memory goal."""
        config = SuggestConfig(
            cache=Path("/path/to/cache.db"),
            num_buckets=3,
            goal=OptimizationGoalType.MEMORY,
            max_length=512,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="minimize_memory")
        mock_suggestion.edges = [128, 256, 512]
        mock_suggestion.overflow_max = 512
        mock_suggestion.estimated_efficiency = 0.85
        mock_suggestion.rationale = "Memory optimized"

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(SUGGEST_PATCH, create=True, return_value=mock_suggestion),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)

            result = await data_batching_suggest(config)

            assert result.goal == "minimize_memory"
