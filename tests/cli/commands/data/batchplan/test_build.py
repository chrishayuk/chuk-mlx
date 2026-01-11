"""Tests for batchplan build command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batchplan._types import (
    BatchPlanBuildConfig,
    BatchPlanMode,
)
from chuk_lazarus.cli.commands.data.batchplan.build import data_batchplan_build

LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
BUILDER_PATCH = "chuk_lazarus.data.batching.BatchPlanBuilder"
SAVE_PATCH = "chuk_lazarus.data.batching.save_batch_plan"
CONFIG_PATCH = "chuk_lazarus.data.batching.BatchingConfig"


class TestDataBatchplanBuild:
    """Tests for data_batchplan_build command."""

    @pytest.mark.asyncio
    async def test_build_predictable_mode(self, mock_length_cache, mock_batch_plan):
        """Test building batch plan in predictable mode."""
        config = BatchPlanBuildConfig(
            lengths=Path("/path/to/cache.db"),
            bucket_edges="128,256,512",
            token_budget=2048,
            overflow_max=2048,
            predictable=True,
            seed=42,
            epochs=2,
            output=Path("/path/to/plan.msgpack"),
            dataset_hash="hash123",
        )

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUILDER_PATCH, create=True) as mock_builder_cls,
            patch(SAVE_PATCH, create=True),
            patch(CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)
            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_batch_plan)
            mock_builder_cls.return_value = mock_builder

            result = await data_batchplan_build(config)

            assert result.mode == BatchPlanMode.PREDICTABLE
            assert result.epochs == 2
            assert result.total_batches == 50
            mock_config_cls.predictable.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_throughput_mode(self, mock_length_cache, mock_batch_plan):
        """Test building batch plan in throughput mode."""
        config = BatchPlanBuildConfig(
            lengths=Path("/path/to/cache.db"),
            bucket_edges="128,256,512",
            token_budget=4096,
            overflow_max=4096,
            predictable=False,
            seed=None,
            epochs=1,
            output=Path("/path/to/plan.msgpack"),
            dataset_hash=None,
        )

        with (
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUILDER_PATCH, create=True) as mock_builder_cls,
            patch(SAVE_PATCH, create=True),
            patch(CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)
            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_batch_plan)
            mock_builder_cls.return_value = mock_builder

            result = await data_batchplan_build(config)

            assert result.mode == BatchPlanMode.THROUGHPUT
            mock_config_cls.throughput.assert_called_once()
