"""Tests for batchplan info command."""

from pathlib import Path
from unittest.mock import patch

import pytest

from chuk_lazarus.cli.commands.data.batchplan._types import (
    BatchPlanInfoConfig,
    BatchPlanInfoResult,
    InvalidRankError,
)
from chuk_lazarus.cli.commands.data.batchplan.info import data_batchplan_info

LOAD_PLAN_PATCH = "chuk_lazarus.data.batching.load_batch_plan"


class TestDataBatchplanInfo:
    """Tests for data_batchplan_info command."""

    @pytest.mark.asyncio
    async def test_info_without_sharding(self, mock_batch_plan):
        """Test showing batch plan info without sharding."""
        config = BatchPlanInfoConfig(
            plan=Path("/path/to/plan.msgpack"),
            rank=None,
            world_size=None,
            show_batches=None,
        )

        with patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan):
            result = await data_batchplan_info(config)

            assert isinstance(result, BatchPlanInfoResult)
            assert result.epochs == 2
            assert result.shard_info is None

    @pytest.mark.asyncio
    async def test_info_with_sharding(self, mock_batch_plan):
        """Test showing batch plan info with sharding."""
        config = BatchPlanInfoConfig(
            plan=Path("/path/to/plan.msgpack"),
            rank=1,
            world_size=4,
            show_batches=None,
        )

        with patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan):
            result = await data_batchplan_info(config)

            assert isinstance(result, BatchPlanInfoResult)
            assert result.shard_info == "rank 1/4"
            mock_batch_plan.shard.assert_called_once_with(1, 4)

    @pytest.mark.asyncio
    async def test_info_with_invalid_rank(self, mock_batch_plan):
        """Test info with invalid rank returns error."""
        config = BatchPlanInfoConfig(
            plan=Path("/path/to/plan.msgpack"),
            rank=5,
            world_size=4,
            show_batches=None,
        )

        with patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan):
            result = await data_batchplan_info(config)

            assert isinstance(result, InvalidRankError)
            assert "Error:" in result.to_display()

    def test_info_with_negative_rank_validation(self):
        """Test that negative rank fails Pydantic validation."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPlanInfoConfig(
                plan=Path("/path/to/plan.msgpack"),
                rank=-1,
                world_size=4,
                show_batches=None,
            )

    @pytest.mark.asyncio
    async def test_info_show_batches(self, mock_batch_plan):
        """Test showing sample batches."""
        config = BatchPlanInfoConfig(
            plan=Path("/path/to/plan.msgpack"),
            rank=None,
            world_size=None,
            show_batches=3,
        )

        with patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan):
            result = await data_batchplan_info(config)

            assert isinstance(result, BatchPlanInfoResult)
            assert len(result.sample_batches) == 3
