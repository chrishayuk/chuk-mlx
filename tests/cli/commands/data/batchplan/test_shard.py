"""Tests for batchplan shard command."""

from pathlib import Path
from unittest.mock import patch

import pytest

from chuk_lazarus.cli.commands.data.batchplan._types import BatchPlanShardConfig
from chuk_lazarus.cli.commands.data.batchplan.shard import data_batchplan_shard

LOAD_PLAN_PATCH = "chuk_lazarus.data.batching.load_batch_plan"
SAVE_PLAN_PATCH = "chuk_lazarus.data.batching.save_batch_plan"


class TestDataBatchplanShard:
    """Tests for data_batchplan_shard command."""

    @pytest.mark.asyncio
    async def test_shard_creation(self, tmp_path, mock_batch_plan):
        """Test creating sharded batch plans."""
        output_dir = tmp_path / "shards"
        config = BatchPlanShardConfig(
            plan=Path("/path/to/plan.msgpack"),
            world_size=4,
            output=output_dir,
        )

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(SAVE_PLAN_PATCH, create=True) as mock_save,
        ):
            result = await data_batchplan_shard(config)

            assert result.world_size == 4
            assert len(result.shard_details) == 4
            assert mock_batch_plan.shard.call_count == 4
            assert mock_save.call_count == 4
            assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_shard_display(self, tmp_path, mock_batch_plan):
        """Test shard result display."""
        output_dir = tmp_path / "shards"
        config = BatchPlanShardConfig(
            plan=Path("/path/to/plan.msgpack"),
            world_size=2,
            output=output_dir,
        )

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(SAVE_PLAN_PATCH, create=True),
        ):
            result = await data_batchplan_shard(config)

            display = result.to_display()
            assert "Batch Plan Sharding" in display
            assert "Rank 0:" in display
            assert "Rank 1:" in display
