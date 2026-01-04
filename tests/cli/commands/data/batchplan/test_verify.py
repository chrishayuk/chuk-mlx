"""Tests for batchplan verify command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batchplan._types import BatchPlanVerifyConfig
from chuk_lazarus.cli.commands.data.batchplan.verify import data_batchplan_verify

LOAD_PLAN_PATCH = "chuk_lazarus.data.batching.load_batch_plan"
LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
BUILDER_PATCH = "chuk_lazarus.data.batching.BatchPlanBuilder"
CONFIG_PATCH = "chuk_lazarus.data.batching.BatchingConfig"


class TestDataBatchplanVerify:
    """Tests for data_batchplan_verify command."""

    @pytest.mark.asyncio
    async def test_verify_matching_fingerprints(self, mock_length_cache, mock_batch_plan):
        """Test verification with matching fingerprints."""
        config = BatchPlanVerifyConfig(
            plan=Path("/path/to/plan.msgpack"),
            lengths=Path("/path/to/cache.db"),
        )

        # Create a rebuilt plan with same fingerprint
        rebuilt_plan = MagicMock()
        rebuilt_plan.fingerprint = mock_batch_plan.fingerprint

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUILDER_PATCH, create=True) as mock_builder_cls,
            patch(CONFIG_PATCH, create=True),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)
            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=rebuilt_plan)
            mock_builder_cls.return_value = mock_builder

            result = await data_batchplan_verify(config)

            assert result.match is True
            assert "MATCH" in result.to_display()

    @pytest.mark.asyncio
    async def test_verify_mismatching_fingerprints(self, mock_length_cache, mock_batch_plan):
        """Test verification with mismatching fingerprints."""
        config = BatchPlanVerifyConfig(
            plan=Path("/path/to/plan.msgpack"),
            lengths=Path("/path/to/cache.db"),
        )

        # Create a rebuilt plan with different fingerprint
        rebuilt_plan = MagicMock()
        rebuilt_plan.fingerprint = "different_fingerprint"
        rebuilt_plan.num_epochs = mock_batch_plan.num_epochs
        rebuilt_plan.iter_epoch.return_value = iter([MagicMock()] * 20)

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
            patch(BUILDER_PATCH, create=True) as mock_builder_cls,
            patch(CONFIG_PATCH, create=True),
        ):
            mock_cache_cls.load = AsyncMock(return_value=mock_length_cache)
            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=rebuilt_plan)
            mock_builder_cls.return_value = mock_builder

            result = await data_batchplan_verify(config)

            assert result.match is False
            assert "MISMATCH" in result.to_display()
