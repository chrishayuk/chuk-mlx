"""Tests for batchplan types."""

from argparse import Namespace
from pathlib import Path

from chuk_lazarus.cli.commands.data.batchplan._types import (
    BatchPlanBuildConfig,
    BatchPlanBuildResult,
    BatchPlanInfoConfig,
    BatchPlanMode,
    BatchPlanShardResult,
    BatchPlanVerifyResult,
    InvalidRankError,
)


class TestBatchPlanBuildConfig:
    """Tests for BatchPlanBuildConfig."""

    def test_from_args_predictable(self):
        """Test creating config in predictable mode."""
        args = Namespace(
            lengths="/path/to/cache.db",
            bucket_edges="128,256,512",
            token_budget=2048,
            overflow_max=2048,
            predictable=True,
            seed=42,
            epochs=2,
            output="/path/to/plan.msgpack",
            dataset_hash="hash123",
        )
        config = BatchPlanBuildConfig.from_args(args)

        assert config.predictable is True
        assert config.mode == BatchPlanMode.PREDICTABLE
        assert config.seed == 42

    def test_from_args_throughput(self):
        """Test creating config in throughput mode."""
        args = Namespace(
            lengths="/path/to/cache.db",
            bucket_edges="128,256,512",
            token_budget=2048,
            overflow_max=2048,
            predictable=False,
            seed=None,
            epochs=1,
            output="/path/to/plan.msgpack",
            dataset_hash=None,
        )
        config = BatchPlanBuildConfig.from_args(args)

        assert config.predictable is False
        assert config.mode == BatchPlanMode.THROUGHPUT

    def test_get_bucket_edges(self):
        """Test parsing bucket edges."""
        args = Namespace(
            lengths="/path/to/cache.db",
            bucket_edges="128, 256, 512, 1024",
            token_budget=2048,
            overflow_max=2048,
            predictable=True,
            seed=42,
            epochs=1,
            output="/path/to/plan.msgpack",
            dataset_hash=None,
        )
        config = BatchPlanBuildConfig.from_args(args)

        assert config.get_bucket_edges() == (128, 256, 512, 1024)


class TestBatchPlanBuildResult:
    """Tests for BatchPlanBuildResult."""

    def test_to_display(self):
        """Test display formatting."""
        result = BatchPlanBuildResult(
            lengths_cache="/path/to/cache.db",
            epochs=2,
            token_budget=2048,
            mode=BatchPlanMode.PREDICTABLE,
            total_batches=100,
            fingerprint="abc123",
            output_path=Path("/path/to/plan.msgpack"),
            epoch_details=[
                {"epoch": 0, "batches": 50, "samples": 500, "tokens": 10000},
                {"epoch": 1, "batches": 50, "samples": 500, "tokens": 10000},
            ],
        )
        display = result.to_display()

        assert "Batch Plan Built" in display
        assert "predictable" in display
        assert "100" in display
        assert "Epoch 0:" in display
        assert "Epoch 1:" in display


class TestBatchPlanInfoConfig:
    """Tests for BatchPlanInfoConfig."""

    def test_from_args_basic(self):
        """Test basic args parsing."""
        args = Namespace(
            plan="/path/to/plan.msgpack",
            rank=None,
            world_size=None,
            show_batches=None,
        )
        config = BatchPlanInfoConfig.from_args(args)

        assert config.plan == Path("/path/to/plan.msgpack")
        assert config.rank is None

    def test_from_args_with_sharding(self):
        """Test args with sharding."""
        args = Namespace(
            plan="/path/to/plan.msgpack",
            rank=1,
            world_size=4,
            show_batches=5,
        )
        config = BatchPlanInfoConfig.from_args(args)

        assert config.rank == 1
        assert config.world_size == 4
        assert config.show_batches == 5


class TestInvalidRankError:
    """Tests for InvalidRankError."""

    def test_to_display(self):
        """Test error display."""
        error = InvalidRankError(rank=5, world_size=4)
        assert "Error:" in error.to_display()
        assert "0, 4" in error.to_display()


class TestBatchPlanVerifyResult:
    """Tests for BatchPlanVerifyResult."""

    def test_matching_fingerprints(self):
        """Test display when fingerprints match."""
        result = BatchPlanVerifyResult(
            original_fingerprint="abc123",
            rebuilt_fingerprint="abc123",
            match=True,
            epoch_comparison=[],
        )
        display = result.to_display()

        assert "MATCH" in display
        assert "reproducible" in display

    def test_mismatching_fingerprints(self):
        """Test display when fingerprints don't match."""
        result = BatchPlanVerifyResult(
            original_fingerprint="abc123",
            rebuilt_fingerprint="def456",
            match=False,
            epoch_comparison=[{"epoch": 0, "count_differs": False, "matches": 20, "total": 25}],
        )
        display = result.to_display()

        assert "MISMATCH" in display
        assert "Warning" in display


class TestBatchPlanShardResult:
    """Tests for BatchPlanShardResult."""

    def test_to_display(self):
        """Test display formatting."""
        result = BatchPlanShardResult(
            source_plan="/path/to/plan.msgpack",
            world_size=4,
            total_batches=100,
            shard_details=[
                {"rank": 0, "batches": 25, "path": "/out/rank_0"},
                {"rank": 1, "batches": 25, "path": "/out/rank_1"},
            ],
            output_dir=Path("/out"),
        )
        display = result.to_display()

        assert "Batch Plan Sharding" in display
        assert "World size:    4" in display
        assert "Rank 0:" in display
        assert "Rank 1:" in display
