"""Tests for batch plan sharding utilities."""

from chuk_lazarus.data.batching import (
    BatchPlan,
    BatchPlanMeta,
    EpochPlan,
    MicrobatchSpec,
    PadPolicy,
)
from chuk_lazarus.distributed import (
    DistributedConfig,
    interleave_microbatches,
    shard_batch_plan,
)
from chuk_lazarus.distributed.sharding import compute_shard_stats


def create_test_plan(num_batches: int = 10) -> BatchPlan:
    """Create a test batch plan."""
    meta = BatchPlanMeta(
        dataset_hash="test123",
        tokenizer_hash="tok456",
        bucket_edges=(128, 256, 512),
        overflow_max=1024,
        token_budget=4096,
        pad_policy=PadPolicy.PAD_TO_MAX_IN_BATCH,
        num_epochs=1,
        base_seed=42,
        created_at="2024-01-01T00:00:00Z",
    )

    microbatches = [
        MicrobatchSpec(
            samples=(f"sample_{i}",),
            bucket_id=i % 3,
            max_len=128 * (1 + i % 3),
            index=i,
        )
        for i in range(num_batches)
    ]

    epoch = EpochPlan(
        epoch=0,
        microbatches=tuple(microbatches),
        seed=42,
        total_samples=num_batches,
        total_tokens=num_batches * 100,
    )

    plan = BatchPlan(meta=meta)
    plan.add_epoch(epoch)
    return plan


class TestShardBatchPlan:
    """Tests for shard_batch_plan function."""

    def test_shard_single_worker(self):
        """Test sharding with single worker returns original."""
        plan = create_test_plan(10)
        config = DistributedConfig(rank=0, world_size=1)

        sharded = shard_batch_plan(plan, config)
        assert sharded.total_microbatches == 10

    def test_shard_two_workers(self):
        """Test sharding between two workers."""
        plan = create_test_plan(10)

        # Rank 0 gets indices 0, 2, 4, 6, 8
        config0 = DistributedConfig(rank=0, world_size=2)
        sharded0 = shard_batch_plan(plan, config0)
        assert sharded0.total_microbatches == 5

        # Rank 1 gets indices 1, 3, 5, 7, 9
        config1 = DistributedConfig(rank=1, world_size=2)
        sharded1 = shard_batch_plan(plan, config1)
        assert sharded1.total_microbatches == 5

    def test_shard_four_workers(self):
        """Test sharding between four workers."""
        plan = create_test_plan(12)

        for rank in range(4):
            config = DistributedConfig(rank=rank, world_size=4)
            sharded = shard_batch_plan(plan, config)
            assert sharded.total_microbatches == 3

    def test_shard_uneven(self):
        """Test sharding with uneven distribution."""
        plan = create_test_plan(10)

        # 10 batches / 4 workers = 2 or 3 each
        total = 0
        for rank in range(4):
            config = DistributedConfig(rank=rank, world_size=4)
            sharded = shard_batch_plan(plan, config)
            total += sharded.total_microbatches

        assert total == 10


class TestInterleaveMicrobatches:
    """Tests for interleave_microbatches function."""

    def test_interleave_empty(self):
        """Test interleaving empty list."""
        result = interleave_microbatches([])
        assert result == []

    def test_interleave_single_bucket(self):
        """Test interleaving with single bucket."""
        microbatches = [
            MicrobatchSpec(samples=("a",), bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=("b",), bucket_id=0, max_len=128, index=1),
        ]
        result = interleave_microbatches(microbatches)
        assert len(result) == 2

    def test_interleave_two_buckets(self):
        """Test interleaving between two buckets."""
        microbatches = [
            # Bucket 0
            MicrobatchSpec(samples=("a",), bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=("b",), bucket_id=0, max_len=128, index=1),
            # Bucket 1
            MicrobatchSpec(samples=("c",), bucket_id=1, max_len=256, index=2),
            MicrobatchSpec(samples=("d",), bucket_id=1, max_len=256, index=3),
        ]

        result = interleave_microbatches(microbatches)

        # Should alternate between buckets
        assert len(result) == 4
        bucket_ids = [mb.bucket_id for mb in result]
        # First should be from bucket 0, then 1, then 0, then 1
        assert bucket_ids == [0, 1, 0, 1]

    def test_interleave_uneven_buckets(self):
        """Test interleaving with uneven bucket sizes."""
        microbatches = [
            # Bucket 0: 3 items
            MicrobatchSpec(samples=("a",), bucket_id=0, max_len=128, index=0),
            MicrobatchSpec(samples=("b",), bucket_id=0, max_len=128, index=1),
            MicrobatchSpec(samples=("c",), bucket_id=0, max_len=128, index=2),
            # Bucket 1: 1 item
            MicrobatchSpec(samples=("d",), bucket_id=1, max_len=256, index=3),
        ]

        result = interleave_microbatches(microbatches)
        assert len(result) == 4
        # All items should be present
        samples = [mb.samples[0] for mb in result]
        assert set(samples) == {"a", "b", "c", "d"}


class TestComputeShardStats:
    """Tests for compute_shard_stats function."""

    def test_basic_stats(self):
        """Test computing shard statistics."""
        plan = create_test_plan(10)
        stats = compute_shard_stats(plan, world_size=2)

        assert len(stats) == 2
        assert 0 in stats
        assert 1 in stats

        # Both ranks should have batches
        assert stats[0]["num_batches"] > 0
        assert stats[1]["num_batches"] > 0

        # Total should equal original
        total_batches = stats[0]["num_batches"] + stats[1]["num_batches"]
        assert total_batches == 10
