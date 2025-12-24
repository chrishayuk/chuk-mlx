"""Tests for async token-budget batch sampler."""

import asyncio

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import (
    BatchSpec,
    BucketSpec,
    TokenBudgetBatchSampler,
)


class TestBatchSpec:
    """Tests for BatchSpec model."""

    def test_create(self):
        """Test creating batch spec."""
        spec = BatchSpec(
            sample_ids=("s1", "s2", "s3"),
            bucket_id=1,
            max_length=256,
            token_count=600,
        )
        assert spec.batch_size == 3
        assert spec.max_length == 256
        assert spec.token_count == 600

    def test_padded_token_count(self):
        """Test padded token count calculation."""
        spec = BatchSpec(
            sample_ids=("s1", "s2", "s3"),
            bucket_id=1,
            max_length=256,
            token_count=600,
        )
        assert spec.padded_token_count == 3 * 256  # 768

    def test_immutable(self):
        """Test that batch spec is immutable."""
        spec = BatchSpec(
            sample_ids=("s1", "s2"),
            bucket_id=0,
            max_length=128,
            token_count=200,
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            spec.max_length = 512


class TestTokenBudgetBatchSampler:
    """Tests for TokenBudgetBatchSampler."""

    def test_create_basic(self):
        """Test creating sampler."""
        lengths = {f"s{i}": 100 + i for i in range(10)}
        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=1024,
            seed=42,
        )

        assert sampler.num_samples == 10
        assert sampler.num_skipped == 0

    def test_skip_out_of_range(self):
        """Test that out-of-range samples are skipped."""
        lengths = {
            "too_short": 5,
            "ok_1": 100,
            "ok_2": 200,
            "too_long": 1000,
        }
        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512, min_length=10)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=1024,
        )

        assert sampler.num_samples == 2
        assert sampler.num_skipped == 2

    def test_bucket_assignment(self):
        """Test samples are assigned to correct buckets."""
        lengths = {
            "bucket_0_a": 50,
            "bucket_0_b": 100,
            "bucket_1_a": 150,
            "bucket_1_b": 200,
            "bucket_2_a": 300,
        }
        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=1024,
        )

        sizes = sampler.bucket_sizes()
        assert sizes[0] == 2  # 50, 100
        assert sizes[1] == 2  # 150, 200
        assert sizes[2] == 1  # 300

    def test_iter_epoch_basic(self):
        """Test basic epoch iteration."""

        async def run():
            lengths = {f"s{i}": 100 for i in range(10)}
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,  # Fits ~4 samples per batch
            )

            batches = []
            async for batch in sampler.iter_epoch(epoch=0):
                batches.append(batch)

            # Should have batches covering all 10 samples
            total_samples = sum(b.batch_size for b in batches)
            assert total_samples == 10

        asyncio.run(run())

    def test_iter_epoch_deterministic(self):
        """Test that epoch iteration is deterministic."""

        async def run():
            lengths = {f"s{i}": 100 for i in range(20)}
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,
                seed=42,
            )

            # Run same epoch twice
            batches_1 = []
            async for batch in sampler.iter_epoch(epoch=0):
                batches_1.append(batch.sample_ids)

            batches_2 = []
            async for batch in sampler.iter_epoch(epoch=0):
                batches_2.append(batch.sample_ids)

            assert batches_1 == batches_2

        asyncio.run(run())

    def test_different_epochs_different_order(self):
        """Test that different epochs have different order."""

        async def run():
            lengths = {f"s{i}": 100 for i in range(20)}
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,
                seed=42,
            )

            batches_epoch_0 = []
            async for batch in sampler.iter_epoch(epoch=0):
                batches_epoch_0.extend(batch.sample_ids)

            batches_epoch_1 = []
            async for batch in sampler.iter_epoch(epoch=1):
                batches_epoch_1.extend(batch.sample_ids)

            # Same samples, different order
            assert set(batches_epoch_0) == set(batches_epoch_1)
            assert batches_epoch_0 != batches_epoch_1

        asyncio.run(run())

    def test_token_budget_respected(self):
        """Test that token budget is respected."""

        async def run():
            lengths = {f"s{i}": 100 for i in range(20)}
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,  # Max 4 samples * 128 = 512
            )

            async for batch in sampler.iter_epoch(epoch=0):
                # Each batch should fit in budget
                assert batch.padded_token_count <= 512

        asyncio.run(run())

    def test_multiple_buckets(self):
        """Test iteration with multiple buckets."""

        async def run():
            lengths = {
                **{f"short_{i}": 50 for i in range(5)},
                **{f"medium_{i}": 150 for i in range(5)},
                **{f"long_{i}": 300 for i in range(5)},
            }
            bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=1024,
            )

            batches = []
            async for batch in sampler.iter_epoch(epoch=0):
                batches.append(batch)

            # All samples should be covered
            all_samples = set()
            for batch in batches:
                all_samples.update(batch.sample_ids)

            assert len(all_samples) == 15

        asyncio.run(run())

    def test_interleave_buckets(self):
        """Test bucket interleaving."""

        async def run():
            lengths = {
                **{f"bucket_0_{i}": 50 for i in range(10)},
                **{f"bucket_1_{i}": 150 for i in range(10)},
            }
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,
            )

            # With interleaving
            batches_interleaved = []
            async for batch in sampler.iter_epoch(epoch=0, interleave_buckets=True):
                batches_interleaved.append(batch.bucket_id)

            # Without interleaving
            batches_sequential = []
            async for batch in sampler.iter_epoch(epoch=0, interleave_buckets=False):
                batches_sequential.append(batch.bucket_id)

            # Sequential should have all bucket 0 first, then bucket 1
            # Interleaved should mix them
            if len(set(batches_sequential)) > 1:
                # Find transition point in sequential
                first_bucket = batches_sequential[0]
                transition_idx = next(
                    (i for i, b in enumerate(batches_sequential) if b != first_bucket),
                    len(batches_sequential),
                )
                # All before transition should be same bucket
                assert all(b == first_bucket for b in batches_sequential[:transition_idx])

        asyncio.run(run())

    def test_drop_last(self):
        """Test drop_last option."""

        async def run():
            # 7 samples, budget fits 4 per batch -> 1 full batch + 3 leftover
            lengths = {f"s{i}": 100 for i in range(7)}
            bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

            sampler_keep = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,  # Fits 4 samples
                drop_last=False,
            )

            sampler_drop = TokenBudgetBatchSampler(
                lengths=lengths,
                bucket_spec=bucket_spec,
                token_budget=512,
                drop_last=True,
            )

            batches_keep = [b async for b in sampler_keep.iter_epoch(epoch=0)]
            batches_drop = [b async for b in sampler_drop.iter_epoch(epoch=0)]

            samples_keep = sum(b.batch_size for b in batches_keep)
            samples_drop = sum(b.batch_size for b in batches_drop)

            # Keep last should have all 7
            assert samples_keep == 7
            # Drop last should only have full batches (4)
            assert samples_drop == 4

        asyncio.run(run())

    def test_estimate_batches_per_epoch(self):
        """Test batch count estimation."""
        lengths = {f"s{i}": 100 for i in range(20)}
        bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=512,  # Fits ~4 samples
        )

        estimate = sampler.estimate_batches_per_epoch()
        # 20 samples / 4 per batch = 5 batches
        assert estimate == 5

    def test_compute_metrics(self):
        """Test metrics computation."""
        lengths = {
            "short_1": 50,
            "short_2": 80,
            "long_1": 150,
        }
        bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=1024,
        )

        metrics = sampler.compute_metrics()

        assert metrics.total_samples == 3
        assert metrics.total_tokens == 50 + 80 + 150  # 280
        # 50, 80 -> bucket 0 (max 128), 150 -> bucket 1 (max 256)
        assert metrics.padded_tokens == 128 + 128 + 256  # 512
        assert metrics.padding_waste > 0

    def test_get_bucket_distribution(self):
        """Test bucket distribution calculation."""
        lengths = {
            **{f"bucket_0_{i}": 50 for i in range(6)},
            **{f"bucket_1_{i}": 150 for i in range(4)},
        }
        bucket_spec = BucketSpec(edges=(128,), overflow_max=256)

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=1024,
        )

        dist = sampler.get_bucket_distribution()
        assert abs(dist[0] - 0.6) < 0.01  # 6/10
        assert abs(dist[1] - 0.4) < 0.01  # 4/10
