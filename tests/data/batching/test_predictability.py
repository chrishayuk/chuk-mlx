"""Tests for predictability mode (Phase 2)."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import (
    BatchFingerprint,
    BatchingConfig,
    BatchingMode,
    BatchSpec,
    BucketSpec,
    PadPolicy,
    TokenBudgetBatchSampler,
    compute_batch_fingerprint,
    verify_batch_fingerprint,
)
from chuk_lazarus.data.batching.planning.predictability import (
    load_fingerprint,
    save_fingerprint,
)


class TestPadPolicy:
    """Tests for PadPolicy enum."""

    def test_values(self):
        assert PadPolicy.PAD_TO_BUCKET.value == "pad_to_bucket"
        assert PadPolicy.PAD_TO_MAX_IN_BATCH.value == "pad_to_max"

    def test_enum_members(self):
        assert len(list(PadPolicy)) == 2


class TestBatchingMode:
    """Tests for BatchingMode enum."""

    def test_values(self):
        assert BatchingMode.PREDICTABLE.value == "predictable"
        assert BatchingMode.THROUGHPUT.value == "throughput"

    def test_enum_members(self):
        assert len(list(BatchingMode)) == 2


class TestBatchingConfig:
    """Tests for BatchingConfig."""

    def test_create_default(self):
        config = BatchingConfig()
        assert config.mode == BatchingMode.THROUGHPUT
        assert config.pad_policy == PadPolicy.PAD_TO_MAX_IN_BATCH
        assert config.token_budget == 4096
        assert config.bucket_edges == (128, 256, 512, 1024)
        assert config.overflow_max == 2048
        assert config.seed == 42

    def test_predictable_factory(self):
        config = BatchingConfig.predictable()
        assert config.mode == BatchingMode.PREDICTABLE
        assert config.pad_policy == PadPolicy.PAD_TO_BUCKET
        assert config.drop_last is True
        assert config.interleave_buckets is False
        assert config.is_predictable is True

    def test_throughput_factory(self):
        config = BatchingConfig.throughput()
        assert config.mode == BatchingMode.THROUGHPUT
        assert config.pad_policy == PadPolicy.PAD_TO_MAX_IN_BATCH
        assert config.drop_last is False
        assert config.interleave_buckets is True
        assert config.is_predictable is False

    def test_custom_params(self):
        config = BatchingConfig.predictable(
            token_budget=8192,
            bucket_edges=(256, 512),
            overflow_max=1024,
            seed=123,
        )
        assert config.token_budget == 8192
        assert config.bucket_edges == (256, 512)
        assert config.overflow_max == 1024
        assert config.seed == 123

    def test_immutable(self):
        config = BatchingConfig()
        with pytest.raises((TypeError, AttributeError, ValidationError)):  # Frozen
            config.token_budget = 1000

    def test_is_predictable_property(self):
        assert BatchingConfig.predictable().is_predictable is True
        assert BatchingConfig.throughput().is_predictable is False

    def test_get_pad_length_bucket_mode(self):
        config = BatchingConfig.predictable(bucket_edges=(128, 256, 512))

        # Bucket 0 -> 128
        assert config.get_pad_length(bucket_id=0, max_in_batch=100) == 128
        # Bucket 1 -> 256
        assert config.get_pad_length(bucket_id=1, max_in_batch=200) == 256
        # Bucket 2 -> 512
        assert config.get_pad_length(bucket_id=2, max_in_batch=400) == 512
        # Overflow -> overflow_max
        assert config.get_pad_length(bucket_id=3, max_in_batch=600) == 2048

    def test_get_pad_length_throughput_mode(self):
        config = BatchingConfig.throughput(bucket_edges=(128, 256, 512))

        # Throughput mode uses max in batch
        assert config.get_pad_length(bucket_id=0, max_in_batch=100) == 100
        assert config.get_pad_length(bucket_id=1, max_in_batch=200) == 200
        assert config.get_pad_length(bucket_id=2, max_in_batch=450) == 450

    def test_get_pad_length_with_overrides(self):
        config = BatchingConfig(
            mode=BatchingMode.PREDICTABLE,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            bucket_edges=(128, 256),
            bucket_max_lengths={0: 100, 1: 200},  # Custom overrides
        )

        assert config.get_pad_length(bucket_id=0, max_in_batch=80) == 100
        assert config.get_pad_length(bucket_id=1, max_in_batch=180) == 200


class TestBatchFingerprint:
    """Tests for BatchFingerprint."""

    def test_create(self):
        fp = BatchFingerprint(
            fingerprint="abc123",
            full_hash="abc123" * 10 + "abcd",  # 64 chars
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="config123",
        )
        assert fp.fingerprint == "abc123"
        assert fp.num_batches == 10
        assert fp.total_samples == 100
        assert fp.total_tokens == 5000

    def test_matches(self):
        fp1 = BatchFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="c1",
        )
        fp2 = BatchFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="c1",
        )
        assert fp1.matches(fp2)

    def test_not_matches(self):
        fp1 = BatchFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="c1",
        )
        fp2 = BatchFingerprint(
            fingerprint="def",
            full_hash="def456",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="c1",
        )
        assert not fp1.matches(fp2)

    def test_matches_config(self):
        fp1 = BatchFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="same_config",
        )
        fp2 = BatchFingerprint(
            fingerprint="def",
            full_hash="def456",
            num_batches=20,
            total_samples=200,
            total_tokens=10000,
            config_hash="same_config",
        )
        assert fp1.matches_config(fp2)


class TestComputeBatchFingerprint:
    """Tests for compute_batch_fingerprint."""

    def test_basic_fingerprint(self):
        batches = [
            BatchSpec(
                sample_ids=("s1", "s2"),
                bucket_id=0,
                max_length=128,
                token_count=200,
            ),
            BatchSpec(
                sample_ids=("s3", "s4"),
                bucket_id=0,
                max_length=128,
                token_count=220,
            ),
        ]

        fp = compute_batch_fingerprint(batches)

        assert isinstance(fp, BatchFingerprint)
        assert len(fp.fingerprint) == 16
        assert len(fp.full_hash) == 64
        assert fp.num_batches == 2
        assert fp.total_samples == 4
        assert fp.total_tokens == 420

    def test_fingerprint_deterministic(self):
        batches = [
            BatchSpec(
                sample_ids=("s1", "s2"),
                bucket_id=0,
                max_length=128,
                token_count=200,
            ),
        ]

        fp1 = compute_batch_fingerprint(batches)
        fp2 = compute_batch_fingerprint(batches)

        assert fp1.fingerprint == fp2.fingerprint
        assert fp1.full_hash == fp2.full_hash

    def test_different_batches_different_fingerprint(self):
        batches1 = [
            BatchSpec(
                sample_ids=("s1", "s2"),
                bucket_id=0,
                max_length=128,
                token_count=200,
            ),
        ]
        batches2 = [
            BatchSpec(
                sample_ids=("s3", "s4"),
                bucket_id=0,
                max_length=128,
                token_count=200,
            ),
        ]

        fp1 = compute_batch_fingerprint(batches1)
        fp2 = compute_batch_fingerprint(batches2)

        assert fp1.fingerprint != fp2.fingerprint

    def test_with_config(self):
        batches = [
            BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]
        config = BatchingConfig.predictable()

        fp = compute_batch_fingerprint(batches, config=config)

        assert fp.config_hash != "none"

    def test_n_batches_limit(self):
        batches = [
            BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
            BatchSpec(
                sample_ids=("s2",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
            BatchSpec(
                sample_ids=("s3",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]

        fp_all = compute_batch_fingerprint(batches)
        fp_limited = compute_batch_fingerprint(batches, n_batches=2)

        assert fp_all.num_batches == 3
        assert fp_limited.num_batches == 2
        assert fp_all.fingerprint != fp_limited.fingerprint


class TestVerifyBatchFingerprint:
    """Tests for verify_batch_fingerprint."""

    def test_verify_match(self):
        batches = [
            BatchSpec(
                sample_ids=("s1", "s2"),
                bucket_id=0,
                max_length=128,
                token_count=200,
            ),
        ]

        expected = compute_batch_fingerprint(batches)
        matches, error = verify_batch_fingerprint(batches, expected)

        assert matches is True
        assert error is None

    def test_verify_mismatch(self):
        batches1 = [
            BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]
        batches2 = [
            BatchSpec(
                sample_ids=("s2",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]

        expected = compute_batch_fingerprint(batches1)
        matches, error = verify_batch_fingerprint(batches2, expected)

        assert matches is False
        assert error is not None
        assert "mismatch" in error.lower()

    def test_verify_with_string_fingerprint(self):
        batches = [
            BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]

        fp = compute_batch_fingerprint(batches)
        matches, error = verify_batch_fingerprint(batches, fp.fingerprint)

        assert matches is True
        assert error is None

    def test_verify_string_mismatch(self):
        batches = [
            BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            ),
        ]

        matches, error = verify_batch_fingerprint(batches, "nonexistent_fp")

        assert matches is False
        assert error is not None


class TestSaveLoadFingerprint:
    """Tests for save/load fingerprint."""

    def test_roundtrip(self):
        fp = BatchFingerprint(
            fingerprint="abc123",
            full_hash="abc123" * 10 + "abcd",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="config123",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_fingerprint(fp, path)
            loaded = load_fingerprint(path)

            assert loaded.fingerprint == fp.fingerprint
            assert loaded.full_hash == fp.full_hash
            assert loaded.num_batches == fp.num_batches
            assert loaded.total_samples == fp.total_samples
        finally:
            Path(path).unlink()

    def test_save_creates_valid_json(self):
        fp = BatchFingerprint(
            fingerprint="abc123",
            full_hash="abc123" * 10 + "abcd",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="config123",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_fingerprint(fp, path)

            with open(path) as f:
                data = json.load(f)

            assert "fingerprint" in data
            assert "full_hash" in data
            assert "num_batches" in data
        finally:
            Path(path).unlink()


class TestAsyncFingerprint:
    """Tests for async fingerprint operations."""

    @pytest.mark.asyncio
    async def test_compute_async(self):
        from chuk_lazarus.data.batching import compute_batch_fingerprint_async

        async def batch_gen():
            yield BatchSpec(
                sample_ids=("s1",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            )
            yield BatchSpec(
                sample_ids=("s2",),
                bucket_id=0,
                max_length=128,
                token_count=100,
            )

        fp = await compute_batch_fingerprint_async(batch_gen())

        assert isinstance(fp, BatchFingerprint)
        assert fp.num_batches == 2
        assert fp.total_samples == 2

    @pytest.mark.asyncio
    async def test_save_load_async(self):
        from chuk_lazarus.data.batching.planning.predictability import (
            load_fingerprint_async,
            save_fingerprint_async,
        )

        fp = BatchFingerprint(
            fingerprint="abc123",
            full_hash="abc123" * 10 + "abcd",
            num_batches=10,
            total_samples=100,
            total_tokens=5000,
            config_hash="config123",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            await save_fingerprint_async(fp, path)
            loaded = await load_fingerprint_async(path)

            assert loaded.fingerprint == fp.fingerprint
            assert loaded.full_hash == fp.full_hash
        finally:
            Path(path).unlink()


class TestIntegrationWithSampler:
    """Integration tests with TokenBudgetBatchSampler."""

    @pytest.mark.asyncio
    async def test_sampler_fingerprint_deterministic(self):
        """Verify that sampler produces deterministic fingerprints."""
        lengths = {f"s{i}": 100 + i * 10 for i in range(20)}
        bucket_spec = BucketSpec.default()

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=4096,
            seed=42,
        )

        # Collect batches from epoch 0
        batches1 = [batch async for batch in sampler.iter_epoch(epoch=0)]

        # Collect batches from epoch 0 again
        batches2 = [batch async for batch in sampler.iter_epoch(epoch=0)]

        fp1 = compute_batch_fingerprint(batches1)
        fp2 = compute_batch_fingerprint(batches2)

        assert fp1.matches(fp2)
        assert fp1.num_batches == fp2.num_batches
        assert fp1.total_samples == fp2.total_samples

    @pytest.mark.asyncio
    async def test_different_epochs_different_fingerprint(self):
        """Different epochs should produce different fingerprints."""
        lengths = {f"s{i}": 100 + i * 10 for i in range(20)}
        bucket_spec = BucketSpec.default()

        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            bucket_spec=bucket_spec,
            token_budget=4096,
            seed=42,
        )

        batches_e0 = [batch async for batch in sampler.iter_epoch(epoch=0)]
        batches_e1 = [batch async for batch in sampler.iter_epoch(epoch=1)]

        fp0 = compute_batch_fingerprint(batches_e0)
        fp1 = compute_batch_fingerprint(batches_e1)

        # Same number of batches, different order
        assert fp0.num_batches == fp1.num_batches
        assert not fp0.matches(fp1)
