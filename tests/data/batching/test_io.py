"""Tests for batching I/O (BatchWriter and BatchReader)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chuk_lazarus.data.batching import (
    BatchingConfig,
    BatchPlanBuilder,
    BatchReader,
    BatchWriter,
    CollatedBatch,
    default_collate,
)


class TestBatchWriter:
    """Tests for BatchWriter class."""

    @pytest.fixture
    def sample_plan_and_data(self):
        """Create a sample plan and data for testing."""
        # Sample lengths and data - enough samples to form complete batches
        lengths = {f"s{i}": 50 + (i * 10) for i in range(20)}
        samples = {
            sid: {"input_ids": list(range(length)), "loss_mask": [1] * length}
            for sid, length in lengths.items()
        }

        config = BatchingConfig.predictable(
            token_budget=512,
            bucket_edges=(128, 256),
            overflow_max=512,
            seed=42,
        )

        plan = asyncio.run(
            BatchPlanBuilder(
                lengths=lengths,
                batching_config=config,
                dataset_hash="test_v1",
                tokenizer_hash="test_tok_v1",
            ).build(num_epochs=2)
        )

        return plan, samples

    def test_write_all(self, sample_plan_and_data):
        """Test writing all epochs."""
        plan, samples = sample_plan_and_data

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BatchWriter(plan, samples, tmpdir)
            files = writer.write_all()

            # Check files were created
            assert len(files) > 0

            # Check manifest exists
            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

            # Check plan was saved
            plan_dir = Path(tmpdir) / "plan"
            assert plan_dir.exists()

            # Check epoch directories exist
            for epoch in range(plan.num_epochs):
                epoch_dir = Path(tmpdir) / f"epoch_{epoch}"
                assert epoch_dir.exists()

    def test_write_epoch(self, sample_plan_and_data):
        """Test writing a single epoch."""
        plan, samples = sample_plan_and_data

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BatchWriter(plan, samples, tmpdir)
            files = writer.write_epoch(0)

            # Check files were created
            assert len(files) > 0

            # All files should be in epoch_0
            for f in files:
                assert "epoch_0" in str(f)

    def test_batch_content(self, sample_plan_and_data):
        """Test that batch files contain correct content."""
        plan, samples = sample_plan_and_data

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BatchWriter(plan, samples, tmpdir)
            files = writer.write_epoch(0)

            # Load first batch and verify content
            data = np.load(files[0], allow_pickle=True)

            assert "input_ids" in data
            assert "loss_mask" in data
            assert "sample_ids" in data
            assert "bucket_id" in data
            assert "max_len" in data
            assert "index" in data


class TestBatchReader:
    """Tests for BatchReader class."""

    @pytest.fixture
    def written_batches(self):
        """Create written batches for testing."""
        # Enough samples to form complete batches with drop_last=True
        lengths = {f"s{i}": 50 + (i * 10) for i in range(20)}
        samples = {
            sid: {"input_ids": list(range(length)), "loss_mask": [1] * length}
            for sid, length in lengths.items()
        }

        config = BatchingConfig.predictable(
            token_budget=512,
            bucket_edges=(128, 256),
            overflow_max=512,
            seed=42,
        )

        plan = asyncio.run(
            BatchPlanBuilder(
                lengths=lengths,
                batching_config=config,
                dataset_hash="test_v1",
                tokenizer_hash="test_tok_v1",
            ).build(num_epochs=2)
        )

        tmpdir = tempfile.mkdtemp()
        writer = BatchWriter(plan, samples, tmpdir)
        writer.write_all()

        return tmpdir, plan

    def test_read_epoch(self, written_batches):
        """Test reading batches from an epoch."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)

        batches = list(reader.iter_epoch(0))
        assert len(batches) > 0

        # Verify batch structure
        for batch in batches:
            assert "input_ids" in batch
            assert "loss_mask" in batch

    def test_num_epochs(self, written_batches):
        """Test num_epochs property."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)
        assert reader.num_epochs == plan.num_epochs

    def test_fingerprint(self, written_batches):
        """Test fingerprint property."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)
        assert reader.fingerprint == plan.fingerprint

    def test_verify_fingerprint(self, written_batches):
        """Test fingerprint verification."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)
        assert reader.verify_fingerprint(plan.fingerprint)
        assert not reader.verify_fingerprint("wrong_fingerprint")

    def test_get_batch(self, written_batches):
        """Test getting a specific batch."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)
        batch = reader.get_batch(epoch=0, index=0)

        assert "input_ids" in batch
        assert "loss_mask" in batch

    def test_iter_epoch_specs(self, written_batches):
        """Test iterating with MicrobatchSpec."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)

        for mb, batch in reader.iter_epoch_specs(0):
            # mb should have samples attribute
            assert hasattr(mb, "samples")
            # batch should have data
            assert "input_ids" in batch

    def test_missing_epoch_raises(self, written_batches):
        """Test that missing epoch raises FileNotFoundError."""
        batch_dir, plan = written_batches

        reader = BatchReader(batch_dir)

        with pytest.raises(FileNotFoundError):
            list(reader.iter_epoch(999))


class TestDefaultCollate:
    """Tests for default_collate function."""

    def test_basic_collate(self):
        """Test basic collation."""
        samples = [
            {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "loss_mask": [1, 1]},
        ]

        result = default_collate(samples, max_len=4, pad_id=0)

        assert result["input_ids"].shape == (2, 4)
        assert result["loss_mask"].shape == (2, 4)

        # Check padding
        assert list(result["input_ids"][0]) == [1, 2, 3, 0]
        assert list(result["input_ids"][1]) == [4, 5, 0, 0]

    def test_truncation(self):
        """Test truncation when sequence exceeds max_len."""
        samples = [{"input_ids": [1, 2, 3, 4, 5], "loss_mask": [1, 1, 1, 1, 1]}]

        result = default_collate(samples, max_len=3, pad_id=0)

        assert result["input_ids"].shape == (1, 3)
        assert list(result["input_ids"][0]) == [1, 2, 3]

    def test_missing_loss_mask(self):
        """Test default loss_mask generation."""
        samples = [{"input_ids": [1, 2, 3]}]

        result = default_collate(samples, max_len=4, pad_id=0)

        # Should generate all 1s for loss_mask
        assert list(result["loss_mask"][0]) == [1, 1, 1, 0]


class TestCollatedBatch:
    """Tests for CollatedBatch model."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        batch = CollatedBatch(
            input_ids=np.array([[1, 2, 3]]),
            loss_mask=np.array([[1, 1, 1]]),
            sample_ids=("s1",),
            bucket_id=0,
            max_len=3,
            index=0,
        )

        d = batch.to_dict()

        assert "input_ids" in d
        assert "loss_mask" in d
        assert d["sample_ids"] == ["s1"]
        assert d["bucket_id"] == 0

    def test_from_npz(self):
        """Test loading from NPZ data."""
        data = {
            "input_ids": np.array([[1, 2, 3]]),
            "loss_mask": np.array([[1, 1, 1]]),
            "sample_ids": np.array(["s1"]),
            "bucket_id": np.array(0),
            "max_len": np.array(3),
            "index": np.array(0),
        }

        batch = CollatedBatch.from_npz(data)

        assert batch.bucket_id == 0
        assert batch.max_len == 3
        assert batch.index == 0
        assert batch.sample_ids == ("s1",)
