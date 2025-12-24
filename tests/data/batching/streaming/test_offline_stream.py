"""Tests for offline dataset stream."""

import json

import pytest

from chuk_lazarus.data.batching.streaming import (
    OfflineDatasetStream,
    SampleSource,
    StreamState,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample JSONL dataset file."""
    data = [
        {
            "input_ids": list(range(50)),
            "loss_mask": [0] * 20 + [1] * 30,
            "sample_id": "sample_0",
        },
        {
            "input_ids": list(range(100)),
            "loss_mask": [0] * 40 + [1] * 60,
            "sample_id": "sample_1",
        },
        {
            "input_ids": list(range(200)),
            "loss_mask": [0] * 80 + [1] * 120,
            "sample_id": "sample_2",
        },
    ]

    path = tmp_path / "dataset.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


@pytest.fixture
def dataset_with_meta(tmp_path):
    """Create a dataset with full metadata."""
    data = [
        {
            "input_ids": list(range(100)),
            "loss_mask": [0] * 50 + [1] * 50,
            "meta": {
                "sample_id": "meta_sample_0",
                "episode_id": "ep_001",
                "reward": 1.0,
                "difficulty": 0.5,
                "success": True,
            },
        },
        {
            "input_ids": list(range(150)),
            "loss_mask": [0] * 75 + [1] * 75,
            "meta": {
                "sample_id": "meta_sample_1",
                "episode_id": "ep_002",
                "reward": 0.0,
                "difficulty": 0.8,
                "success": False,
            },
        },
    ]

    path = tmp_path / "meta_dataset.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return path


class TestOfflineDatasetStream:
    """Tests for OfflineDatasetStream."""

    def test_basic_iteration(self, sample_dataset):
        """Test basic iteration over dataset."""
        stream = OfflineDatasetStream(path=sample_dataset)

        samples = list(stream)

        assert len(samples) == 3
        assert samples[0].sample_id == "sample_0"
        assert samples[0].length == 50
        assert samples[1].length == 100
        assert samples[2].length == 200

    def test_source_is_offline(self, sample_dataset):
        """Test that samples are marked as offline source."""
        stream = OfflineDatasetStream(path=sample_dataset)

        for sample in stream:
            assert sample.source == SampleSource.OFFLINE
            assert not sample.is_gym_sample

    def test_length_filtering(self, sample_dataset):
        """Test filtering by length."""
        stream = OfflineDatasetStream(
            path=sample_dataset,
            min_length=60,
            max_length=150,
        )

        samples = list(stream)

        assert len(samples) == 1
        assert samples[0].length == 100

    def test_state_transitions(self, sample_dataset):
        """Test stream state transitions."""
        stream = OfflineDatasetStream(path=sample_dataset)

        assert stream.state == StreamState.IDLE

        # Start iteration
        iterator = iter(stream)
        next(iterator)
        assert stream.state == StreamState.STREAMING

        # Exhaust stream
        list(iterator)
        assert stream.state == StreamState.EXHAUSTED

    def test_reset(self, sample_dataset):
        """Test resetting stream."""
        stream = OfflineDatasetStream(path=sample_dataset)

        # First pass
        samples1 = list(stream)
        assert stream.state == StreamState.EXHAUSTED

        # Reset
        stream.reset()
        assert stream.state == StreamState.IDLE

        # Second pass
        samples2 = list(stream)
        assert len(samples2) == len(samples1)

    def test_metrics(self, sample_dataset):
        """Test metrics accumulation."""
        stream = OfflineDatasetStream(path=sample_dataset)

        _ = list(stream)

        metrics = stream.metrics
        assert metrics.samples_produced == 3
        assert metrics.total_tokens == 350  # 50 + 100 + 200
        assert metrics.min_length_seen == 50
        assert metrics.max_length_seen == 200

    def test_dataset_id(self, sample_dataset):
        """Test dataset ID assignment."""
        # Default: use filename
        stream = OfflineDatasetStream(path=sample_dataset)
        for sample in stream:
            assert sample.dataset_id == "dataset"

        # Custom dataset ID
        stream = OfflineDatasetStream(path=sample_dataset, dataset_id="my_data")
        stream.reset()
        for sample in stream:
            assert sample.dataset_id == "my_data"

    def test_with_metadata(self, dataset_with_meta):
        """Test loading samples with full metadata."""
        stream = OfflineDatasetStream(path=dataset_with_meta)

        samples = list(stream)

        assert len(samples) == 2
        assert samples[0].sample_id == "meta_sample_0"
        assert samples[0].episode_id == "ep_001"
        assert samples[0].reward == 1.0
        assert samples[0].difficulty == 0.5
        assert samples[0].success is True

        assert samples[1].success is False

    def test_count_samples(self, sample_dataset):
        """Test counting samples without loading."""
        stream = OfflineDatasetStream(path=sample_dataset)

        count = stream.count_samples()

        assert count == 3

    def test_peek(self, sample_dataset):
        """Test peeking at samples."""
        stream = OfflineDatasetStream(path=sample_dataset)

        peeked = stream.peek(n=2)

        assert len(peeked) == 2
        # Stream should still work after peek
        samples = list(stream)
        assert len(samples) == 3

    def test_get_length_distribution(self, sample_dataset):
        """Test getting length distribution."""
        stream = OfflineDatasetStream(path=sample_dataset)

        lengths = stream.get_length_distribution()

        assert len(lengths) == 3
        assert lengths["sample_0"] == 50
        assert lengths["sample_1"] == 100
        assert lengths["sample_2"] == 200

    def test_shuffle(self, sample_dataset):
        """Test shuffled iteration."""
        stream = OfflineDatasetStream(path=sample_dataset, shuffle=True, seed=42)

        samples = list(stream)

        assert len(samples) == 3
        # With shuffle, order might be different
        sample_ids = [s.sample_id for s in samples]
        assert set(sample_ids) == {"sample_0", "sample_1", "sample_2"}

    def test_empty_lines_skipped(self, tmp_path):
        """Test that empty lines are skipped."""
        path = tmp_path / "with_empty.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace line
            f.write(json.dumps({"input_ids": [4, 5, 6], "loss_mask": [1, 1, 1]}) + "\n")

        stream = OfflineDatasetStream(path=path)
        samples = list(stream)

        assert len(samples) == 2


class TestAsyncOfflineDatasetStream:
    """Tests for async offline stream."""

    @pytest.mark.asyncio
    async def test_async_iteration(self, sample_dataset):
        """Test async iteration."""
        from chuk_lazarus.data.batching.streaming import AsyncOfflineDatasetStream

        async with AsyncOfflineDatasetStream(path=sample_dataset) as stream:
            samples = []
            async for sample in stream:
                samples.append(sample)

        assert len(samples) == 3
        assert samples[0].source == SampleSource.OFFLINE
