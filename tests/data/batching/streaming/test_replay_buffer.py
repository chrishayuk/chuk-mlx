"""Tests for replay buffer module."""

from chuk_lazarus.data.batching.streaming import (
    BufferEvictionPolicy,
    ReplayBuffer,
    ReplayBufferConfig,
    SampleSource,
    StreamSample,
)


def make_sample(sample_id: str, length: int = 100, **kwargs) -> StreamSample:
    """Helper to create test samples."""
    return StreamSample(
        input_ids=tuple(range(length)),
        loss_mask=tuple([1] * length),
        sample_id=sample_id,
        dataset_id="test",
        source=SampleSource.OFFLINE,
        **kwargs,
    )


class TestReplayBufferConfig:
    """Tests for ReplayBufferConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = ReplayBufferConfig()

        assert config.max_size == 100_000
        assert config.eviction_policy == BufferEvictionPolicy.FIFO
        assert config.priority_alpha == 0.6
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReplayBufferConfig(
            max_size=1000,
            eviction_policy=BufferEvictionPolicy.PRIORITY,
            priority_alpha=0.8,
        )

        assert config.max_size == 1000
        assert config.eviction_policy == BufferEvictionPolicy.PRIORITY
        assert config.priority_alpha == 0.8


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_empty_buffer(self):
        """Test empty buffer properties."""
        buffer = ReplayBuffer()

        assert buffer.size == 0
        assert buffer.is_empty
        assert not buffer.is_full
        assert not buffer.can_sample
        assert buffer.total_added == 0
        assert buffer.total_evicted == 0

    def test_add_sample(self):
        """Test adding a single sample."""
        buffer = ReplayBuffer()
        sample = make_sample("test_001")

        buffer.add(sample)

        assert buffer.size == 1
        assert not buffer.is_empty
        assert buffer.total_added == 1

    def test_add_batch(self):
        """Test adding multiple samples at once."""
        buffer = ReplayBuffer()
        samples = [make_sample(f"test_{i}") for i in range(10)]

        buffer.add_batch(samples)

        assert buffer.size == 10
        assert buffer.total_added == 10

    def test_sample_uniform(self):
        """Test uniform sampling."""
        buffer = ReplayBuffer(ReplayBufferConfig(seed=42))
        samples = [make_sample(f"test_{i}") for i in range(100)]
        buffer.add_batch(samples)

        sampled = buffer.sample_uniform(10)

        assert len(sampled) == 10
        for s in sampled:
            assert s.is_replay
            assert s.replay_count == 1

    def test_sample_with_priority(self):
        """Test priority-weighted sampling."""
        config = ReplayBufferConfig(priority_alpha=1.0, seed=42)
        buffer = ReplayBuffer(config)

        # Add low priority samples
        for i in range(10):
            buffer.add(make_sample(f"low_{i}"), priority=0.1)

        # Add high priority samples
        for i in range(10):
            buffer.add(make_sample(f"high_{i}"), priority=10.0)

        # Sample many times and count
        high_count = 0
        for _ in range(100):
            sampled = buffer.sample(1)
            if sampled[0].sample_id.startswith("high_"):
                high_count += 1

        # High priority should be sampled more often
        assert high_count > 50  # Should be significantly more than 50%

    def test_get_all(self):
        """Test getting all samples."""
        buffer = ReplayBuffer()
        samples = [make_sample(f"test_{i}") for i in range(5)]
        buffer.add_batch(samples)

        all_samples = buffer.get_all()

        assert len(all_samples) == 5
        for s in all_samples:
            assert not s.is_replay  # Should not be marked as replays

    def test_snapshot(self):
        """Test creating a snapshot."""
        buffer = ReplayBuffer()
        samples = [make_sample(f"test_{i}") for i in range(10)]
        buffer.add_batch(samples)

        snapshot = buffer.snapshot()

        assert snapshot.size == 10
        assert snapshot.total_added == 10
        assert snapshot.total_evicted == 0
        assert len(snapshot.get_lengths()) == 10

    def test_fifo_eviction(self):
        """Test FIFO eviction policy."""
        config = ReplayBufferConfig(
            max_size=5,
            eviction_policy=BufferEvictionPolicy.FIFO,
        )
        buffer = ReplayBuffer(config)

        # Add more samples than max_size
        for i in range(10):
            buffer.add(make_sample(f"test_{i}"))

        assert buffer.size == 5
        assert buffer.total_evicted == 5

        # Oldest samples should be evicted
        remaining_ids = {s.sample_id for s in buffer.get_all()}
        assert "test_0" not in remaining_ids
        assert "test_9" in remaining_ids

    def test_priority_eviction(self):
        """Test priority-based eviction."""
        config = ReplayBufferConfig(
            max_size=5,
            eviction_policy=BufferEvictionPolicy.PRIORITY,
        )
        buffer = ReplayBuffer(config)

        # Add samples with varying priorities
        buffer.add(make_sample("low_1"), priority=0.1)
        buffer.add(make_sample("high_1"), priority=10.0)
        buffer.add(make_sample("low_2"), priority=0.1)
        buffer.add(make_sample("high_2"), priority=10.0)
        buffer.add(make_sample("low_3"), priority=0.1)

        # Add one more to trigger eviction
        buffer.add(make_sample("high_3"), priority=10.0)

        assert buffer.size == 5
        remaining_ids = {s.sample_id for s in buffer.get_all()}

        # All high priority should remain, some low evicted
        assert "high_1" in remaining_ids
        assert "high_2" in remaining_ids
        assert "high_3" in remaining_ids

    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        buffer = ReplayBuffer()
        buffer.add(make_sample("easy", difficulty=0.2))
        buffer.add(make_sample("medium", difficulty=0.5))
        buffer.add(make_sample("hard", difficulty=0.8))

        easy_samples = buffer.filter_by_difficulty(0.0, 0.3)
        assert len(easy_samples) == 1
        assert easy_samples[0].sample_id == "easy"

        hard_samples = buffer.filter_by_difficulty(0.7, 1.0)
        assert len(hard_samples) == 1
        assert hard_samples[0].sample_id == "hard"

    def test_filter_by_success(self):
        """Test filtering by success status."""
        buffer = ReplayBuffer()
        buffer.add(make_sample("success_1", success=True))
        buffer.add(make_sample("success_2", success=True))
        buffer.add(make_sample("failure_1", success=False))

        successes = buffer.filter_by_success(True)
        failures = buffer.filter_by_success(False)

        assert len(successes) == 2
        assert len(failures) == 1

    def test_update_priority(self):
        """Test updating sample priority."""
        buffer = ReplayBuffer()
        buffer.add(make_sample("test_001"), priority=1.0)

        assert buffer.update_priority("test_001", 5.0)
        assert not buffer.update_priority("nonexistent", 5.0)

    def test_clear(self):
        """Test clearing buffer."""
        buffer = ReplayBuffer()
        buffer.add_batch([make_sample(f"test_{i}") for i in range(10)])

        buffer.clear()

        assert buffer.size == 0
        assert buffer.is_empty
        assert buffer.total_evicted == 10

    def test_statistics_tracking(self):
        """Test difficulty and reward statistics."""
        config = ReplayBufferConfig(track_difficulty=True, track_success=True)
        buffer = ReplayBuffer(config)

        buffer.add(make_sample("s1", difficulty=0.2, reward=0.5, success=True))
        buffer.add(make_sample("s2", difficulty=0.4, reward=1.0, success=True))
        buffer.add(make_sample("s3", difficulty=0.6, reward=0.0, success=False))

        assert 0.3 < buffer.mean_difficulty < 0.5  # ~0.4
        assert 0.4 < buffer.mean_reward < 0.6  # ~0.5
        assert 0.6 < buffer.success_rate < 0.7  # ~0.67

    def test_serialization(self):
        """Test buffer serialization and deserialization."""
        buffer = ReplayBuffer(ReplayBufferConfig(max_size=100))
        samples = [make_sample(f"test_{i}", difficulty=0.5) for i in range(10)]
        buffer.add_batch(samples)

        # Serialize
        data = buffer.to_dict()

        # Deserialize
        restored = ReplayBuffer.from_dict(data)

        assert restored.size == buffer.size
        assert restored.total_added == buffer.total_added
        assert restored.config.max_size == buffer.config.max_size
