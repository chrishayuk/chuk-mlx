"""Tests for streaming types module."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching.streaming import (
    EpisodeStatus,
    SampleSource,
    StreamMetrics,
    StreamSample,
    StreamState,
)


class TestStreamSample:
    """Tests for StreamSample model."""

    def test_create_basic(self):
        """Test creating a basic stream sample."""
        sample = StreamSample(
            input_ids=(1, 2, 3, 4, 5),
            loss_mask=(0, 0, 1, 1, 1),
            sample_id="test_001",
            dataset_id="my_dataset",
            source=SampleSource.OFFLINE,
        )

        assert sample.length == 5
        assert sample.sample_id == "test_001"
        assert sample.source == SampleSource.OFFLINE
        assert not sample.is_gym_sample
        assert not sample.is_replay

    def test_create_gym_sample(self):
        """Test creating a gym sample."""
        sample = StreamSample(
            input_ids=tuple(range(100)),
            loss_mask=tuple([0] * 50 + [1] * 50),
            sample_id="gym_001",
            dataset_id="puzzle_gym",
            source=SampleSource.GYM,
            episode_id="ep_001",
            episode_status=EpisodeStatus.SUCCESS,
            step_index=2,
            total_steps=5,
            reward=1.0,
            difficulty=0.5,
            success=True,
        )

        assert sample.is_gym_sample
        assert not sample.is_replay
        assert sample.episode_id == "ep_001"
        assert sample.success is True

    def test_with_replay(self):
        """Test creating a replay copy."""
        original = StreamSample(
            input_ids=(1, 2, 3),
            loss_mask=(1, 1, 1),
            sample_id="test_001",
            dataset_id="test",
            source=SampleSource.OFFLINE,
            replay_count=0,
        )

        replay = original.with_replay()

        assert replay.source == SampleSource.REPLAY
        assert replay.replay_count == 1
        assert replay.is_replay
        assert replay.sample_id == original.sample_id
        assert replay.input_ids == original.input_ids

    def test_immutable(self):
        """Test that StreamSample is immutable."""
        sample = StreamSample(
            input_ids=(1, 2, 3),
            loss_mask=(1, 1, 1),
            sample_id="test",
            dataset_id="test",
            source=SampleSource.OFFLINE,
        )

        with pytest.raises(ValidationError):
            sample.sample_id = "new_id"

    def test_difficulty_validation(self):
        """Test difficulty field validation."""
        # Valid difficulty
        sample = StreamSample(
            input_ids=(1, 2, 3),
            loss_mask=(1, 1, 1),
            sample_id="test",
            dataset_id="test",
            source=SampleSource.OFFLINE,
            difficulty=0.5,
        )
        assert sample.difficulty == 0.5

        # Invalid difficulty (out of range)
        with pytest.raises(ValidationError):
            StreamSample(
                input_ids=(1, 2, 3),
                loss_mask=(1, 1, 1),
                sample_id="test",
                dataset_id="test",
                source=SampleSource.OFFLINE,
                difficulty=1.5,  # > 1.0
            )


class TestStreamMetrics:
    """Tests for StreamMetrics model."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = StreamMetrics()

        assert metrics.samples_produced == 0
        assert metrics.samples_filtered == 0
        assert metrics.filter_rate == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.mean_length == 0.0

    def test_record_sample(self):
        """Test recording samples."""
        metrics = StreamMetrics()

        sample = StreamSample(
            input_ids=tuple(range(100)),
            loss_mask=tuple([1] * 100),
            sample_id="test_001",
            dataset_id="test",
            source=SampleSource.OFFLINE,
            difficulty=0.5,
            reward=0.8,
        )

        metrics.record_sample(sample)

        assert metrics.samples_produced == 1
        assert metrics.total_tokens == 100
        assert metrics.min_length_seen == 100
        assert metrics.max_length_seen == 100
        assert metrics.mean_length == 100.0
        assert metrics.mean_difficulty == 0.5
        assert metrics.mean_reward == 0.8

    def test_record_multiple_samples(self):
        """Test recording multiple samples."""
        metrics = StreamMetrics()

        for i, length in enumerate([50, 100, 150]):
            sample = StreamSample(
                input_ids=tuple(range(length)),
                loss_mask=tuple([1] * length),
                sample_id=f"test_{i}",
                dataset_id="test",
                source=SampleSource.OFFLINE,
            )
            metrics.record_sample(sample)

        assert metrics.samples_produced == 3
        assert metrics.total_tokens == 300
        assert metrics.min_length_seen == 50
        assert metrics.max_length_seen == 150
        assert metrics.mean_length == 100.0

    def test_record_filtered(self):
        """Test recording filtered samples."""
        metrics = StreamMetrics()

        metrics.record_sample(
            StreamSample(
                input_ids=(1, 2, 3),
                loss_mask=(1, 1, 1),
                sample_id="kept",
                dataset_id="test",
                source=SampleSource.OFFLINE,
            )
        )
        metrics.record_filtered()
        metrics.record_filtered()

        assert metrics.samples_produced == 1
        assert metrics.samples_filtered == 2
        assert metrics.samples_total == 3
        assert metrics.filter_rate == 2 / 3

    def test_record_episode(self):
        """Test recording episodes."""
        metrics = StreamMetrics()

        metrics.record_episode(success=True)
        metrics.record_episode(success=True)
        metrics.record_episode(success=False)

        assert metrics.episodes_completed == 3
        assert metrics.episodes_successful == 2
        assert metrics.episodes_failed == 1
        assert metrics.success_rate == 2 / 3

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = StreamMetrics()
        metrics.record_sample(
            StreamSample(
                input_ids=tuple(range(100)),
                loss_mask=tuple([1] * 100),
                sample_id="test",
                dataset_id="test",
                source=SampleSource.OFFLINE,
            )
        )

        d = metrics.to_dict()

        assert "samples_produced" in d
        assert "mean_length" in d
        assert d["samples_produced"] == 1


class TestEnums:
    """Tests for streaming enums."""

    def test_sample_source_values(self):
        """Test SampleSource enum values."""
        assert SampleSource.OFFLINE.value == "offline"
        assert SampleSource.GYM.value == "gym"
        assert SampleSource.REPLAY.value == "replay"
        assert SampleSource.SYNTHETIC.value == "synthetic"

    def test_episode_status_values(self):
        """Test EpisodeStatus enum values."""
        assert EpisodeStatus.SUCCESS.value == "success"
        assert EpisodeStatus.FAILURE.value == "failure"
        assert EpisodeStatus.TIMEOUT.value == "timeout"

    def test_stream_state_values(self):
        """Test StreamState enum values."""
        assert StreamState.IDLE.value == "idle"
        assert StreamState.STREAMING.value == "streaming"
        assert StreamState.EXHAUSTED.value == "exhausted"
