"""
Type definitions for streaming infrastructure.

Provides Pydantic models for streaming samples, configuration, and metrics.
All enums for type safety - no magic strings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums - No magic strings
# =============================================================================


class SampleSource(str, Enum):
    """Source of a streaming sample."""

    OFFLINE = "offline"  # Static dataset file
    GYM = "gym"  # Live gym environment
    REPLAY = "replay"  # Replay buffer
    SYNTHETIC = "synthetic"  # Generated on-the-fly


class EpisodeStatus(str, Enum):
    """Status of a gym episode."""

    PENDING = "pending"  # Not started
    RUNNING = "running"  # In progress
    SUCCESS = "success"  # Completed successfully
    FAILURE = "failure"  # Failed
    TIMEOUT = "timeout"  # Timed out
    ERROR = "error"  # Encountered error


class StreamState(str, Enum):
    """State of a sample stream."""

    IDLE = "idle"  # Not started
    STREAMING = "streaming"  # Actively producing samples
    PAUSED = "paused"  # Temporarily paused
    EXHAUSTED = "exhausted"  # No more samples
    ERROR = "error"  # Encountered error


# =============================================================================
# Core Stream Types
# =============================================================================


class StreamSample(BaseModel):
    """
    A sample from a stream with provenance tracking.

    Extends the base Sample with streaming-specific metadata
    for online learning and curriculum tracking.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core sample data (compatible with Sample schema)
    input_ids: tuple[int, ...] = Field(description="Token IDs")
    loss_mask: tuple[int, ...] = Field(description="Loss mask (1=train, 0=ignore)")
    segment_ids: tuple[int, ...] | None = Field(default=None, description="Segment IDs for packed")

    # Identity
    sample_id: str = Field(description="Unique sample identifier")
    dataset_id: str = Field(description="Source dataset identifier")

    # Streaming-specific
    source: SampleSource = Field(description="Where sample came from")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When sample was produced",
    )

    # Episode tracking (for gym samples)
    episode_id: str | None = Field(default=None, description="Episode ID")
    episode_status: EpisodeStatus | None = Field(default=None, description="Episode status")
    step_index: int | None = Field(default=None, description="Step within episode")
    total_steps: int | None = Field(default=None, description="Total steps in episode")

    # Curriculum / RL
    reward: float | None = Field(default=None, description="Reward signal")
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0, description="Difficulty 0-1")
    success: bool | None = Field(default=None, description="Whether episode succeeded")

    # Replay tracking
    replay_count: int = Field(default=0, ge=0, description="Times replayed")
    priority: float = Field(default=1.0, ge=0.0, description="Sampling priority")

    @property
    def length(self) -> int:
        """Sequence length."""
        return len(self.input_ids)

    @property
    def is_gym_sample(self) -> bool:
        """Whether this sample is from a gym environment."""
        return self.source == SampleSource.GYM

    @property
    def is_replay(self) -> bool:
        """Whether this sample is a replay."""
        return self.source == SampleSource.REPLAY or self.replay_count > 0

    def with_replay(self) -> StreamSample:
        """Create a copy marked as replayed."""
        return StreamSample(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            segment_ids=self.segment_ids,
            sample_id=self.sample_id,
            dataset_id=self.dataset_id,
            source=SampleSource.REPLAY,
            timestamp=datetime.now(),
            episode_id=self.episode_id,
            episode_status=self.episode_status,
            step_index=self.step_index,
            total_steps=self.total_steps,
            reward=self.reward,
            difficulty=self.difficulty,
            success=self.success,
            replay_count=self.replay_count + 1,
            priority=self.priority,
        )


# =============================================================================
# Configuration Types
# =============================================================================


class StreamConfig(BaseModel):
    """Configuration for a sample stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Source configuration
    source_type: SampleSource = Field(description="Type of sample source")
    dataset_path: str | None = Field(default=None, description="Path for offline datasets")

    # Gym configuration
    env_name: str | None = Field(default=None, description="Gym environment name")
    host: str = Field(default="localhost", description="Gym server host")
    port: int = Field(default=8023, description="Gym server port")

    # Streaming behavior
    prefetch_size: int = Field(default=100, ge=1, description="Prefetch buffer size")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Connection timeout")

    # Filtering
    min_length: int = Field(default=1, ge=1, description="Minimum sequence length")
    max_length: int = Field(default=8192, ge=1, description="Maximum sequence length")
    filter_failures: bool = Field(default=False, description="Filter failed episodes")


# =============================================================================
# Metrics Types
# =============================================================================


class StreamMetrics(BaseModel):
    """Metrics for a sample stream."""

    model_config = ConfigDict(frozen=False)  # Mutable for accumulation

    # Counts
    samples_produced: int = Field(default=0, ge=0)
    samples_filtered: int = Field(default=0, ge=0)
    samples_errored: int = Field(default=0, ge=0)

    # Episode tracking (gym only)
    episodes_completed: int = Field(default=0, ge=0)
    episodes_successful: int = Field(default=0, ge=0)
    episodes_failed: int = Field(default=0, ge=0)

    # Timing
    total_time_seconds: float = Field(default=0.0, ge=0.0)
    samples_per_second: float = Field(default=0.0, ge=0.0)

    # Length statistics
    total_tokens: int = Field(default=0, ge=0)
    min_length_seen: int | None = Field(default=None)
    max_length_seen: int | None = Field(default=None)

    # Curriculum tracking
    difficulty_sum: float = Field(default=0.0, ge=0.0)
    reward_sum: float = Field(default=0.0)

    @property
    def samples_total(self) -> int:
        """Total samples seen (produced + filtered)."""
        return self.samples_produced + self.samples_filtered

    @property
    def filter_rate(self) -> float:
        """Fraction of samples filtered out."""
        if self.samples_total == 0:
            return 0.0
        return self.samples_filtered / self.samples_total

    @property
    def success_rate(self) -> float:
        """Episode success rate."""
        if self.episodes_completed == 0:
            return 0.0
        return self.episodes_successful / self.episodes_completed

    @property
    def mean_difficulty(self) -> float:
        """Mean difficulty of produced samples."""
        if self.samples_produced == 0:
            return 0.0
        return self.difficulty_sum / self.samples_produced

    @property
    def mean_reward(self) -> float:
        """Mean reward of produced samples."""
        if self.samples_produced == 0:
            return 0.0
        return self.reward_sum / self.samples_produced

    @property
    def mean_length(self) -> float:
        """Mean sequence length."""
        if self.samples_produced == 0:
            return 0.0
        return self.total_tokens / self.samples_produced

    def record_sample(self, sample: StreamSample) -> None:
        """Record metrics for a produced sample."""
        self.samples_produced += 1
        self.total_tokens += sample.length

        if self.min_length_seen is None or sample.length < self.min_length_seen:
            self.min_length_seen = sample.length
        if self.max_length_seen is None or sample.length > self.max_length_seen:
            self.max_length_seen = sample.length

        if sample.difficulty is not None:
            self.difficulty_sum += sample.difficulty
        if sample.reward is not None:
            self.reward_sum += sample.reward

    def record_filtered(self) -> None:
        """Record a filtered sample."""
        self.samples_filtered += 1

    def record_episode(self, success: bool) -> None:
        """Record episode completion."""
        self.episodes_completed += 1
        if success:
            self.episodes_successful += 1
        else:
            self.episodes_failed += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "samples_produced": self.samples_produced,
            "samples_filtered": self.samples_filtered,
            "filter_rate": self.filter_rate,
            "samples_per_second": self.samples_per_second,
            "mean_length": self.mean_length,
            "mean_difficulty": self.mean_difficulty,
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "episodes_completed": self.episodes_completed,
        }
