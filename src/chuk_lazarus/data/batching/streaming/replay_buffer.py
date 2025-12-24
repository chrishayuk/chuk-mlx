"""
Replay buffer for online learning.

Provides a bounded, priority-aware buffer for storing and sampling
training samples. Supports multiple eviction policies and efficient
random sampling.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .types import SampleSource, StreamSample

# =============================================================================
# Enums
# =============================================================================


class BufferEvictionPolicy(str, Enum):
    """Policy for evicting samples when buffer is full."""

    FIFO = "fifo"  # First in, first out (oldest samples)
    PRIORITY = "priority"  # Lowest priority first
    RANDOM = "random"  # Random eviction


# =============================================================================
# Configuration
# =============================================================================


class ReplayBufferConfig(BaseModel):
    """Configuration for replay buffer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_size: int = Field(default=100_000, ge=1, description="Maximum buffer size")
    eviction_policy: BufferEvictionPolicy = Field(
        default=BufferEvictionPolicy.FIFO,
        description="Eviction policy when full",
    )
    priority_alpha: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Priority exponent (0=uniform, 1=full priority)",
    )
    seed: int = Field(default=42, description="Random seed for sampling")

    # Sampling behavior
    min_samples_before_sampling: int = Field(
        default=1,
        ge=1,
        description="Minimum samples before sampling is allowed",
    )
    default_priority: float = Field(
        default=1.0,
        ge=0.0,
        description="Default priority for new samples",
    )

    # Curriculum support
    track_difficulty: bool = Field(default=True, description="Track difficulty distribution")
    track_success: bool = Field(default=True, description="Track success rate")


# =============================================================================
# Buffer Snapshot
# =============================================================================


class BufferSnapshot(BaseModel):
    """
    Immutable snapshot of replay buffer state.

    Used for building batch plans from buffer contents.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: tuple[StreamSample, ...] = Field(description="Buffer samples")
    timestamp: datetime = Field(default_factory=datetime.now, description="Snapshot time")

    # Statistics
    total_added: int = Field(description="Total samples ever added")
    total_evicted: int = Field(description="Total samples evicted")

    @property
    def size(self) -> int:
        """Number of samples in snapshot."""
        return len(self.samples)

    def get_lengths(self) -> dict[str, int]:
        """Get length map for batch planning."""
        return {s.sample_id: s.length for s in self.samples}


# =============================================================================
# Replay Buffer
# =============================================================================


class ReplayBuffer:
    """
    Bounded replay buffer for online learning.

    Stores samples with priority for experience replay. Supports
    multiple eviction policies and priority-aware sampling.

    Example:
        buffer = ReplayBuffer(ReplayBufferConfig(max_size=10000))

        # Add samples from gym
        async for sample in gym_stream:
            buffer.add(sample)

        # Sample for training
        batch = buffer.sample(n=32)

        # Snapshot for batch planning
        snapshot = buffer.snapshot()
        plan = await BatchPlanBuilder(lengths=snapshot.get_lengths(), ...).build()
    """

    def __init__(self, config: ReplayBufferConfig | None = None):
        """Initialize replay buffer."""
        self.config = config or ReplayBufferConfig()
        self._rng = random.Random(self.config.seed)

        # Storage
        self._buffer: deque[StreamSample] = deque(maxlen=self.config.max_size)
        self._priorities: deque[float] = deque(maxlen=self.config.max_size)

        # Tracking
        self._total_added: int = 0
        self._total_evicted: int = 0

        # Statistics
        self._difficulty_sum: float = 0.0
        self._reward_sum: float = 0.0
        self._success_count: int = 0
        self._failure_count: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Whether buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return len(self._buffer) >= self.config.max_size

    @property
    def can_sample(self) -> bool:
        """Whether buffer has enough samples for sampling."""
        return len(self._buffer) >= self.config.min_samples_before_sampling

    @property
    def total_added(self) -> int:
        """Total samples ever added."""
        return self._total_added

    @property
    def total_evicted(self) -> int:
        """Total samples evicted."""
        return self._total_evicted

    @property
    def mean_difficulty(self) -> float:
        """Mean difficulty of samples in buffer."""
        if self.size == 0:
            return 0.0
        return self._difficulty_sum / self.size

    @property
    def mean_reward(self) -> float:
        """Mean reward of samples in buffer."""
        if self.size == 0:
            return 0.0
        return self._reward_sum / self.size

    @property
    def success_rate(self) -> float:
        """Success rate of samples with success/failure tracking."""
        total = self._success_count + self._failure_count
        if total == 0:
            return 0.0
        return self._success_count / total

    # =========================================================================
    # Core Operations
    # =========================================================================

    def add(
        self,
        sample: StreamSample,
        priority: float | None = None,
    ) -> None:
        """
        Add a sample to the buffer.

        Args:
            sample: Sample to add
            priority: Optional priority override (uses sample.priority if None)
        """
        # Handle eviction if full
        if self.is_full:
            self._evict_one()

        # Add sample
        self._buffer.append(sample)
        self._priorities.append(priority if priority is not None else sample.priority)
        self._total_added += 1

        # Update statistics
        if sample.difficulty is not None and self.config.track_difficulty:
            self._difficulty_sum += sample.difficulty
        if sample.reward is not None:
            self._reward_sum += sample.reward
        if sample.success is not None and self.config.track_success:
            if sample.success:
                self._success_count += 1
            else:
                self._failure_count += 1

    def add_batch(
        self,
        samples: list[StreamSample],
        priorities: list[float] | None = None,
    ) -> None:
        """Add multiple samples at once."""
        if priorities is None:
            priorities = [s.priority for s in samples]

        for sample, priority in zip(samples, priorities):
            self.add(sample, priority)

    def sample(self, n: int) -> list[StreamSample]:
        """
        Sample n samples from buffer.

        Uses priority-weighted sampling if priority_alpha > 0.

        Args:
            n: Number of samples to draw (with replacement if n > size)

        Returns:
            List of sampled StreamSamples (marked as replays)
        """
        if self.is_empty:
            return []

        n = min(n, self.size) if n <= self.size else n

        if self.config.priority_alpha == 0.0:
            # Uniform sampling
            indices = [self._rng.randrange(self.size) for _ in range(n)]
        else:
            # Priority-weighted sampling
            weights = [p**self.config.priority_alpha for p in self._priorities]
            indices = self._rng.choices(range(self.size), weights=weights, k=n)

        return [self._buffer[i].with_replay() for i in indices]

    def sample_uniform(self, n: int) -> list[StreamSample]:
        """Sample n samples uniformly (ignoring priorities)."""
        if self.is_empty:
            return []

        n = min(n, self.size)
        indices = self._rng.sample(range(self.size), n)
        return [self._buffer[i].with_replay() for i in indices]

    def get_all(self) -> list[StreamSample]:
        """Get all samples in buffer (not marked as replays)."""
        return list(self._buffer)

    def snapshot(self) -> BufferSnapshot:
        """
        Create an immutable snapshot of buffer state.

        Used for building batch plans from current buffer contents.
        """
        return BufferSnapshot(
            samples=tuple(self._buffer),
            total_added=self._total_added,
            total_evicted=self._total_evicted,
        )

    # =========================================================================
    # Filtering and Queries
    # =========================================================================

    def filter(
        self,
        predicate: Callable[[StreamSample], bool],
    ) -> list[StreamSample]:
        """Return samples matching predicate."""
        return [s for s in self._buffer if predicate(s)]

    def filter_by_difficulty(
        self,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
    ) -> list[StreamSample]:
        """Filter samples by difficulty range."""
        return [
            s
            for s in self._buffer
            if s.difficulty is not None and min_difficulty <= s.difficulty <= max_difficulty
        ]

    def filter_by_success(self, success: bool) -> list[StreamSample]:
        """Filter samples by success status."""
        return [s for s in self._buffer if s.success == success]

    def filter_by_source(self, source: SampleSource) -> list[StreamSample]:
        """Filter samples by source."""
        return [s for s in self._buffer if s.source == source]

    # =========================================================================
    # Priority Management
    # =========================================================================

    def update_priority(self, sample_id: str, priority: float) -> bool:
        """
        Update priority for a sample by ID.

        Returns True if sample was found and updated.
        """
        for i, sample in enumerate(self._buffer):
            if sample.sample_id == sample_id:
                self._priorities[i] = priority
                return True
        return False

    def update_priorities_by_reward(self, alpha: float = 0.1) -> None:
        """
        Update priorities based on rewards.

        Higher reward = higher priority. Uses exponential smoothing.
        """
        for i, sample in enumerate(self._buffer):
            if sample.reward is not None:
                # Normalize reward to [0, 2] range and use as priority
                normalized = max(0.0, sample.reward + 1.0)  # Assume reward in [-1, 1]
                old = self._priorities[i]
                self._priorities[i] = (1 - alpha) * old + alpha * normalized

    # =========================================================================
    # Buffer Management
    # =========================================================================

    def clear(self) -> None:
        """Clear all samples from buffer."""
        evicted = len(self._buffer)
        self._buffer.clear()
        self._priorities.clear()
        self._total_evicted += evicted
        self._difficulty_sum = 0.0
        self._reward_sum = 0.0
        self._success_count = 0
        self._failure_count = 0

    def _evict_one(self) -> None:
        """Evict one sample according to policy."""
        if self.is_empty:
            return

        if self.config.eviction_policy == BufferEvictionPolicy.FIFO:
            # Evict oldest (front of deque)
            evicted = self._buffer.popleft()
            self._priorities.popleft()
        elif self.config.eviction_policy == BufferEvictionPolicy.PRIORITY:
            # Evict lowest priority
            min_idx = min(range(len(self._priorities)), key=lambda i: self._priorities[i])
            evicted = self._buffer[min_idx]
            del self._buffer[min_idx]
            del self._priorities[min_idx]
        else:  # RANDOM
            idx = self._rng.randrange(len(self._buffer))
            evicted = self._buffer[idx]
            del self._buffer[idx]
            del self._priorities[idx]

        self._total_evicted += 1

        # Update statistics
        if evicted.difficulty is not None:
            self._difficulty_sum -= evicted.difficulty
        if evicted.reward is not None:
            self._reward_sum -= evicted.reward
        if evicted.success is not None:
            if evicted.success:
                self._success_count -= 1
            else:
                self._failure_count -= 1

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize buffer state."""
        return {
            "config": self.config.model_dump(),
            "samples": [s.model_dump() for s in self._buffer],
            "priorities": list(self._priorities),
            "total_added": self._total_added,
            "total_evicted": self._total_evicted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReplayBuffer:
        """Deserialize buffer state."""
        config = ReplayBufferConfig(**data["config"])
        buffer = cls(config)

        for sample_data, priority in zip(data["samples"], data["priorities"]):
            sample = StreamSample(**sample_data)
            buffer.add(sample, priority)

        buffer._total_added = data["total_added"]
        buffer._total_evicted = data["total_evicted"]

        return buffer
