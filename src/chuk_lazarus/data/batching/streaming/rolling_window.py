"""
Rolling batch plan window for online learning.

Builds batch plans over rolling buffer snapshots, enabling
continuous training with streaming data.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..planning import BatchingConfig, BatchPlan, BatchPlanBuilder
from .replay_buffer import BufferSnapshot, ReplayBuffer

# =============================================================================
# Configuration
# =============================================================================


class WindowConfig(BaseModel):
    """Configuration for rolling batch plan window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Window sizing
    window_microbatches: int = Field(
        default=1000,
        ge=1,
        description="Target microbatches per window",
    )
    min_samples: int = Field(
        default=100,
        ge=1,
        description="Minimum samples before building window",
    )

    # Overlap
    overlap_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of samples to overlap between windows",
    )

    # Batching config passthrough
    token_budget: int = Field(default=4096, ge=1)
    bucket_edges: tuple[int, ...] = Field(default=(128, 256, 512, 1024))
    overflow_max: int = Field(default=2048, ge=1)
    seed: int = Field(default=42)

    def to_batching_config(self, window_seed: int) -> BatchingConfig:
        """Convert to BatchingConfig for plan building."""
        return BatchingConfig.throughput(
            token_budget=self.token_budget,
            bucket_edges=self.bucket_edges,
            overflow_max=self.overflow_max,
            seed=window_seed,
        )


# =============================================================================
# Window State
# =============================================================================


class WindowState(BaseModel):
    """State of a rolling window."""

    model_config = ConfigDict(frozen=False)  # Mutable for progress tracking

    window_id: int = Field(description="Window sequence number")
    created_at: datetime = Field(default_factory=datetime.now)

    # Snapshot info
    snapshot_size: int = Field(description="Samples in snapshot")
    snapshot_total_added: int = Field(description="Total added to buffer at snapshot time")

    # Plan info
    plan_microbatches: int = Field(description="Microbatches in plan")
    plan_fingerprint: str = Field(description="Plan fingerprint for verification")

    # Progress (mutable)
    microbatches_consumed: int = Field(default=0, description="Microbatches consumed from plan")

    @property
    def is_exhausted(self) -> bool:
        """Whether all microbatches have been consumed."""
        return self.microbatches_consumed >= self.plan_microbatches

    @property
    def progress(self) -> float:
        """Fraction of window consumed."""
        if self.plan_microbatches == 0:
            return 1.0
        return self.microbatches_consumed / self.plan_microbatches


# =============================================================================
# Rolling Window
# =============================================================================


class RollingBatchPlanWindow:
    """
    Build batch plans over rolling buffer snapshots.

    Enables continuous training with streaming data by:
    1. Snapshotting replay buffer state
    2. Building a batch plan from the snapshot
    3. Iterating through the plan for training
    4. Building next window when current exhausted

    Example:
        buffer = ReplayBuffer(ReplayBufferConfig(max_size=100000))
        window = RollingBatchPlanWindow(buffer, WindowConfig(window_microbatches=500))

        # Collect samples
        async for sample in gym_stream:
            buffer.add(sample)

            # Train when window available
            if window.has_window():
                for mb in window.iter_current():
                    train_step(mb)

            # Build new window when needed
            if window.should_build_next():
                await window.build_next_window()
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        config: WindowConfig | None = None,
    ):
        """
        Initialize rolling window.

        Args:
            buffer: Replay buffer to snapshot
            config: Window configuration
        """
        self.buffer = buffer
        self.config = config or WindowConfig()

        # Current window state
        self._window_id: int = 0
        self._current_snapshot: BufferSnapshot | None = None
        self._current_plan: BatchPlan | None = None
        self._current_state: WindowState | None = None

        # Tracking
        self._total_windows: int = 0
        self._total_microbatches: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def has_window(self) -> bool:
        """Whether a window is currently available."""
        return self._current_plan is not None

    @property
    def current_state(self) -> WindowState | None:
        """Current window state."""
        return self._current_state

    @property
    def current_plan(self) -> BatchPlan | None:
        """Current batch plan."""
        return self._current_plan

    @property
    def can_build_window(self) -> bool:
        """Whether buffer has enough samples to build a window."""
        return self.buffer.size >= self.config.min_samples

    @property
    def should_build_next(self) -> bool:
        """Whether it's time to build the next window."""
        if not self.has_window:
            return self.can_build_window
        return self._current_state.is_exhausted and self.can_build_window

    @property
    def total_windows(self) -> int:
        """Total windows built."""
        return self._total_windows

    @property
    def total_microbatches(self) -> int:
        """Total microbatches produced across all windows."""
        return self._total_microbatches

    # =========================================================================
    # Window Building
    # =========================================================================

    async def build_next_window(self) -> WindowState:
        """
        Build the next window from buffer snapshot.

        Returns:
            WindowState for the new window

        Raises:
            ValueError if buffer doesn't have enough samples
        """
        if not self.can_build_window:
            raise ValueError(
                f"Buffer has {self.buffer.size} samples, "
                f"need {self.config.min_samples} to build window"
            )

        # Snapshot buffer
        self._current_snapshot = self.buffer.snapshot()

        # Build plan from snapshot lengths
        lengths = self._current_snapshot.get_lengths()
        window_seed = self.config.seed + self._window_id
        batching_config = self.config.to_batching_config(window_seed)

        # Use snapshot timestamp as dataset hash for uniqueness
        dataset_hash = f"buffer_{self._current_snapshot.timestamp.isoformat()}"

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=batching_config,
            dataset_hash=dataset_hash,
            tokenizer_hash="streaming",  # Placeholder for streaming context
        )
        self._current_plan = await builder.build(num_epochs=1)

        # Update state
        self._window_id += 1
        self._total_windows += 1

        self._current_state = WindowState(
            window_id=self._window_id,
            snapshot_size=self._current_snapshot.size,
            snapshot_total_added=self._current_snapshot.total_added,
            plan_microbatches=self._current_plan.total_microbatches,
            plan_fingerprint=self._current_plan.fingerprint,
        )

        return self._current_state

    def build_next_window_sync(self) -> WindowState:
        """
        Synchronous version of build_next_window.

        For use in non-async contexts.
        """
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.build_next_window())

    # =========================================================================
    # Iteration
    # =========================================================================

    def iter_current(self):
        """
        Iterate over microbatches in current window.

        Yields MicrobatchSpec objects from the current plan.
        Updates consumed count as iteration progresses.

        Example:
            for mb in window.iter_current():
                samples = [snapshot_samples[sid] for sid in mb.samples]
                batch = collate(samples, mb.max_len)
                train_step(batch)
        """
        if not self.has_window:
            return

        for mb in self._current_plan.iter_epoch(epoch=0):
            # Update progress before yielding so it's accurate when caller checks
            self._current_state.microbatches_consumed += 1
            self._total_microbatches += 1
            yield mb

    def iter_current_with_samples(self):
        """
        Iterate over microbatches with their samples.

        Yields (MicrobatchSpec, list[StreamSample]) tuples.
        """
        if not self.has_window or self._current_snapshot is None:
            return

        # Build sample lookup
        sample_lookup = {s.sample_id: s for s in self._current_snapshot.samples}

        for mb in self.iter_current():
            samples = [sample_lookup[sid] for sid in mb.samples]
            yield mb, samples

    # =========================================================================
    # Window Management
    # =========================================================================

    def discard_current(self) -> None:
        """Discard current window without fully consuming it."""
        self._current_plan = None
        self._current_snapshot = None
        self._current_state = None

    def get_remaining_microbatches(self) -> int:
        """Get number of remaining microbatches in current window."""
        if not self.has_window:
            return 0
        return self._current_state.plan_microbatches - self._current_state.microbatches_consumed

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize window state for checkpointing."""
        return {
            "window_id": self._window_id,
            "total_windows": self._total_windows,
            "total_microbatches": self._total_microbatches,
            "config": self.config.model_dump(),
            "current_state": self._current_state.model_dump() if self._current_state else None,
        }

    @classmethod
    def from_dict(cls, data: dict, buffer: ReplayBuffer) -> RollingBatchPlanWindow:
        """Restore window state from checkpoint."""
        config = WindowConfig(**data["config"])
        window = cls(buffer, config)
        window._window_id = data["window_id"]
        window._total_windows = data["total_windows"]
        window._total_microbatches = data["total_microbatches"]
        if data["current_state"]:
            window._current_state = WindowState(**data["current_state"])
        return window
