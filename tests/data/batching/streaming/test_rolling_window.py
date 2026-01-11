"""Tests for rolling batch plan window."""

import pytest

from chuk_lazarus.data.batching.streaming import (
    ReplayBuffer,
    ReplayBufferConfig,
    RollingBatchPlanWindow,
    SampleSource,
    StreamSample,
    WindowConfig,
)


def make_sample(sample_id: str, length: int = 100) -> StreamSample:
    """Helper to create test samples."""
    return StreamSample(
        input_ids=tuple(range(length)),
        loss_mask=tuple([1] * length),
        sample_id=sample_id,
        dataset_id="test",
        source=SampleSource.OFFLINE,
    )


@pytest.fixture
def populated_buffer():
    """Create a buffer with samples."""
    buffer = ReplayBuffer(ReplayBufferConfig(max_size=10000))
    for i in range(500):
        length = 50 + (i % 200)  # Varying lengths
        buffer.add(make_sample(f"sample_{i}", length=length))
    return buffer


class TestWindowConfig:
    """Tests for WindowConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = WindowConfig()

        assert config.window_microbatches == 1000
        assert config.min_samples == 100
        assert config.overlap_fraction == 0.1
        assert config.token_budget == 4096

    def test_to_batching_config(self):
        """Test conversion to BatchingConfig."""
        config = WindowConfig(
            token_budget=2048,
            bucket_edges=(128, 256, 512),
            overflow_max=1024,
        )

        batching_config = config.to_batching_config(window_seed=42)

        assert batching_config.token_budget == 2048
        assert batching_config.bucket_edges == (128, 256, 512)
        assert batching_config.overflow_max == 1024


class TestRollingBatchPlanWindow:
    """Tests for RollingBatchPlanWindow."""

    def test_empty_buffer(self):
        """Test with empty buffer."""
        buffer = ReplayBuffer()
        window = RollingBatchPlanWindow(buffer)

        assert not window.has_window
        assert not window.can_build_window
        assert not window.should_build_next

    def test_can_build_window(self, populated_buffer):
        """Test checking if window can be built."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100),
        )

        assert window.can_build_window
        assert window.should_build_next  # No current window

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        buffer = ReplayBuffer()
        for i in range(50):  # Only 50 samples
            buffer.add(make_sample(f"sample_{i}"))

        window = RollingBatchPlanWindow(
            buffer,
            WindowConfig(min_samples=100),  # Need 100
        )

        assert not window.can_build_window

    @pytest.mark.asyncio
    async def test_insufficient_samples_raises(self):
        """Test that building with insufficient samples raises."""
        buffer = ReplayBuffer()
        for i in range(50):  # Only 50 samples
            buffer.add(make_sample(f"sample_{i}"))

        window = RollingBatchPlanWindow(
            buffer,
            WindowConfig(min_samples=100),  # Need 100
        )

        with pytest.raises(ValueError, match="need 100"):
            await window.build_next_window()

    @pytest.mark.asyncio
    async def test_build_window(self, populated_buffer):
        """Test building a window."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        state = await window.build_next_window()

        assert window.has_window
        assert state.window_id == 1
        assert state.snapshot_size == 500
        assert state.plan_microbatches > 0
        assert len(state.plan_fingerprint) > 0

    @pytest.mark.asyncio
    async def test_window_iteration(self, populated_buffer):
        """Test iterating through window microbatches."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        await window.build_next_window()

        # Count microbatches
        count = 0
        for mb in window.iter_current():
            count += 1
            assert len(mb.samples) > 0

        # Check final state after exhausting iterator
        assert count == window.current_state.plan_microbatches
        assert window.current_state.microbatches_consumed == count
        assert window.current_state.is_exhausted

    @pytest.mark.asyncio
    async def test_window_with_samples(self, populated_buffer):
        """Test iteration with sample data."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        await window.build_next_window()

        # Iterate with samples
        count = 0
        for mb, samples in window.iter_current_with_samples():
            assert len(samples) == len(mb.samples)
            for sample, sample_id in zip(samples, mb.samples):
                assert sample.sample_id == sample_id
            count += 1
            if count >= 5:  # Just test a few
                break

    @pytest.mark.asyncio
    async def test_multiple_windows(self, populated_buffer):
        """Test building multiple windows sequentially."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        # First window
        state1 = await window.build_next_window()
        assert state1.window_id == 1

        # Exhaust it
        for _ in window.iter_current():
            pass

        # Second window
        state2 = await window.build_next_window()
        assert state2.window_id == 2
        assert window.total_windows == 2

    @pytest.mark.asyncio
    async def test_should_build_next(self, populated_buffer):
        """Test should_build_next logic."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        # Initially should build (no current window)
        assert window.should_build_next

        # After building, shouldn't (not exhausted)
        await window.build_next_window()
        assert not window.should_build_next

        # Exhaust the window
        for _ in window.iter_current():
            pass

        # Now should build next
        assert window.should_build_next

    @pytest.mark.asyncio
    async def test_discard_current(self, populated_buffer):
        """Test discarding current window."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        await window.build_next_window()
        assert window.has_window

        window.discard_current()

        assert not window.has_window
        assert window.current_state is None

    @pytest.mark.asyncio
    async def test_remaining_microbatches(self, populated_buffer):
        """Test counting remaining microbatches."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        await window.build_next_window()
        total = window.current_state.plan_microbatches

        # Consume exactly 5 microbatches
        count = 0
        for _ in window.iter_current():
            count += 1
            if count >= 5:
                break

        # After consuming 5, we should have consumed exactly 5
        assert window.current_state.microbatches_consumed == 5
        remaining = window.get_remaining_microbatches()
        assert remaining == total - 5

    @pytest.mark.asyncio
    async def test_serialization(self, populated_buffer):
        """Test window state serialization."""
        window = RollingBatchPlanWindow(
            populated_buffer,
            WindowConfig(min_samples=100, token_budget=2048),
        )

        await window.build_next_window()

        # Consume some
        for i, _ in enumerate(window.iter_current()):
            if i >= 3:
                break

        # Serialize
        data = window.to_dict()

        # Restore
        restored = RollingBatchPlanWindow.from_dict(data, populated_buffer)

        assert restored.total_windows == window.total_windows
        assert restored.total_microbatches == window.total_microbatches
        assert restored.config.token_budget == window.config.token_budget
