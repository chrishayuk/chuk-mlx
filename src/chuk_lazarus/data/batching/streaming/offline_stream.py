"""
Offline dataset stream implementation.

Provides a SampleStream implementation for static dataset files (JSONL).
Converts raw samples to StreamSample format for unified processing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from .protocols import SampleStream
from .types import (
    EpisodeStatus,
    SampleSource,
    StreamMetrics,
    StreamSample,
    StreamState,
)


class OfflineDatasetStream:
    """
    Stream samples from a static dataset file.

    Reads JSONL files containing tokenized samples and produces
    StreamSample objects for the batching pipeline.

    Supports multiple file formats:
    - Full Sample format (with meta, loss_mask, etc.)
    - Simple format (just input_ids)
    - Instruction format (instruction/response pairs - tokenization required)

    Example:
        stream = OfflineDatasetStream(
            path="train.jsonl",
            dataset_id="my_dataset",
        )

        for sample in stream:
            print(sample.sample_id, sample.length)

        # Reuse for multiple epochs
        stream.reset()
        for sample in stream:
            ...
    """

    def __init__(
        self,
        path: str | Path,
        dataset_id: str | None = None,
        min_length: int = 1,
        max_length: int = 8192,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Initialize offline dataset stream.

        Args:
            path: Path to JSONL dataset file
            dataset_id: Optional dataset identifier (defaults to filename)
            min_length: Minimum sequence length to include
            max_length: Maximum sequence length to include
            shuffle: Whether to shuffle samples (loads all into memory)
            seed: Random seed for shuffling
        """
        self.path = Path(path)
        self.dataset_id = dataset_id or self.path.stem
        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed

        # State
        self._state: StreamState = StreamState.IDLE
        self._metrics: StreamMetrics = StreamMetrics()
        self._file_handle = None
        self._sample_index: int = 0

        # For shuffling
        self._cached_samples: list[StreamSample] | None = None

    # =========================================================================
    # Protocol Properties
    # =========================================================================

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        return self._state

    @property
    def metrics(self) -> StreamMetrics:
        """Accumulated stream metrics."""
        return self._metrics

    # =========================================================================
    # Iterator Protocol
    # =========================================================================

    def __iter__(self) -> Iterator[StreamSample]:
        """Iterate over samples in the dataset."""
        self._state = StreamState.STREAMING

        if self.shuffle:
            # Load all and shuffle
            yield from self._iter_shuffled()
        else:
            # Stream from file
            yield from self._iter_sequential()

        self._state = StreamState.EXHAUSTED

    def _iter_sequential(self) -> Iterator[StreamSample]:
        """Iterate through file sequentially."""
        with open(self.path) as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    sample = self._parse_line(line, line_num)
                    if sample is not None:
                        if self._filter_sample(sample):
                            self._metrics.record_sample(sample)
                            yield sample
                        else:
                            self._metrics.record_filtered()
                except Exception:
                    self._metrics.samples_errored += 1

    def _iter_shuffled(self) -> Iterator[StreamSample]:
        """Load all samples, shuffle, and iterate."""
        import random

        if self._cached_samples is None:
            self._cached_samples = []
            with open(self.path) as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        sample = self._parse_line(line, line_num)
                        if sample is not None and self._filter_sample(sample):
                            self._cached_samples.append(sample)
                    except Exception:
                        self._metrics.samples_errored += 1

        # Shuffle with seed
        rng = random.Random(self.seed)
        indices = list(range(len(self._cached_samples)))
        rng.shuffle(indices)

        for idx in indices:
            sample = self._cached_samples[idx]
            self._metrics.record_sample(sample)
            yield sample

    def _parse_line(self, line: str, line_num: int) -> StreamSample | None:
        """Parse a JSONL line into a StreamSample."""
        data = json.loads(line)

        # Handle different formats
        if "input_ids" in data:
            return self._parse_tokenized(data, line_num)
        elif "instruction" in data or "prompt" in data:
            # Instruction format - would need tokenizer
            # For now, skip these (require external tokenization)
            return None
        else:
            return None

    def _parse_tokenized(self, data: dict, line_num: int) -> StreamSample:
        """Parse tokenized sample format."""
        input_ids = tuple(data["input_ids"])

        # Get or generate loss_mask
        if "loss_mask" in data:
            loss_mask = tuple(data["loss_mask"])
        else:
            # Default: all tokens contribute to loss
            loss_mask = tuple([1] * len(input_ids))

        # Get optional segment_ids
        segment_ids = None
        if "segment_ids" in data:
            segment_ids = tuple(data["segment_ids"])

        # Get or generate sample_id
        if "meta" in data and isinstance(data["meta"], dict):
            sample_id = data["meta"].get("sample_id", f"{self.dataset_id}_{line_num}")
            episode_id = data["meta"].get("episode_id")
            reward = data["meta"].get("reward")
            difficulty = data["meta"].get("difficulty")
            difficulty_score = data["meta"].get("difficulty_score")
            success = data["meta"].get("success")
        elif "sample_id" in data:
            sample_id = data["sample_id"]
            episode_id = data.get("episode_id")
            reward = data.get("reward")
            difficulty = data.get("difficulty")
            difficulty_score = data.get("difficulty_score")
            success = data.get("success")
        else:
            sample_id = f"{self.dataset_id}_{line_num}"
            episode_id = None
            reward = None
            difficulty = None
            difficulty_score = None
            success = None

        # Normalize difficulty
        if difficulty is not None and isinstance(difficulty, str):
            # Map difficulty level names to scores
            difficulty_map = {
                "trivial": 0.1,
                "easy": 0.3,
                "medium": 0.5,
                "hard": 0.7,
                "expert": 0.9,
            }
            difficulty = difficulty_map.get(difficulty.lower(), 0.5)
        elif difficulty_score is not None:
            difficulty = difficulty_score

        return StreamSample(
            input_ids=input_ids,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            sample_id=sample_id,
            dataset_id=self.dataset_id,
            source=SampleSource.OFFLINE,
            timestamp=datetime.now(),
            episode_id=episode_id,
            episode_status=EpisodeStatus.SUCCESS if success else None,
            reward=reward,
            difficulty=difficulty,
            success=success,
        )

    def _filter_sample(self, sample: StreamSample) -> bool:
        """Check if sample passes filters."""
        length = sample.length
        return self.min_length <= length <= self.max_length

    # =========================================================================
    # Stream Control
    # =========================================================================

    def reset(self) -> None:
        """Reset stream to beginning."""
        self._state = StreamState.IDLE
        self._sample_index = 0
        # Keep metrics accumulated, but reset position
        # Increment seed for different shuffle order
        self.seed += 1

    def close(self) -> None:
        """Close stream and release resources."""
        self._state = StreamState.IDLE
        self._cached_samples = None

    # =========================================================================
    # Utilities
    # =========================================================================

    def count_samples(self) -> int:
        """Count total samples in file (without filtering)."""
        count = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def peek(self, n: int = 5) -> list[StreamSample]:
        """Peek at first n samples without advancing stream."""
        samples = []
        with open(self.path) as f:
            for line_num, line in enumerate(f):
                if len(samples) >= n:
                    break
                if not line.strip():
                    continue
                try:
                    sample = self._parse_line(line, line_num)
                    if sample is not None and self._filter_sample(sample):
                        samples.append(sample)
                except Exception:
                    pass
        return samples

    def get_length_distribution(self) -> dict[str, int]:
        """Get length distribution without loading full dataset."""
        lengths: dict[str, int] = {}
        with open(self.path) as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    sample = self._parse_line(line, line_num)
                    if sample is not None:
                        lengths[sample.sample_id] = sample.length
                except Exception:
                    pass
        return lengths


# Type check that we implement the protocol
def _check_protocol() -> None:
    """Verify OfflineDatasetStream implements SampleStream."""
    stream: SampleStream = OfflineDatasetStream(path="test.jsonl")  # noqa: F841


# Async version for completeness
class AsyncOfflineDatasetStream:
    """
    Async version of OfflineDatasetStream.

    Uses aiofiles for non-blocking file I/O.
    """

    def __init__(
        self,
        path: str | Path,
        dataset_id: str | None = None,
        min_length: int = 1,
        max_length: int = 8192,
    ):
        """Initialize async offline stream."""
        self.path = Path(path)
        self.dataset_id = dataset_id or self.path.stem
        self.min_length = min_length
        self.max_length = max_length
        self._state = StreamState.IDLE
        self._metrics = StreamMetrics()

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    async def __aiter__(self):
        """Async iterate over samples."""
        import aiofiles

        self._state = StreamState.STREAMING

        async with aiofiles.open(self.path) as f:
            line_num = 0
            async for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    sample = self._parse_tokenized(data, line_num)
                    if sample is not None and self._filter_sample(sample):
                        self._metrics.record_sample(sample)
                        yield sample
                    else:
                        self._metrics.record_filtered()
                except Exception:
                    self._metrics.samples_errored += 1

                line_num += 1

        self._state = StreamState.EXHAUSTED

    def _parse_tokenized(self, data: dict, line_num: int) -> StreamSample | None:
        """Parse tokenized sample format."""
        if "input_ids" not in data:
            return None

        input_ids = tuple(data["input_ids"])
        loss_mask = tuple(data.get("loss_mask", [1] * len(input_ids)))
        segment_ids = tuple(data["segment_ids"]) if "segment_ids" in data else None

        sample_id = data.get("sample_id", f"{self.dataset_id}_{line_num}")
        if "meta" in data and isinstance(data["meta"], dict):
            sample_id = data["meta"].get("sample_id", sample_id)

        return StreamSample(
            input_ids=input_ids,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            sample_id=sample_id,
            dataset_id=self.dataset_id,
            source=SampleSource.OFFLINE,
        )

    def _filter_sample(self, sample: StreamSample) -> bool:
        return self.min_length <= sample.length <= self.max_length

    async def reset(self) -> None:
        self._state = StreamState.IDLE

    async def close(self) -> None:
        self._state = StreamState.IDLE

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
