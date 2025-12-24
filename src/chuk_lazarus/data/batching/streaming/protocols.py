"""
Protocol definitions for sample streams.

Defines the core SampleStream protocol that all sample sources implement.
Supports both sync and async iteration patterns.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Protocol, runtime_checkable

from .types import StreamMetrics, StreamSample, StreamState


@runtime_checkable
class SampleStream(Protocol):
    """
    Protocol for synchronous sample sources.

    Provides iterator interface for streaming samples from:
    - Offline datasets (JSONL files)
    - Replay buffers
    - Synthetic generators

    Example:
        stream = OfflineDatasetStream(path="train.jsonl")
        for sample in stream:
            process(sample)
    """

    @property
    @abstractmethod
    def state(self) -> StreamState:
        """Current stream state."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> StreamMetrics:
        """Accumulated stream metrics."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[StreamSample]:
        """Iterate over samples."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset stream to beginning."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close stream and release resources."""
        ...


@runtime_checkable
class AsyncSampleStream(Protocol):
    """
    Protocol for asynchronous sample sources.

    Provides async iterator interface for streaming samples from:
    - Gym environments (network I/O)
    - Remote datasets
    - Async replay buffers

    Example:
        async with GymEpisodeStream(host="localhost", port=8023) as stream:
            async for sample in stream:
                await process(sample)
    """

    @property
    @abstractmethod
    def state(self) -> StreamState:
        """Current stream state."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> StreamMetrics:
        """Accumulated stream metrics."""
        ...

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[StreamSample]:
        """Async iterate over samples."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset stream to beginning."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close stream and release resources."""
        ...

    @abstractmethod
    async def __aenter__(self) -> AsyncSampleStream:
        """Enter async context."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        ...
