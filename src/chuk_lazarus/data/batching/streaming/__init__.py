"""
Streaming infrastructure for dynamic gym integration.

This module provides:
- SampleStream: Protocol for sample sources (offline, gym, replay buffer)
- ReplayBuffer: Bounded buffer for online learning
- RollingBatchPlanWindow: Build plans over rolling buffer snapshots
- GymEpisodeStream: Stream episodes from puzzle gym environments

Supports both offline (static datasets) and online (gym) training modes.

Design principles:
- Async-native: All streaming operations are async
- Protocol-based: Composable stream sources
- Type-safe: Pydantic models with no magic strings
- Memory-efficient: Bounded buffers with configurable eviction
"""

from .gym_stream import (
    GymConfig,
    GymEpisode,
    GymEpisodeStream,
    GymOutputMode,
    GymStep,
    GymTransport,
    MockGymStream,
)
from .offline_stream import (
    AsyncOfflineDatasetStream,
    OfflineDatasetStream,
)
from .protocols import (
    AsyncSampleStream,
    SampleStream,
)
from .replay_buffer import (
    BufferEvictionPolicy,
    BufferSnapshot,
    ReplayBuffer,
    ReplayBufferConfig,
)
from .rolling_window import (
    RollingBatchPlanWindow,
    WindowConfig,
    WindowState,
)
from .telnet_client import (
    DifficultyProfile,
    PuzzleDifficulty,
    PuzzleGame,
    PuzzleObservation,
    PuzzleResult,
    TelnetClientConfig,
    TelnetGymClient,
)
from .types import (
    EpisodeStatus,
    SampleSource,
    StreamConfig,
    StreamMetrics,
    StreamSample,
    StreamState,
)

__all__ = [
    # Types
    "SampleSource",
    "EpisodeStatus",
    "StreamSample",
    "StreamConfig",
    "StreamMetrics",
    "StreamState",
    # Protocols
    "SampleStream",
    "AsyncSampleStream",
    # Replay Buffer
    "ReplayBuffer",
    "ReplayBufferConfig",
    "BufferEvictionPolicy",
    "BufferSnapshot",
    # Offline Stream
    "OfflineDatasetStream",
    "AsyncOfflineDatasetStream",
    # Rolling Window
    "RollingBatchPlanWindow",
    "WindowConfig",
    "WindowState",
    # Gym Stream
    "GymConfig",
    "GymTransport",
    "GymOutputMode",
    "GymStep",
    "GymEpisode",
    "GymEpisodeStream",
    "MockGymStream",
    # Telnet Client
    "TelnetClientConfig",
    "TelnetGymClient",
    "PuzzleGame",
    "PuzzleDifficulty",
    "PuzzleObservation",
    "PuzzleResult",
    "DifficultyProfile",
]
