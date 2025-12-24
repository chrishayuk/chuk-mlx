"""
Gym episode stream for puzzle arcade integration.

Provides async streaming of episodes from puzzle gym environments.
Connects via telnet/TCP/WebSocket to the puzzle-arcade-server.

Requires external tokenizer for converting text to token IDs.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .types import (
    EpisodeStatus,
    SampleSource,
    StreamMetrics,
    StreamSample,
    StreamState,
)

# =============================================================================
# Tokenizer Protocol
# =============================================================================


class Tokenizer(Protocol):
    """Protocol for tokenizers compatible with gym streaming."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        ...


# =============================================================================
# Gym Configuration
# =============================================================================


class GymTransport(str, Enum):
    """Transport protocol for gym connection."""

    TELNET = "telnet"
    TCP = "tcp"
    WEBSOCKET = "websocket"


class GymOutputMode(str, Enum):
    """Output mode for gym responses."""

    NORMAL = "normal"  # Human-readable
    AGENT = "agent"  # Agent-friendly
    JSON = "json"  # Structured JSON


class GymConfig(BaseModel):
    """Configuration for gym connection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Connection
    host: str = Field(default="localhost", description="Gym server host")
    port: int = Field(default=8023, description="Gym server port")
    transport: GymTransport = Field(default=GymTransport.TELNET, description="Transport protocol")
    output_mode: GymOutputMode = Field(default=GymOutputMode.JSON, description="Output format")

    # Timeouts
    connect_timeout: float = Field(default=10.0, gt=0, description="Connection timeout")
    read_timeout: float = Field(default=30.0, gt=0, description="Read timeout per step")
    episode_timeout: float = Field(default=300.0, gt=0, description="Max episode duration")

    # Puzzle selection
    puzzle_types: tuple[str, ...] | None = Field(
        default=None,
        description="Specific puzzle types to request (None = random)",
    )
    difficulty_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Difficulty range (0-1)",
    )

    # Retry
    max_retries: int = Field(default=3, ge=0, description="Max connection retries")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries")


# =============================================================================
# Episode Data
# =============================================================================


class GymStep(BaseModel):
    """A single step in a gym episode."""

    model_config = ConfigDict(frozen=True)

    step_index: int = Field(description="Step number in episode")
    observation: str = Field(description="Observation/prompt text")
    action: str | None = Field(default=None, description="Action taken")
    reward: float = Field(default=0.0, description="Step reward")
    done: bool = Field(default=False, description="Episode terminated")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional info")


class GymEpisode(BaseModel):
    """A complete gym episode."""

    model_config = ConfigDict(frozen=True)

    episode_id: str = Field(description="Unique episode identifier")
    puzzle_type: str = Field(description="Type of puzzle")
    difficulty: float = Field(description="Difficulty level 0-1")

    steps: tuple[GymStep, ...] = Field(description="Episode steps")
    total_reward: float = Field(description="Cumulative reward")
    success: bool = Field(description="Whether puzzle was solved")
    status: EpisodeStatus = Field(description="Final status")

    start_time: datetime = Field(description="Episode start time")
    end_time: datetime = Field(description="Episode end time")

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


# =============================================================================
# Gym Episode Stream
# =============================================================================


class GymEpisodeStream:
    """
    Async stream of episodes from puzzle gym environments.

    Connects to puzzle-arcade-server and streams episodes as
    StreamSamples for training.

    Example:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        config = GymConfig(host="localhost", port=8023)

        async with GymEpisodeStream(config, tokenizer) as stream:
            async for sample in stream:
                buffer.add(sample)
                if buffer.size >= 1000:
                    # Train on buffer
                    ...

    Note:
        This is designed to work with the puzzle-arcade-server.
        The actual protocol implementation depends on the server's interface.
    """

    def __init__(
        self,
        config: GymConfig,
        tokenizer: Tokenizer,
        dataset_id: str = "gym",
        max_episodes: int | None = None,
        min_length: int = 1,
        max_length: int = 8192,
    ):
        """
        Initialize gym episode stream.

        Args:
            config: Gym connection configuration
            tokenizer: Tokenizer for encoding text to tokens
            dataset_id: Dataset identifier for samples
            max_episodes: Maximum episodes to stream (None = infinite)
            min_length: Minimum sequence length
            max_length: Maximum sequence length
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_id = dataset_id
        self.max_episodes = max_episodes
        self.min_length = min_length
        self.max_length = max_length

        # State
        self._state = StreamState.IDLE
        self._metrics = StreamMetrics()
        self._episode_count = 0

        # Connection
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    # =========================================================================
    # Protocol Properties
    # =========================================================================

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    # =========================================================================
    # Connection
    # =========================================================================

    async def connect(self) -> None:
        """Establish connection to gym server."""
        if self._connected:
            return

        for attempt in range(self.config.max_retries + 1):
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, self.config.port),
                    timeout=self.config.connect_timeout,
                )
                self._connected = True
                self._state = StreamState.STREAMING

                # Send initial configuration
                await self._configure_session()
                return

            except (ConnectionError, asyncio.TimeoutError) as e:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self._state = StreamState.ERROR
                    raise ConnectionError(
                        f"Failed to connect to gym at {self.config.host}:{self.config.port} "
                        f"after {self.config.max_retries + 1} attempts: {e}"
                    ) from e

    async def disconnect(self) -> None:
        """Close connection to gym server."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False
        self._state = StreamState.IDLE

    async def _configure_session(self) -> None:
        """Send session configuration to server."""
        # Set output mode to JSON for structured responses
        if self.config.output_mode == GymOutputMode.JSON:
            await self._send_command("mode json")

        # Configure difficulty if specified
        if self.config.difficulty_range != (0.0, 1.0):
            min_d, max_d = self.config.difficulty_range
            await self._send_command(f"difficulty {min_d} {max_d}")

    async def _send_command(self, command: str) -> str:
        """Send a command and get response."""
        if not self._writer or not self._reader:
            raise RuntimeError("Not connected to gym server")

        self._writer.write((command + "\n").encode())
        await self._writer.drain()

        response = await asyncio.wait_for(
            self._reader.readline(),
            timeout=self.config.read_timeout,
        )
        return response.decode().strip()

    # =========================================================================
    # Episode Generation
    # =========================================================================

    async def _run_episode(self) -> GymEpisode | None:
        """Run a single episode and return the result."""
        episode_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        steps: list[GymStep] = []
        total_reward = 0.0

        try:
            # Select puzzle
            puzzle_type = await self._select_puzzle()

            # Get initial observation
            observation = await self._get_observation()
            step_index = 0

            while True:
                # Record step
                step = GymStep(
                    step_index=step_index,
                    observation=observation,
                )
                steps.append(step)

                # Get action (placeholder - in real use, this comes from model)
                # For data collection, we might use the server's solution
                action = await self._get_action()

                # Execute action
                response = await self._send_command(action)
                result = self._parse_response(response)

                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                total_reward += reward

                # Update step with action and result
                steps[-1] = GymStep(
                    step_index=step_index,
                    observation=observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=result.get("info", {}),
                )

                if done:
                    break

                # Next observation
                observation = result.get("observation", "")
                step_index += 1

                # Timeout check
                if (datetime.now() - start_time).total_seconds() > self.config.episode_timeout:
                    return GymEpisode(
                        episode_id=episode_id,
                        puzzle_type=puzzle_type,
                        difficulty=result.get("difficulty", 0.5),
                        steps=tuple(steps),
                        total_reward=total_reward,
                        success=False,
                        status=EpisodeStatus.TIMEOUT,
                        start_time=start_time,
                        end_time=datetime.now(),
                    )

            # Determine success
            success = result.get("success", total_reward > 0)

            return GymEpisode(
                episode_id=episode_id,
                puzzle_type=puzzle_type,
                difficulty=result.get("difficulty", 0.5),
                steps=tuple(steps),
                total_reward=total_reward,
                success=success,
                status=EpisodeStatus.SUCCESS if success else EpisodeStatus.FAILURE,
                start_time=start_time,
                end_time=datetime.now(),
            )

        except asyncio.TimeoutError:
            return GymEpisode(
                episode_id=episode_id,
                puzzle_type="unknown",
                difficulty=0.5,
                steps=tuple(steps),
                total_reward=total_reward,
                success=False,
                status=EpisodeStatus.TIMEOUT,
                start_time=start_time,
                end_time=datetime.now(),
            )
        except Exception:
            return None

    async def _select_puzzle(self) -> str:
        """Select a puzzle type."""
        if self.config.puzzle_types:
            # Cycle through specified puzzles
            idx = self._episode_count % len(self.config.puzzle_types)
            puzzle = self.config.puzzle_types[idx]
            await self._send_command(f"puzzle {puzzle}")
            return puzzle
        else:
            # Random puzzle
            response = await self._send_command("puzzle random")
            result = self._parse_response(response)
            return result.get("puzzle_type", "unknown")

    async def _get_observation(self) -> str:
        """Get current observation from server."""
        response = await self._send_command("observe")
        result = self._parse_response(response)
        return result.get("observation", "")

    async def _get_action(self) -> str:
        """Get action to take."""
        # In real use, this would come from the model
        # For data collection, request solution hint
        response = await self._send_command("hint")
        result = self._parse_response(response)
        return result.get("action", "")

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse server response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw": response}

    # =========================================================================
    # Sample Conversion
    # =========================================================================

    def _episode_to_samples(self, episode: GymEpisode) -> list[StreamSample]:
        """Convert an episode to training samples."""
        samples = []

        for step in episode.steps:
            # Build training sequence: observation + action
            if step.action is None:
                continue

            prompt = step.observation
            response = step.action

            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt)
            response_tokens = self.tokenizer.encode(response)
            eos = self.tokenizer.eos_token_id

            input_ids = prompt_tokens + response_tokens + [eos]
            loss_mask = [0] * len(prompt_tokens) + [1] * (len(response_tokens) + 1)

            # Check length
            if len(input_ids) < self.min_length or len(input_ids) > self.max_length:
                continue

            sample = StreamSample(
                input_ids=tuple(input_ids),
                loss_mask=tuple(loss_mask),
                sample_id=f"{episode.episode_id}_{step.step_index}",
                dataset_id=self.dataset_id,
                source=SampleSource.GYM,
                episode_id=episode.episode_id,
                episode_status=episode.status,
                step_index=step.step_index,
                total_steps=episode.num_steps,
                reward=step.reward,
                difficulty=episode.difficulty,
                success=episode.success,
            )
            samples.append(sample)

        return samples

    # =========================================================================
    # Async Iterator
    # =========================================================================

    async def __aiter__(self) -> AsyncIterator[StreamSample]:
        """Async iterate over samples from gym episodes."""
        self._state = StreamState.STREAMING

        try:
            await self.connect()

            while True:
                # Check episode limit
                if self.max_episodes and self._episode_count >= self.max_episodes:
                    break

                # Run episode
                episode = await self._run_episode()
                if episode is None:
                    self._metrics.samples_errored += 1
                    continue

                self._episode_count += 1
                self._metrics.record_episode(episode.success)

                # Convert to samples
                samples = self._episode_to_samples(episode)
                for sample in samples:
                    self._metrics.record_sample(sample)
                    yield sample

        finally:
            self._state = StreamState.EXHAUSTED

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> GymEpisodeStream:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def reset(self) -> None:
        """Reset stream state."""
        self._episode_count = 0
        self._state = StreamState.IDLE

    async def close(self) -> None:
        """Close stream and release resources."""
        await self.disconnect()


# =============================================================================
# Mock Gym Stream for Testing
# =============================================================================


class MockGymStream:
    """
    Mock gym stream for testing without actual server.

    Generates synthetic puzzle episodes for testing the
    streaming infrastructure.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_episodes: int = 100,
        steps_per_episode: int = 5,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
        success_rate: float = 0.7,
        seed: int = 42,
    ):
        """Initialize mock gym stream."""
        import random

        self.tokenizer = tokenizer
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.difficulty_range = difficulty_range
        self.success_rate = success_rate
        self._rng = random.Random(seed)

        self._state = StreamState.IDLE
        self._metrics = StreamMetrics()
        self._episode_count = 0

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    async def __aiter__(self) -> AsyncIterator[StreamSample]:
        """Generate mock samples."""
        self._state = StreamState.STREAMING

        puzzle_types = ["arithmetic", "logic", "pattern", "word"]

        for ep_idx in range(self.num_episodes):
            episode_id = f"mock_{ep_idx:04d}"
            puzzle = self._rng.choice(puzzle_types)
            difficulty = self._rng.uniform(*self.difficulty_range)
            success = self._rng.random() < self.success_rate

            self._metrics.record_episode(success)

            for step_idx in range(self.steps_per_episode):
                # Generate mock observation/action
                observation = f"Solve this {puzzle} puzzle (step {step_idx + 1}): [mock problem]"
                action = f"The answer is: [mock solution for step {step_idx + 1}]"

                # Tokenize
                prompt_tokens = self.tokenizer.encode(observation)
                response_tokens = self.tokenizer.encode(action)
                eos = self.tokenizer.eos_token_id

                input_ids = prompt_tokens + response_tokens + [eos]
                loss_mask = [0] * len(prompt_tokens) + [1] * (len(response_tokens) + 1)

                reward = 1.0 if success and step_idx == self.steps_per_episode - 1 else 0.0

                sample = StreamSample(
                    input_ids=tuple(input_ids),
                    loss_mask=tuple(loss_mask),
                    sample_id=f"{episode_id}_{step_idx}",
                    dataset_id="mock_gym",
                    source=SampleSource.GYM,
                    episode_id=episode_id,
                    episode_status=EpisodeStatus.SUCCESS if success else EpisodeStatus.FAILURE,
                    step_index=step_idx,
                    total_steps=self.steps_per_episode,
                    reward=reward,
                    difficulty=difficulty,
                    success=success,
                )

                self._metrics.record_sample(sample)
                yield sample

            self._episode_count += 1

        self._state = StreamState.EXHAUSTED

    async def reset(self) -> None:
        self._state = StreamState.IDLE
        self._episode_count = 0

    async def close(self) -> None:
        self._state = StreamState.IDLE

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
