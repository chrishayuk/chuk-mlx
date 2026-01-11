"""Gym run command handler.

This module provides the async gym stream implementation.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path

from ._types import GymRunConfig, GymRunResult

logger = logging.getLogger(__name__)


async def gym_run(config: GymRunConfig) -> GymRunResult:
    """Run gym episode streaming and collect samples.

    Args:
        config: Gym run configuration

    Returns:
        GymRunResult with streaming outcomes
    """
    from ....data.batching.streaming import (
        GymConfig,
        GymEpisodeStream,
        GymOutputMode,
        GymTransport,
        MockGymStream,
        ReplayBuffer,
        ReplayBufferConfig,
    )
    from ....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Configure replay buffer
    buffer_config = ReplayBufferConfig(
        max_size=config.buffer_size,
        seed=config.seed,
    )
    buffer = ReplayBuffer(buffer_config)

    # Configure gym stream
    if config.mock:
        logger.info("Using mock gym stream for testing")
        stream = MockGymStream(
            tokenizer=tokenizer,
            num_episodes=config.num_episodes,
            steps_per_episode=config.steps_per_episode,
            difficulty_range=(config.difficulty_min, config.difficulty_max),
            success_rate=config.success_rate,
            seed=config.seed,
        )
    else:
        transport = GymTransport(config.transport)
        output_mode = GymOutputMode(config.output_mode)

        gym_config = GymConfig(
            host=config.host,
            port=config.port,
            transport=transport,
            output_mode=output_mode,
            connect_timeout=config.timeout,
            max_retries=config.retries,
            difficulty_range=(config.difficulty_min, config.difficulty_max),
        )

        stream = GymEpisodeStream(
            config=gym_config,
            tokenizer=tokenizer,
        )

    # Run streaming
    logger.info(f"Starting gym stream to {config.host}:{config.port}")
    print(f"\n{'=' * 60}")
    print("Gym Episode Streaming")
    print(f"{'=' * 60}")

    sample_count = 0
    episode_ids: set[str] = set()

    async with stream:
        async for sample in stream:
            buffer.add(sample)
            sample_count += 1
            if sample.episode_id:
                episode_ids.add(sample.episode_id)

            if sample_count % 100 == 0:
                print(
                    f"  Samples: {sample_count}, "
                    f"Episodes: {len(episode_ids)}, "
                    f"Buffer: {buffer.size}"
                )

            if config.max_samples and sample_count >= config.max_samples:
                logger.info(f"Reached max samples: {config.max_samples}")
                break

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Total samples:    {sample_count}")
    print(f"  Total episodes:   {len(episode_ids)}")
    print(f"  Buffer size:      {buffer.size}")
    print(f"  Success rate:     {buffer.success_rate:.1%}")
    print(f"  Mean difficulty:  {buffer.mean_difficulty:.2f}")
    print(f"  Mean reward:      {buffer.mean_reward:.2f}")

    output_path = None
    if config.output:
        output_path = Path(config.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        buffer_data = buffer.to_dict()
        with open(output_path, "w") as f:
            json.dump(buffer_data, f, indent=2, default=str)

        print(f"\n  Buffer saved to: {output_path}")

    return GymRunResult(
        total_samples=sample_count,
        total_episodes=len(episode_ids),
        buffer_size=buffer.size,
        success_rate=buffer.success_rate,
        mean_difficulty=buffer.mean_difficulty,
        mean_reward=buffer.mean_reward,
        output_path=output_path,
    )


async def gym_run_cmd(args: Namespace) -> None:
    """CLI entry point for gym run command.

    Args:
        args: Parsed command-line arguments
    """
    config = GymRunConfig.from_args(args)
    result = await gym_run(config)
    print(result.to_display())


__all__ = [
    "gym_run",
    "gym_run_cmd",
]
