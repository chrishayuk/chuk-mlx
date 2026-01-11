"""GRPO training command handler.

This module provides the async GRPO training implementation.
The CLI command is a thin wrapper that delegates to GRPOTrainer.run().

GRPO (Group Relative Policy Optimization) is an RL algorithm that:
- Generates multiple responses per prompt
- Uses group-relative advantages (no value function needed)
- Works well with verifiable rewards (e.g., arithmetic correctness)

The reward function is provided via a user-defined Python script with:
    def reward_fn(prompt: str, response: str) -> float
    def get_prompts() -> list[str]
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import GRPOConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_grpo_cmd(args: Namespace) -> None:
    """CLI entry point for GRPO training command.

    This is a thin wrapper that:
    1. Converts CLI args to GRPOConfig
    2. Calls GRPOTrainer.run() which handles all the logic
    3. Prints the result

    Args:
        args: Parsed command-line arguments
    """
    from ....training.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

    # Convert CLI args to config
    config = GRPOConfig.from_args(args)

    # Validate reward script is provided
    if config.reward_script is None:
        raise ValueError("--reward-script is required for GRPO training")

    # Create trainer config from CLI config
    trainer_config = GRPOTrainerConfig(
        model=config.model,
        ref_model=config.reference_model,
        reward_script=config.reward_script,
        output_dir=config.output,
        num_iterations=config.iterations,
        prompts_per_iteration=config.prompts_per_iteration,
        group_size=config.group_size,
        learning_rate=config.learning_rate,
        kl_coef=config.kl_coef,
        max_response_length=config.max_response_length,
        temperature=config.temperature,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
    )

    # Run training - all logic is in the trainer
    result = await GRPOTrainer.run(trainer_config)

    # Format output for CLI
    cli_result = TrainResult(
        mode=TrainMode.GRPO,
        checkpoint_dir=result.output_dir,
        epochs_completed=result.iterations_completed,
    )
    print(cli_result.to_display())


__all__ = [
    "train_grpo_cmd",
]
