"""SFT training command handler.

This module provides the async SFT training implementation.
The CLI command is a thin wrapper that delegates to SFTTrainer.run().
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import SFTConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_sft_cmd(args: Namespace) -> None:
    """CLI entry point for SFT training command.

    This is a thin wrapper that:
    1. Converts CLI args to SFTConfig using from_args()
    2. Calls SFTTrainer.run() which handles all the logic
    3. Prints the result

    Args:
        args: Parsed command-line arguments
    """
    from ....training.trainers.sft_trainer import SFTTrainer, SFTTrainingConfig

    # Parse CLI args using shared config
    cli_config = SFTConfig.from_args(args)

    # Convert to trainer config
    trainer_config = SFTTrainingConfig(
        model=cli_config.model,
        data_path=cli_config.data,
        eval_data_path=cli_config.eval_data,
        output_dir=cli_config.output,
        num_epochs=cli_config.epochs,
        max_steps=cli_config.max_steps,
        batch_size=cli_config.batch_size,
        learning_rate=cli_config.learning_rate,
        max_length=cli_config.max_length,
        use_lora=cli_config.use_lora,
        lora_rank=cli_config.lora_rank,
        mask_prompt=cli_config.mask_prompt,
        log_interval=cli_config.log_interval,
    )

    # Run training - all logic is in the trainer
    result = SFTTrainer.run(trainer_config)

    # Format output for CLI
    cli_result = TrainResult(
        mode=TrainMode.SFT,
        checkpoint_dir=result.output_dir,
        epochs_completed=result.epochs_completed,
    )
    print(cli_result.to_display())


__all__ = [
    "train_sft_cmd",
]
