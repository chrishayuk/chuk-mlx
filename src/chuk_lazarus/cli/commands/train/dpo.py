"""DPO training command handler.

This module provides the async DPO training implementation.
The CLI command is a thin wrapper that delegates to DPOTrainer.run().
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import DPOConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_dpo_cmd(args: Namespace) -> None:
    """CLI entry point for DPO training command.

    This is a thin wrapper that:
    1. Converts CLI args to DPOConfig using from_args()
    2. Calls DPOTrainer.run() which handles all the logic
    3. Prints the result

    Args:
        args: Parsed command-line arguments
    """
    from ....training.trainers.dpo_trainer import DPOTrainer, DPOTrainingConfig

    # Parse CLI args using shared config
    cli_config = DPOConfig.from_args(args)

    # Convert to trainer config
    trainer_config = DPOTrainingConfig(
        model=cli_config.model,
        ref_model=cli_config.ref_model,
        data_path=cli_config.data,
        eval_data_path=cli_config.eval_data,
        output_dir=cli_config.output,
        num_epochs=cli_config.epochs,
        batch_size=cli_config.batch_size,
        learning_rate=cli_config.learning_rate,
        beta=cli_config.beta,
        max_length=cli_config.max_length,
        use_lora=cli_config.use_lora,
        lora_rank=cli_config.lora_rank,
    )

    # Run training - all logic is in the trainer
    result = DPOTrainer.run(trainer_config)

    # Format output for CLI
    cli_result = TrainResult(
        mode=TrainMode.DPO,
        checkpoint_dir=result.output_dir,
        epochs_completed=result.epochs_completed,
    )
    print(cli_result.to_display())


__all__ = [
    "train_dpo_cmd",
]
