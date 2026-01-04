"""SFT training command handler.

This module provides the async SFT training implementation.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import SFTConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_sft(config: SFTConfig) -> TrainResult:
    """Run SFT training.

    Args:
        config: SFT training configuration

    Returns:
        TrainResult with training outcomes
    """
    from ....data import SFTDataset
    from ....models import load_model
    from ....training import SFTTrainer
    from ....training.losses import SFTConfig as SFTTrainerConfig

    logger.info(f"Loading model: {config.model}")
    model = load_model(config.model, use_lora=config.use_lora, lora_rank=config.lora_rank)

    logger.info(f"Loading dataset: {config.data}")
    dataset = SFTDataset(
        str(config.data),
        model.tokenizer,
        max_length=config.max_length,
        mask_prompt=config.mask_prompt,
    )

    eval_dataset = None
    if config.eval_data:
        eval_dataset = SFTDataset(
            str(config.eval_data),
            model.tokenizer,
            max_length=config.max_length,
            mask_prompt=config.mask_prompt,
        )

    trainer_config = SFTTrainerConfig(
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_seq_length=config.max_length,
        checkpoint_dir=str(config.output),
        log_interval=config.log_interval,
    )

    trainer = SFTTrainer(model.model, model.tokenizer, trainer_config)
    trainer.train(dataset, eval_dataset)

    logger.info(f"Training complete. Checkpoints saved to {config.output}")

    return TrainResult(
        mode=TrainMode.SFT,
        checkpoint_dir=config.output,
        epochs_completed=config.epochs,
    )


async def train_sft_cmd(args: Namespace) -> None:
    """CLI entry point for SFT training command.

    Args:
        args: Parsed command-line arguments
    """
    config = SFTConfig.from_args(args)
    result = await train_sft(config)
    print(result.to_display())


__all__ = [
    "train_sft",
    "train_sft_cmd",
]
