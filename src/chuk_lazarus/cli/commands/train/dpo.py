"""DPO training command handler.

This module provides the async DPO training implementation.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import DPOConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_dpo(config: DPOConfig) -> TrainResult:
    """Run DPO training.

    Args:
        config: DPO training configuration

    Returns:
        TrainResult with training outcomes
    """
    from ....data import PreferenceDataset
    from ....models import load_model
    from ....training import DPOTrainer, DPOTrainerConfig
    from ....training.losses import DPOConfig as DPOLossConfig

    logger.info(f"Loading policy model: {config.model}")
    policy_model = load_model(config.model, use_lora=config.use_lora, lora_rank=config.lora_rank)

    logger.info(f"Loading reference model: {config.reference_model}")
    ref_model = load_model(config.reference_model, use_lora=False)

    logger.info(f"Loading dataset: {config.data}")
    dataset = PreferenceDataset(
        str(config.data),
        policy_model.tokenizer,
        max_length=config.max_length,
    )

    eval_dataset = None
    if config.eval_data:
        eval_dataset = PreferenceDataset(
            str(config.eval_data),
            policy_model.tokenizer,
            max_length=config.max_length,
        )

    trainer_config = DPOTrainerConfig(
        dpo=DPOLossConfig(beta=config.beta),
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        checkpoint_dir=str(config.output),
    )

    trainer = DPOTrainer(
        policy_model.model, ref_model.model, policy_model.tokenizer, trainer_config
    )
    trainer.train(dataset, eval_dataset)

    logger.info(f"Training complete. Checkpoints saved to {config.output}")

    return TrainResult(
        mode=TrainMode.DPO,
        checkpoint_dir=config.output,
        epochs_completed=config.epochs,
    )


async def train_dpo_cmd(args: Namespace) -> None:
    """CLI entry point for DPO training command.

    Args:
        args: Parsed command-line arguments
    """
    config = DPOConfig.from_args(args)
    result = await train_dpo(config)
    print(result.to_display())


__all__ = [
    "train_dpo",
    "train_dpo_cmd",
]
