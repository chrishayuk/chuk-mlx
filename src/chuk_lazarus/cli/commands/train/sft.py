"""SFT training command handler.

This module provides the async SFT training implementation.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path

from ._types import SFTConfig, TrainMode, TrainResult

logger = logging.getLogger(__name__)


async def train_sft(config: SFTConfig) -> TrainResult:
    """Run SFT training.

    Args:
        config: SFT training configuration

    Returns:
        TrainResult with training outcomes
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    from ....data import SFTDataset
    from ....models_v2 import LoRAConfig, apply_lora, count_lora_parameters
    from ....training.trainers.sft_trainer import SFTConfig as SFTTrainerConfig
    from ....training.trainers.sft_trainer import SFTTrainer

    # Load model and tokenizer using mlx-lm
    logger.info(f"Loading model: {config.model}")
    model, tokenizer = mlx_load(config.model)
    num_layers = len(model.model.layers)
    logger.info(f"  Model loaded: {num_layers} layers")

    # Apply LoRA if requested
    lora_layers = None
    if config.use_lora:
        lora_config = LoRAConfig(
            rank=config.lora_rank,
            alpha=20.0,  # mlx-lm default scale
            dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        lora_layers = apply_lora(model, lora_config)
        n_params = count_lora_parameters(lora_layers)
        logger.info(f"  Applied LoRA: {len(lora_layers)} layers, {n_params:,} trainable params")

    # Load dataset
    logger.info(f"Loading dataset: {config.data}")
    dataset = SFTDataset(
        str(config.data),
        tokenizer,
        max_length=config.max_length,
        mask_prompt=config.mask_prompt,
    )
    logger.info(f"  Loaded {len(dataset)} training samples")

    eval_dataset = None
    if config.eval_data:
        eval_dataset = SFTDataset(
            str(config.eval_data),
            tokenizer,
            max_length=config.max_length,
            mask_prompt=config.mask_prompt,
        )
        logger.info(f"  Loaded {len(eval_dataset)} eval samples")

    # Configure trainer
    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = SFTTrainerConfig(
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        checkpoint_dir=str(output_dir / "checkpoints"),
        log_interval=config.log_interval,
        max_steps=config.max_steps,
        checkpoint_interval=config.max_steps or 1000,
    )

    # Train
    logger.info(f"Starting training...")
    trainer = SFTTrainer(model, tokenizer, trainer_config)
    trainer.train(dataset, eval_dataset)

    # Save LoRA adapters in mlx-lm compatible format
    if config.use_lora and lora_layers is not None:
        adapter_dir = output_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Collect LoRA weights
        lora_weights = {}
        for name, lora_layer in lora_layers.items():
            lora_weights[f"model.{name}.lora_a"] = lora_layer.lora_A
            lora_weights[f"model.{name}.lora_b"] = lora_layer.lora_B

        # Save weights
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), lora_weights)

        # Save config
        adapter_config = {
            "model": config.model,
            "num_layers": num_layers,
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": config.lora_rank,
                "dropout": 0.0,
                "scale": 20.0,
            },
        }
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        logger.info(f"Saved LoRA adapters to {adapter_dir}")

    logger.info(f"Training complete. Output saved to {config.output}")

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
