"""
SFT Trainer - Supervised Fine-Tuning for Tool Use.

This trainer handles the initial supervised training phase:
- Teach the model tool-calling syntax
- Learn structured output formats
- Build foundation for RL fine-tuning

Stage 1 of the Lazarus training pipeline:
    SFT (learn syntax) -> DPO (learn preferences) -> Deploy

Usage:
    # High-level API (recommended for CLI):
    result = SFTTrainer.run(SFTTrainingConfig(
        model="meta-llama/Llama-3.2-1B",
        data_path="train.jsonl",
        output_dir="./output",
    ))

    # Low-level API (for custom pipelines):
    trainer = SFTTrainer(model, tokenizer, config)
    trainer.train(dataset)
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pydantic import BaseModel, Field

from ...data import SFTDataset
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.sft_loss import sft_loss

logger = logging.getLogger(__name__)


class SFTTrainingConfig(BaseModel):
    """Complete configuration for running SFT training.

    This is the high-level config used by CLI and run() method.
    Includes model path, data paths, and all training parameters.
    """

    # Model
    model: str = Field(..., description="Model path or HuggingFace name")
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(default=16.0, description="LoRA alpha scaling")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="LoRA target modules",
    )

    # Data
    data_path: Path = Field(..., description="Path to training data (JSONL)")
    eval_data_path: Path | None = Field(default=None, description="Path to eval data")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")
    mask_prompt: bool = Field(default=False, description="Mask prompt in loss")

    # Training
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    max_steps: int | None = Field(default=None, description="Max steps (overrides epochs)")

    # Output
    output_dir: Path = Field(default=Path("./checkpoints/sft"), description="Output directory")
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")


class SFTTrainingResult(BaseModel):
    """Result of SFT training."""

    output_dir: Path = Field(..., description="Output directory")
    epochs_completed: int = Field(..., description="Epochs completed")
    final_loss: float | None = Field(default=None, description="Final training loss")
    adapter_path: Path | None = Field(default=None, description="Path to saved LoRA adapter")


@dataclass
class SFTConfig(BaseTrainerConfig):
    """Low-level configuration for SFT trainer.

    Used internally by the trainer. For high-level usage, see SFTTrainingConfig.
    """

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Logging and checkpoints
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints/sft"

    # Early stopping
    max_steps: int | None = None
    min_loss: float | None = None


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning.

    This is Stage 1 of the Lazarus training pipeline:
    - Teaches the model tool-calling syntax
    - Learns structured output formats
    - Prepares for DPO preference learning

    Usage:
        # High-level API (recommended):
        result = SFTTrainer.run(SFTTrainingConfig(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            data_path="train.jsonl",
        ))

        # Low-level API:
        trainer = SFTTrainer(model, tokenizer, config)
        trainer.train(dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        config = config or SFTConfig()
        super().__init__(model, tokenizer, config, optimizer)

    @classmethod
    def run(cls, config: SFTTrainingConfig) -> SFTTrainingResult:
        """Run complete SFT training from config.

        This is the high-level entry point that handles:
        - Model loading (with optional LoRA)
        - Dataset loading
        - Training
        - Checkpoint saving

        Args:
            config: Complete training configuration

        Returns:
            SFTTrainingResult with training outcomes
        """
        from ...models_v2 import LoRAConfig, load_model, load_model_with_lora, save_adapter

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading model: {config.model}")
        lora_layers = None
        lora_config = None

        if config.use_lora:
            lora_config = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=0.0,
                target_modules=config.lora_targets,
            )
            result = load_model_with_lora(config.model, lora_config)
            model = result.model
            tokenizer = result.tokenizer
            lora_layers = result.lora_layers
            logger.info(
                f"  Loaded with LoRA: {len(lora_layers)} layers, "
                f"{result.lora_parameter_count:,} trainable params"
            )
        else:
            result = load_model(config.model)
            model = result.model
            tokenizer = result.tokenizer

        # Load datasets
        logger.info(f"Loading dataset: {config.data_path}")
        train_dataset = SFTDataset(
            str(config.data_path),
            tokenizer,
            max_length=config.max_length,
            mask_prompt=config.mask_prompt,
        )
        logger.info(f"  Loaded {len(train_dataset)} training samples")

        eval_dataset = None
        if config.eval_data_path:
            eval_dataset = SFTDataset(
                str(config.eval_data_path),
                tokenizer,
                max_length=config.max_length,
                mask_prompt=config.mask_prompt,
            )
            logger.info(f"  Loaded {len(eval_dataset)} eval samples")

        # Create trainer config
        trainer_config = SFTConfig(
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            checkpoint_dir=str(output_dir / "checkpoints"),
            log_interval=config.log_interval,
            max_steps=config.max_steps,
            checkpoint_interval=config.checkpoint_interval,
        )

        # Create and run trainer
        trainer = cls(model, tokenizer, trainer_config)

        # Attach LoRA layers for checkpoint saving
        if lora_layers:
            trainer.lora_layers = lora_layers
            trainer.lora_config = lora_config

        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)

        # Save LoRA adapters
        adapter_path = None
        if config.use_lora and lora_layers:
            adapter_path = output_dir / "adapters"
            save_adapter(lora_layers, adapter_path, lora_config=lora_config)
            logger.info(f"Saved LoRA adapters to {adapter_path}")

        # Get final loss from metrics
        final_loss = None
        if trainer.metrics_history:
            final_loss = trainer.metrics_history[-1].get("loss")

        logger.info(f"Training complete. Output saved to {output_dir}")

        return SFTTrainingResult(
            output_dir=output_dir,
            epochs_completed=config.num_epochs,
            final_loss=final_loss,
            adapter_path=adapter_path,
        )

    @property
    def sft_config(self) -> SFTConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute SFT loss for a batch."""
        output = self.model(batch["input_ids"])
        # Handle different model output formats:
        # - Some models return just logits (mlx-lm models)
        # - Some return (logits, cache) tuple
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        loss, metrics = sft_loss(
            logits=logits, labels=batch["labels"], loss_mask=batch["loss_mask"]
        )
        return loss, metrics

    def get_train_batches(self, dataset: SFTDataset) -> Iterator[dict[str, mx.array]]:
        """Get iterator over training batches."""
        return dataset.iter_batches(
            batch_size=self.sft_config.batch_size, shuffle=True, pad_token_id=self.pad_token_id
        )

    def train(
        self,
        train_dataset: SFTDataset,
        eval_dataset: SFTDataset = None,
        callback: Callable[[dict], None] = None,
    ):
        """
        Run SFT training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callback: Optional callback after each log interval
        """
        logger.info(f"Starting SFT training with {len(train_dataset)} samples")
        super().train(
            dataset=train_dataset,
            num_epochs=self.sft_config.num_epochs,
            eval_dataset=eval_dataset,
            callback=callback,
        )

    def evaluate(self, dataset: SFTDataset) -> dict[str, float]:
        """Evaluate on a dataset."""
        all_metrics = {"loss": [], "perplexity": [], "num_tokens": []}

        for batch in dataset.iter_batches(
            batch_size=self.sft_config.batch_size, shuffle=False, pad_token_id=self.pad_token_id
        ):
            output = self.model(batch["input_ids"])
            # Handle different model output formats
            logits = output[0] if isinstance(output, tuple) else output
            loss, metrics = sft_loss(
                logits=logits, labels=batch["labels"], loss_mask=batch["loss_mask"]
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Create SFT-specific metrics accumulator."""
        return {
            "loss": [],
            "perplexity": [],
            "num_tokens": [],
        }

    def _log_metrics(self, metrics: dict[str, float]):
        """Log SFT-specific metrics."""
        import time

        elapsed = time.time() - self._start_time

        # Calculate tokens per second if we have token counts
        epoch_metrics = getattr(self, "_current_epoch_metrics", {})
        total_tokens = sum(epoch_metrics.get("num_tokens", [0]))
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {metrics.get('loss', 0):.4f} | "
            f"PPL: {metrics.get('perplexity', 0):.2f} | "
            f"Tok/s: {tokens_per_sec:.0f} | "
            f"Time: {elapsed:.1f}s"
        )

        self.metrics_history.append(
            {"step": self.global_step, "epoch": self.current_epoch, **metrics}
        )

    def _log_eval_metrics(self, metrics: dict[str, float]):
        """Log evaluation metrics."""
        logger.info(f"Eval | Loss: {metrics['loss']:.4f} | PPL: {metrics['perplexity']:.2f}")

    def _should_stop_early(self, metrics: dict[str, float]) -> bool:
        """Check if we should stop due to reaching target loss."""
        if self.sft_config.min_loss is not None:
            if metrics.get("loss", float("inf")) < self.sft_config.min_loss:
                logger.info(f"Reached target loss: {metrics['loss']:.4f}")
                return True
        return False
