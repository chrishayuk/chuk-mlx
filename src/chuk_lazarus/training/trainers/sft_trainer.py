"""
SFT Trainer - Supervised Fine-Tuning for Tool Use.

This trainer handles the initial supervised training phase:
- Teach the model tool-calling syntax
- Learn structured output formats
- Build foundation for RL fine-tuning

Stage 1 of the Lazarus training pipeline:
    SFT (learn syntax) -> DPO (learn preferences) -> Deploy

Usage:
    trainer = SFTTrainer(
        model=hf_model,
        tokenizer=tokenizer,
        config=config
    )
    trainer.train(dataset)
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ...data import SFTDataset
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.sft_loss import sft_loss

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(BaseTrainerConfig):
    """Configuration for SFT training."""

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
        config = SFTConfig(
            batch_size=4,
            learning_rate=1e-5,
            num_epochs=3
        )
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

    @property
    def sft_config(self) -> SFTConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute SFT loss for a batch."""
        logits, _ = self.model(batch["input_ids"])
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
            logits, _ = self.model(batch["input_ids"])
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
