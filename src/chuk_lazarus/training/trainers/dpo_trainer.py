"""
DPO Trainer - Direct Preference Optimization training loop.

This trainer integrates with your existing chuk-mlx training infrastructure
while adding DPO-specific functionality.
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ...data import PreferenceDataset
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.dpo_loss import DPOConfig, dpo_loss

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainerConfig(BaseTrainerConfig):
    """Configuration for DPO training."""

    # DPO hyperparameters
    dpo: DPOConfig = field(default_factory=DPOConfig)

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Logging and checkpoints
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints/dpo"

    # Early stopping
    max_steps: int | None = None
    target_reward_margin: float = 2.0  # Stop if margin exceeds this


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization.

    Usage:
        trainer = DPOTrainer(
            policy_model=model,
            reference_model=ref_model,
            tokenizer=tokenizer,
            config=config
        )
        trainer.train(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: DPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        config = config or DPOTrainerConfig()
        super().__init__(policy_model, tokenizer, config, optimizer)

        self.policy_model = policy_model
        self.reference_model = reference_model

        # Freeze reference model
        self.reference_model.freeze()

        # DPO-specific state
        self.best_reward_margin = float("-inf")

    @property
    def dpo_config(self) -> DPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute DPO loss for a batch."""
        loss, metrics = dpo_loss(
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            chosen_input_ids=batch["chosen_input_ids"],
            rejected_input_ids=batch["rejected_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            config=self.dpo_config.dpo,
        )
        return loss, metrics

    def get_train_batches(self, dataset: PreferenceDataset) -> Iterator[dict[str, mx.array]]:
        """Get iterator over training batches."""
        return dataset.iter_batches(
            batch_size=self.dpo_config.batch_size, shuffle=True, pad_token_id=self.pad_token_id
        )

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: PreferenceDataset = None,
        callback: Callable[[dict], None] = None,
    ):
        """
        Run DPO training.

        Args:
            train_dataset: Training preference pairs
            eval_dataset: Optional evaluation dataset
            callback: Optional callback called after each log interval
        """
        logger.info(f"Starting DPO training with {len(train_dataset)} preference pairs")
        super().train(
            dataset=train_dataset,
            num_epochs=self.dpo_config.num_epochs,
            eval_dataset=eval_dataset,
            callback=callback,
        )

    def evaluate(self, dataset: PreferenceDataset) -> dict[str, float]:
        """Evaluate on a dataset."""
        all_metrics = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

        for batch in dataset.iter_batches(
            batch_size=self.dpo_config.batch_size, shuffle=False, pad_token_id=self.pad_token_id
        ):
            loss, metrics = dpo_loss(
                policy_model=self.policy_model,
                reference_model=self.reference_model,
                chosen_input_ids=batch["chosen_input_ids"],
                rejected_input_ids=batch["rejected_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                config=self.dpo_config.dpo,
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Create DPO-specific metrics accumulator."""
        return {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

    def _log_metrics(self, metrics: dict[str, float]):
        """Log DPO-specific metrics."""
        elapsed = time.time() - self._start_time
        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {metrics.get('loss', 0):.4f} | "
            f"Margin: {metrics.get('reward_margin', 0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0):.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        self.metrics_history.append({"step": self.global_step, **metrics})

    def _log_eval_metrics(self, metrics: dict[str, float]):
        """Log evaluation metrics."""
        logger.info(
            f"Eval | Margin: {metrics['reward_margin']:.4f} | Acc: {metrics['accuracy']:.2%}"
        )

    def _should_stop_early(self, metrics: dict[str, float]) -> bool:
        """Check if we should stop due to reaching target reward margin."""
        current_margin = metrics.get("reward_margin", 0)
        if current_margin >= self.dpo_config.target_reward_margin:
            logger.info(f"Target reward margin reached: {current_margin:.4f}")
            return True
        return False

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        from pathlib import Path

        path = Path(self.config.checkpoint_dir) / f"{name}.npz"
        weights = dict(self.policy_model.parameters())
        mx.save(str(path), weights)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        weights = mx.load(path)
        self.policy_model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
