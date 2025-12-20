"""
DPO Trainer - Direct Preference Optimization training loop.

This trainer integrates with your existing chuk-mlx training infrastructure
while adding DPO-specific functionality.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..losses.dpo_loss import dpo_loss, DPOConfig
from ...data import PreferenceDataset

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainerConfig:
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
    max_steps: Optional[int] = None
    target_reward_margin: float = 2.0  # Stop if margin exceeds this


class DPOTrainer:
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
        optimizer: optim.Optimizer = None
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config or DPOTrainerConfig()

        # Freeze reference model
        self.reference_model.freeze()

        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Training state
        self.global_step = 0
        self.best_reward_margin = float('-inf')

        # Metrics history
        self.metrics_history: list[Dict] = []

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: PreferenceDataset = None,
        callback: Callable[[Dict], None] = None
    ):
        """
        Run DPO training.

        Args:
            train_dataset: Training preference pairs
            eval_dataset: Optional evaluation dataset
            callback: Optional callback called after each log interval
        """
        logger.info(f"Starting DPO training with {len(train_dataset)} preference pairs")
        logger.info(f"Config: {self.config}")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Get pad token id
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)

        # Create value_and_grad function
        def loss_fn(batch):
            loss, metrics = dpo_loss(
                policy_model=self.policy_model,
                reference_model=self.reference_model,
                chosen_input_ids=batch["chosen_input_ids"],
                rejected_input_ids=batch["rejected_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                config=self.config.dpo
            )
            return loss, metrics

        loss_and_grad_fn = nn.value_and_grad(self.policy_model, loss_fn)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_metrics = {
                "loss": [],
                "chosen_reward": [],
                "rejected_reward": [],
                "reward_margin": [],
                "accuracy": [],
            }

            for batch in train_dataset.iter_batches(
                batch_size=self.config.batch_size,
                shuffle=True,
                pad_token_id=pad_token_id
            ):
                self.global_step += 1

                # Forward + backward
                (loss, metrics), grads = loss_and_grad_fn(batch)

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grads = self._clip_gradients(grads, self.config.max_grad_norm)

                # Update weights
                self.optimizer.update(self.policy_model, grads)

                # Evaluate to materialize
                mx.eval(self.policy_model.parameters())

                # Track metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(float(metrics[key]))

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = {k: sum(v[-self.config.log_interval:]) / len(v[-self.config.log_interval:])
                                   for k, v in epoch_metrics.items() if v}

                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_metrics.get('loss', 0):.4f} | "
                        f"Margin: {avg_metrics.get('reward_margin', 0):.4f} | "
                        f"Acc: {avg_metrics.get('accuracy', 0):.2%} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    self.metrics_history.append({
                        "step": self.global_step,
                        **avg_metrics
                    })

                    if callback:
                        callback(avg_metrics)

                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    logger.info(f"Eval | Margin: {eval_metrics['reward_margin']:.4f} | Acc: {eval_metrics['accuracy']:.2%}")

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                # Early stopping
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps ({self.config.max_steps})")
                    break

                # Target margin reached
                current_margin = avg_metrics.get('reward_margin', 0) if epoch_metrics['reward_margin'] else 0
                if current_margin >= self.config.target_reward_margin:
                    logger.info(f"Target reward margin reached: {current_margin:.4f}")
                    break

            # End of epoch
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate on a dataset."""
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)

        all_metrics = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

        for batch in dataset.iter_batches(
            batch_size=self.config.batch_size,
            shuffle=False,
            pad_token_id=pad_token_id
        ):
            loss, metrics = dpo_loss(
                policy_model=self.policy_model,
                reference_model=self.reference_model,
                chosen_input_ids=batch["chosen_input_ids"],
                rejected_input_ids=batch["rejected_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                config=self.config.dpo
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f"{name}.npz"
        weights = dict(self.policy_model.parameters())
        mx.save(str(path), weights)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        weights = mx.load(path)
        self.policy_model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")

    def _clip_gradients(self, grads: Dict, max_norm: float) -> Dict:
        """Clip gradients by global norm."""
        # Compute global norm
        total_norm_sq = 0.0
        for g in grads.values():
            if isinstance(g, mx.array):
                total_norm_sq += mx.sum(g ** 2)
            elif isinstance(g, dict):
                for v in g.values():
                    if isinstance(v, mx.array):
                        total_norm_sq += mx.sum(v ** 2)

        total_norm = mx.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = mx.minimum(clip_coef, mx.array(1.0))

        # Apply clipping
        clipped = {}
        for k, g in grads.items():
            if isinstance(g, mx.array):
                clipped[k] = g * clip_coef
            elif isinstance(g, dict):
                clipped[k] = {kk: vv * clip_coef for kk, vv in g.items()}
            else:
                clipped[k] = g

        return clipped
