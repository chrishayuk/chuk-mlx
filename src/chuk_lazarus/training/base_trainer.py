"""
Base Trainer - Abstract base class for all trainers.

This module provides a unified interface for training, reducing code duplication
across SFT, DPO, GRPO, and PPO trainers.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainerConfig:
    """Base configuration shared by all trainers."""

    # Training settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Logging and checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints"

    # Early stopping
    max_steps: int | None = None


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Provides common functionality:
    - Optimizer setup
    - Gradient clipping
    - Checkpoint saving/loading
    - Metrics tracking
    - Training loop structure

    Subclasses must implement:
    - compute_loss(): Compute loss and metrics for a batch
    - get_train_batches(): Get iterator over training batches
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: BaseTrainerConfig,
        optimizer: optim.Optimizer | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")

        # Metrics history
        self.metrics_history: list[dict] = []

        # Timing
        self._start_time: float | None = None

    def _create_optimizer(self) -> optim.Optimizer:
        """Create default AdamW optimizer."""
        return optim.AdamW(
            learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """
        Compute loss and metrics for a batch.

        Args:
            batch: Dictionary of batch data

        Returns:
            Tuple of (loss, metrics_dict)
        """
        pass

    @abstractmethod
    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """
        Get iterator over training batches.

        Args:
            dataset: Training dataset

        Returns:
            Iterator yielding batch dictionaries
        """
        pass

    def train(
        self,
        dataset: Any,
        num_epochs: int = 1,
        eval_dataset: Any = None,
        callback: Callable[[dict], None] | None = None,
    ):
        """
        Main training loop.

        Args:
            dataset: Training dataset
            num_epochs: Number of epochs to train
            eval_dataset: Optional evaluation dataset
            callback: Optional callback after each log interval
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Config: {self.config}")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)

        self._start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_metrics = self._create_epoch_metrics()

            avg_metrics: dict[str, float] = {}

            for batch in self.get_train_batches(dataset):
                self.global_step += 1

                # Forward + backward
                (loss, metrics), grads = loss_and_grad_fn(batch)

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grads = self.clip_gradients(grads, self.config.max_grad_norm)

                # Update weights
                self.optimizer.update(self.model, grads)

                # Evaluate to materialize
                mx.eval(self.model.parameters())

                # Track metrics
                self._accumulate_metrics(epoch_metrics, metrics)

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = self._compute_avg_metrics(epoch_metrics)
                    self._log_metrics(avg_metrics)

                    if callback:
                        callback(avg_metrics)

                # Evaluation
                if eval_dataset and hasattr(self.config, "eval_interval"):
                    if self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate(eval_dataset)
                        self._log_eval_metrics(eval_metrics)

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    if avg_metrics:
                        self._save_checkpoint_if_best(avg_metrics)
                    self.save_checkpoint(f"step_{self.global_step}")

                # Early stopping - max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps ({self.config.max_steps})")
                    break

                # Custom early stopping check
                if avg_metrics and self._should_stop_early(avg_metrics):
                    break

            # End of epoch
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """
        Evaluate on a dataset.

        Override in subclass for custom evaluation logic.

        Args:
            dataset: Evaluation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        all_metrics: dict[str, list[float]] = {}

        for batch in self.get_train_batches(dataset):
            _, metrics = self.compute_loss(batch)

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(float(value))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f"{name}.safetensors"

        # Check if model has adapter save method (for LoRA)
        if hasattr(self.model, "save_adapter"):
            self.model.save_adapter(str(path))
        else:
            weights = dict(self.model.parameters())
            flat_weights = self._flatten_params(weights)
            mx.save_safetensors(str(path), flat_weights)

        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        weights = mx.load(path)
        self.model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")

    def clip_gradients(self, grads: Any, max_norm: float) -> Any:
        """Clip gradients by global norm."""
        flat_grads = []

        def collect(g):
            if isinstance(g, dict):
                for v in g.values():
                    collect(v)
            elif isinstance(g, mx.array):
                flat_grads.append(g.reshape(-1))

        collect(grads)

        if not flat_grads:
            return grads

        all_grads = mx.concatenate(flat_grads)
        global_norm = mx.sqrt(mx.sum(all_grads**2))
        clip_factor = mx.minimum(max_norm / (global_norm + 1e-6), mx.array(1.0))

        def apply_clip(g):
            if isinstance(g, dict):
                return {k: apply_clip(v) for k, v in g.items()}
            elif isinstance(g, mx.array):
                return g * clip_factor
            return g

        return apply_clip(grads)

    def _flatten_params(self, params: dict, prefix: str = "") -> dict[str, mx.array]:
        """Flatten nested parameter dict for saving."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_params(v, key))
            elif isinstance(v, mx.array):
                flat[key] = v
        return flat

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Create empty metrics accumulator for epoch. Override for custom metrics."""
        return {"loss": []}

    def _accumulate_metrics(self, epoch_metrics: dict[str, list], metrics: dict[str, Any]):
        """Accumulate batch metrics into epoch metrics."""
        for key in epoch_metrics:
            if key in metrics:
                epoch_metrics[key].append(float(metrics[key]))

    def _compute_avg_metrics(self, epoch_metrics: dict[str, list]) -> dict[str, float]:
        """Compute average of recent metrics."""
        interval = self.config.log_interval
        return {k: sum(v[-interval:]) / len(v[-interval:]) for k, v in epoch_metrics.items() if v}

    def _log_metrics(self, metrics: dict[str, float]):
        """Log training metrics. Override for custom logging."""
        elapsed = time.time() - self._start_time
        loss = metrics.get("loss", 0)
        logger.info(f"Step {self.global_step} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")

        self.metrics_history.append(
            {"step": self.global_step, "epoch": self.current_epoch, **metrics}
        )

    def _log_eval_metrics(self, metrics: dict[str, float]):
        """Log evaluation metrics."""
        loss = metrics.get("loss", 0)
        logger.info(f"Eval | Loss: {loss:.4f}")

    def _save_checkpoint_if_best(self, metrics: dict[str, float]):
        """Save best checkpoint if current metric is better."""
        current_loss = metrics.get("loss", float("inf"))
        if current_loss < self.best_metric:
            self.best_metric = current_loss
            self.save_checkpoint("best")

    def _should_stop_early(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop early. Override for custom logic."""
        return False

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID from tokenizer."""
        return getattr(self.tokenizer, "pad_token_id", 0) or 0
