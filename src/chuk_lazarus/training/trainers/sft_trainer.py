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
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..losses.sft_loss import SFTConfig, sft_loss
from ...data import SFTDataset

logger = logging.getLogger(__name__)


class SFTTrainer:
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
        optimizer: optim.Optimizer = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()

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
        self.best_loss = float('inf')

        # Metrics history
        self.metrics_history: List[Dict] = []

    def train(
        self,
        train_dataset: SFTDataset,
        eval_dataset: SFTDataset = None,
        callback: Callable[[Dict], None] = None
    ):
        """
        Run SFT training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callback: Optional callback after each log interval
        """
        logger.info(f"Starting SFT training with {len(train_dataset)} samples")
        logger.info(f"Config: {self.config}")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Get pad token id
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0

        # Create loss function for value_and_grad
        def loss_fn(batch):
            logits, _ = self.model(batch["input_ids"])
            loss, metrics = sft_loss(
                logits=logits,
                labels=batch["labels"],
                loss_mask=batch["loss_mask"]
            )
            return loss, metrics

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            epoch_metrics = {
                "loss": [],
                "perplexity": [],
                "num_tokens": [],
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
                self.optimizer.update(self.model, grads)

                # Evaluate to materialize
                mx.eval(self.model.parameters())

                # Track metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(float(metrics[key]))

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = {
                        k: sum(v[-self.config.log_interval:]) / len(v[-self.config.log_interval:])
                        for k, v in epoch_metrics.items() if v
                    }

                    elapsed = time.time() - start_time
                    tokens_per_sec = sum(epoch_metrics["num_tokens"]) / elapsed

                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_metrics.get('loss', 0):.4f} | "
                        f"PPL: {avg_metrics.get('perplexity', 0):.2f} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    self.metrics_history.append({
                        "step": self.global_step,
                        "epoch": epoch,
                        **avg_metrics
                    })

                    if callback:
                        callback(avg_metrics)

                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    logger.info(
                        f"Eval | Loss: {eval_metrics['loss']:.4f} | "
                        f"PPL: {eval_metrics['perplexity']:.2f}"
                    )

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    current_loss = avg_metrics.get('loss', float('inf'))
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.save_checkpoint("best")
                    self.save_checkpoint(f"step_{self.global_step}")

                # Early stopping - max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps ({self.config.max_steps})")
                    break

                # Early stopping - min loss
                if self.config.min_loss and avg_metrics.get('loss', float('inf')) < self.config.min_loss:
                    logger.info(f"Reached target loss: {avg_metrics['loss']:.4f}")
                    break

            # End of epoch
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def evaluate(self, dataset: SFTDataset) -> Dict[str, float]:
        """Evaluate on a dataset."""
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0

        all_metrics = {"loss": [], "perplexity": [], "num_tokens": []}

        for batch in dataset.iter_batches(
            batch_size=self.config.batch_size,
            shuffle=False,
            pad_token_id=pad_token_id
        ):
            logits, _ = self.model(batch["input_ids"])
            loss, metrics = sft_loss(
                logits=logits,
                labels=batch["labels"],
                loss_mask=batch["loss_mask"]
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f"{name}.safetensors"

        # Check if model has adapter save method
        if hasattr(self.model, 'save_adapter'):
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

    def _flatten_params(self, params, prefix=""):
        """Flatten nested parameter dict."""
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_params(v, key))
            elif isinstance(v, mx.array):
                flat[key] = v
        return flat

    def _clip_gradients(self, grads, max_norm: float):
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
        global_norm = mx.sqrt(mx.sum(all_grads ** 2))
        clip_factor = mx.minimum(max_norm / (global_norm + 1e-6), mx.array(1.0))

        def apply_clip(g):
            if isinstance(g, dict):
                return {k: apply_clip(v) for k, v in g.items()}
            elif isinstance(g, mx.array):
                return g * clip_factor
            return g

        return apply_clip(grads)
