"""
DPO Trainer - Direct Preference Optimization training loop.

This trainer integrates with your existing chuk-mlx training infrastructure
while adding DPO-specific functionality.

Usage:
    # High-level API (recommended for CLI):
    result = DPOTrainer.run(DPOTrainingConfig(
        model="meta-llama/Llama-3.2-1B",
        data_path="preferences.jsonl",
        output_dir="./output",
    ))

    # Low-level API (for custom pipelines):
    trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
    trainer.train(dataset)
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pydantic import BaseModel, Field

from ...data import PreferenceDataset
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.dpo_loss import DPOConfig, dpo_loss

logger = logging.getLogger(__name__)


class DPOTrainingConfig(BaseModel):
    """Complete configuration for running DPO training.

    This is the high-level config used by CLI and run() method.
    Includes model paths, data paths, and all training parameters.
    """

    # Model
    model: str = Field(..., description="Policy model path or HuggingFace name")
    ref_model: str | None = Field(default=None, description="Reference model (defaults to policy)")
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(default=16.0, description="LoRA alpha scaling")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="LoRA target modules",
    )

    # Data
    data_path: Path = Field(..., description="Path to preference data (JSONL)")
    eval_data_path: Path | None = Field(default=None, description="Path to eval data")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")

    # Training
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    beta: float = Field(default=0.1, gt=0, description="DPO beta parameter")
    max_steps: int | None = Field(default=None, description="Max steps (overrides epochs)")

    # Output
    output_dir: Path = Field(default=Path("./checkpoints/dpo"), description="Output directory")
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")

    @property
    def reference_model(self) -> str:
        """Get reference model name (defaults to policy model)."""
        return self.ref_model or self.model


class DPOTrainingResult(BaseModel):
    """Result of DPO training."""

    output_dir: Path = Field(..., description="Output directory")
    epochs_completed: int = Field(..., description="Epochs completed")
    final_loss: float | None = Field(default=None, description="Final training loss")
    adapter_path: Path | None = Field(default=None, description="Path to saved LoRA adapter")


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
        # High-level API (recommended):
        result = DPOTrainer.run(DPOTrainingConfig(
            model="meta-llama/Llama-3.2-1B",
            data_path="preferences.jsonl",
        ))

        # Low-level API:
        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
        trainer.train(train_dataset)
    """

    @classmethod
    def run(cls, config: DPOTrainingConfig) -> DPOTrainingResult:
        """Run complete DPO training from config.

        This is the high-level entry point that handles:
        - Model loading (policy and reference, with optional LoRA)
        - Dataset loading
        - Training
        - Checkpoint saving

        Args:
            config: Complete training configuration

        Returns:
            DPOTrainingResult with training outcomes
        """
        from ...models_v2 import (
            LoRAConfig,
            load_model,
            load_model_with_lora,
            save_adapter,
        )

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load policy model
        logger.info(f"Loading policy model: {config.model}")
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
            policy_model = result.model
            tokenizer = result.tokenizer
            lora_layers = result.lora_layers
            logger.info(
                f"  Loaded with LoRA: {len(lora_layers)} layers, "
                f"{result.lora_parameter_count:,} trainable params"
            )
        else:
            result = load_model(config.model)
            policy_model = result.model
            tokenizer = result.tokenizer

        # Load reference model (never with LoRA - frozen)
        logger.info(f"Loading reference model: {config.reference_model}")
        ref_result = load_model(config.reference_model)
        ref_model = ref_result.model

        # Load datasets
        logger.info(f"Loading dataset: {config.data_path}")
        train_dataset = PreferenceDataset(
            str(config.data_path),
            tokenizer,
            max_length=config.max_length,
        )
        logger.info(f"  Loaded {len(train_dataset)} preference pairs")

        eval_dataset = None
        if config.eval_data_path:
            eval_dataset = PreferenceDataset(
                str(config.eval_data_path),
                tokenizer,
                max_length=config.max_length,
            )
            logger.info(f"  Loaded {len(eval_dataset)} eval pairs")

        # Create trainer config
        trainer_config = DPOTrainerConfig(
            dpo=DPOConfig(beta=config.beta),
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            checkpoint_dir=str(output_dir / "checkpoints"),
            log_interval=config.log_interval,
            max_steps=config.max_steps,
            checkpoint_interval=config.checkpoint_interval,
        )

        # Create and run trainer
        trainer = cls(policy_model, ref_model, tokenizer, trainer_config)

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

        return DPOTrainingResult(
            output_dir=output_dir,
            epochs_completed=config.num_epochs,
            final_loss=final_loss,
            adapter_path=adapter_path,
        )

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
            batch_size=self.dpo_config.batch_size,
            shuffle=True,
            pad_token_id=self.pad_token_id,
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
            batch_size=self.dpo_config.batch_size,
            shuffle=False,
            pad_token_id=self.pad_token_id,
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
        """Save model checkpoint in safetensors format."""
        from pathlib import Path

        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        weights_path = checkpoint_path / f"{name}.safetensors"
        weights = dict(self.policy_model.parameters())
        mx.save_safetensors(str(weights_path), weights)
        logger.info(f"Saved checkpoint: {weights_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint from safetensors format."""
        weights = mx.load(path)
        self.policy_model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
