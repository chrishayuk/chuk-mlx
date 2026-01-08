"""
Dual-Reward Trainer for Classifier Emergence

Trains V/O projections to create vocabulary-aligned classifiers at intermediate
layers while maintaining answer quality at the output.

This implements the two-phase training for classifier emergence:
- Phase 1: Dual-reward training creates vocab-aligned classifiers
- Phase 2: Freeze classifier layers, train routing layers

Usage:
    config = DualRewardTrainerConfig(
        classifier_layer=12,
        classifier_weight=0.4,
        lora_targets=["v_proj", "o_proj"],
    )
    trainer = DualRewardTrainer(model, tokenizer, config)
    trainer.train(dataset)
"""

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ...models_v2.adapters.lora import LoRAConfig, LoRALinear, apply_lora, count_lora_parameters
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.dual_reward_loss import DualRewardLossConfig, dual_reward_loss

logger = logging.getLogger(__name__)


@dataclass
class DualRewardTrainerConfig(BaseTrainerConfig):
    """Configuration for dual-reward training."""

    # Training settings
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    max_steps: int = 500

    # Classifier settings
    classifier_layer: int = -1  # -1 means 55% depth
    classifier_weight: float = 0.4
    classifier_targets: dict[str, str] = field(default_factory=lambda: {
        "multiply": "multiply",
        "add": "add",
        "subtract": "subtract",
        "divide": "divide",
    })

    # LoRA settings
    lora_rank: int = 16
    lora_targets: list[str] = field(default_factory=lambda: ["v_proj", "o_proj"])

    # Frozen layers (for Phase 2)
    freeze_layers: list[int] = field(default_factory=list)

    # Logging
    log_interval: int = 50
    checkpoint_interval: int = 100
    checkpoint_dir: str = "./checkpoints/dual_reward"


class DualRewardTrainer(BaseTrainer):
    """
    Trainer for dual-reward V/O training.

    Creates vocabulary-aligned classifiers by:
    1. Applying LoRA to V/O projections only
    2. Computing classification loss at intermediate layer
    3. Computing answer loss at final layer
    4. Optimizing combined loss
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: DualRewardTrainerConfig,
        model_config: Any = None,
    ):
        # Don't call super().__init__ yet - we need to set up LoRA first
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = model_config

        # Determine number of layers
        self.num_layers = self._get_num_layers()

        # Set classifier layer
        if config.classifier_layer < 0:
            self.classifier_layer = int(self.num_layers * 0.55)
        else:
            self.classifier_layer = config.classifier_layer

        logger.info(f"Classifier layer: L{self.classifier_layer} / {self.num_layers}")

        # Set up classifier token mapping
        self.classifier_token_ids = {}
        for label, token_str in config.classifier_targets.items():
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                self.classifier_token_ids[label] = token_ids[0]
        logger.info(f"Classifier tokens: {self.classifier_token_ids}")

        # Set up LoRA using centralized apply_lora
        self.lora_config = LoRAConfig(
            rank=config.lora_rank,
            alpha=config.lora_rank * 2.0,  # Standard scaling: alpha = 2 * rank
            dropout=0.0,
            target_modules=config.lora_targets,
        )
        self.lora_layers = apply_lora(model, self.lora_config)
        trainable_params = count_lora_parameters(self.lora_layers)
        logger.info(f"LoRA layers: {len(self.lora_layers)}, trainable params: {trainable_params:,}")

        # Set up optimizer (only for LoRA params)
        self.optimizer = self._create_lora_optimizer()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.metrics_history = []
        self._start_time = None

        # Get embedding weights for logit computation
        self._setup_embeddings()

    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            return len(self.model.layers)
        else:
            raise ValueError("Cannot determine number of layers")

    def _get_layers(self):
        """Get transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        else:
            raise ValueError("Cannot access layers")

    def _setup_embeddings(self):
        """Set up embedding weights for logit computation."""
        if hasattr(self.model, "model"):
            self.embed_tokens = self.model.model.embed_tokens
            self.norm = getattr(self.model.model, "norm", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        else:
            self.embed_tokens = self.model.embed_tokens
            self.norm = getattr(self.model, "norm", None)
            self.lm_head = getattr(self.model, "lm_head", None)

    def _create_lora_optimizer(self) -> optim.Optimizer:
        """Create optimizer for LoRA parameters."""
        return optim.Adam(learning_rate=self.config.learning_rate)

    def _get_lora_params(self) -> list[mx.array]:
        """Get flat list of LoRA parameters from LoRALinear layers."""
        params = []
        for name in sorted(self.lora_layers.keys()):
            lora_layer = self.lora_layers[name]
            params.append(lora_layer.lora_A)
            params.append(lora_layer.lora_B)
        return params

    def _set_lora_params(self, params: list[mx.array]):
        """Set LoRA parameters in LoRALinear layers from flat list."""
        idx = 0
        for name in sorted(self.lora_layers.keys()):
            lora_layer = self.lora_layers[name]
            lora_layer.lora_A = params[idx]
            lora_layer.lora_B = params[idx + 1]
            idx += 2

    def _forward_with_intermediate(
        self, input_ids: mx.array
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass capturing both final and intermediate logits.

        Returns:
            final_logits: Logits from final layer
            classifier_logits: Logits from classifier layer
        """
        # Embed
        if hasattr(self.model, "model"):
            h = self.model.model.embed_tokens(input_ids)
        else:
            h = self.embed_tokens(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Forward through layers
        layers = self._get_layers()
        classifier_h = None

        for layer_idx, layer in enumerate(layers):
            h = layer(h, mask=mask)
            # Handle different return types
            if isinstance(h, tuple):
                h = h[0]
            elif hasattr(h, "hidden_states"):
                h = h.hidden_states

            if layer_idx == self.classifier_layer:
                classifier_h = h

        # Apply final norm
        if self.norm is not None:
            h = self.norm(h)
            classifier_h_normed = self.norm(classifier_h)
        else:
            classifier_h_normed = classifier_h

        # Compute logits
        if self.lm_head is not None:
            final_logits = self.lm_head(h)
            classifier_logits = self.lm_head(classifier_h_normed)
            # Handle HeadOutput or similar wrapper objects
            if hasattr(final_logits, "logits"):
                final_logits = final_logits.logits
            if hasattr(classifier_logits, "logits"):
                classifier_logits = classifier_logits.logits
        else:
            # Use embedding weights
            embed_weight = self.embed_tokens.weight
            if hasattr(embed_weight, "weight"):
                embed_weight = embed_weight.weight
            final_logits = h @ embed_weight.T
            classifier_logits = classifier_h_normed @ embed_weight.T

        return final_logits, classifier_logits

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute dual-reward loss.

        Note: LoRALinear handles weight adaptation automatically during forward pass.
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]
        classifier_labels = batch["classifier_labels"]

        final_logits, classifier_logits = self._forward_with_intermediate(input_ids)

        loss_config = DualRewardLossConfig(
            classifier_layer=self.classifier_layer,
            classifier_weight=self.config.classifier_weight,
        )

        loss, metrics = dual_reward_loss(
            final_logits=final_logits,
            classifier_logits=classifier_logits,
            labels=labels,
            classifier_labels=classifier_labels,
            loss_mask=loss_mask,
            config=loss_config,
        )

        return loss, metrics

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Get training batches from dataset."""
        # Dataset should yield dicts with: prompt, response, operation
        import random

        samples = list(dataset)
        random.shuffle(samples)

        for sample in samples:
            # Tokenize
            prompt = sample["prompt"]
            response = sample["response"]
            operation = sample.get("operation") or sample.get("classification_target")

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

            # Loss mask (only on response)
            loss_mask = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)

            # Classifier label
            classifier_label = self.classifier_token_ids.get(operation, 0)

            yield {
                "input_ids": mx.array([input_ids]),
                "labels": mx.array([labels]),
                "loss_mask": mx.array([loss_mask]),
                "classifier_labels": mx.array([classifier_label]),
            }

    def train(self, dataset: Any):
        """Run dual-reward training."""
        logger.info(f"Starting dual-reward training")
        logger.info(f"Classifier layer: L{self.classifier_layer}")
        logger.info(f"Classifier weight: {self.config.classifier_weight}")
        logger.info(f"LoRA targets: {self.config.lora_targets}")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()

        # Create loss function for gradients
        # LoRALinear handles weight adaptation automatically during forward pass
        def loss_fn(params, batch):
            self._set_lora_params(params)

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]
            classifier_labels = batch["classifier_labels"]

            final_logits, classifier_logits = self._forward_with_intermediate(input_ids)

            loss_config = DualRewardLossConfig(
                classifier_layer=self.classifier_layer,
                classifier_weight=self.config.classifier_weight,
            )

            loss, _ = dual_reward_loss(
                final_logits=final_logits,
                classifier_logits=classifier_logits,
                labels=labels,
                classifier_labels=classifier_labels,
                loss_mask=loss_mask,
                config=loss_config,
            )
            return loss

        grad_fn = mx.value_and_grad(loss_fn)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            for batch in self.get_train_batches(dataset):
                self.global_step += 1

                if self.config.max_steps and self.global_step > self.config.max_steps:
                    break

                # Compute loss and gradients
                params = self._get_lora_params()
                loss, grads = grad_fn(params, batch)

                # Update
                new_params = [
                    p - self.config.learning_rate * g
                    for p, g in zip(params, grads)
                ]
                self._set_lora_params(new_params)

                # Evaluate LoRA parameters
                mx.eval([layer.lora_A for layer in self.lora_layers.values()] +
                        [layer.lora_B for layer in self.lora_layers.values()])

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    # Compute full metrics
                    _, metrics = self.compute_loss(batch)

                    elapsed = time.time() - self._start_time
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {float(metrics['loss']):.4f} | "
                        f"Cls: {float(metrics['classifier_loss']):.4f} | "
                        f"Ans: {float(metrics['answer_loss']):.4f} | "
                        f"Cls Acc: {float(metrics['classifier_accuracy']):.1%} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    self.metrics_history.append({
                        "step": self.global_step,
                        **{k: float(v) for k, v in metrics.items()}
                    })

                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Steps: {self.global_step}")

    def save_checkpoint(self, name: str):
        """Save LoRA checkpoint using safetensors format."""
        from ...models_v2.loader import save_adapter

        checkpoint_path = Path(self.config.checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters using centralized save_adapter
        save_adapter(self.lora_layers, checkpoint_path, lora_config=self.lora_config)

        # Save additional dual-reward config
        config_out = {
            "classifier_layer": self.classifier_layer,
            "classifier_weight": self.config.classifier_weight,
            "classifier_token_ids": self.classifier_token_ids,
            "global_step": self.global_step,
        }
        with open(checkpoint_path / "dual_reward_config.json", "w") as f:
            json.dump(config_out, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def evaluate_classifier(self, test_prompts: list[tuple[str, str]]) -> dict:
        """
        Evaluate classifier accuracy on test prompts.

        Args:
            test_prompts: List of (prompt, expected_operation) tuples

        Returns:
            Dict with accuracy and per-class results

        Note: LoRALinear handles weight adaptation automatically during forward pass.
        """
        correct = 0
        results = []

        for prompt, expected in test_prompts:
            input_ids = mx.array([self.tokenizer.encode(prompt)])
            _, classifier_logits = self._forward_with_intermediate(input_ids)

            # Get prediction at last token
            cls_logits = classifier_logits[0, -1, :]
            probs = mx.softmax(cls_logits)

            # Find best class
            best_class = None
            best_prob = 0
            for class_name, token_id in self.classifier_token_ids.items():
                prob = float(probs[token_id].item())
                if prob > best_prob:
                    best_prob = prob
                    best_class = class_name

            is_correct = best_class == expected
            if is_correct:
                correct += 1

            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": best_class,
                "confidence": best_prob,
                "correct": is_correct,
            })

        accuracy = correct / len(test_prompts) if test_prompts else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_prompts),
            "results": results,
        }
