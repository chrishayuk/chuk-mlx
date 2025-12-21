"""
GRPO Trainer - Group Relative Policy Optimization for LLMs.

GRPO is particularly well-suited for training Mistral as a tool-using agent
because:
- Generates multiple responses per prompt (natural for LLMs)
- No value function needed (simpler than PPO for text)
- Group-relative advantages work well with sparse rewards
- Can integrate MCP tools during sample generation
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

from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.grpo_loss import GRPOBatch, GRPOConfig, grpo_loss
from ..utils.log_probs import compute_sequence_log_prob, extract_log_probs

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainerConfig(BaseTrainerConfig):
    """Configuration for GRPO training."""

    # GRPO hyperparameters
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Training settings
    num_iterations: int = 1000
    prompts_per_iteration: int = 16
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Generation settings
    max_response_length: int = 256
    temperature: float = 1.0

    # Logging and checkpoints
    log_interval: int = 1
    checkpoint_interval: int = 50
    checkpoint_dir: str = "./checkpoints/grpo"

    # Early stopping
    max_steps: int | None = None
    target_reward: float | None = None


class GRPOTrainer(BaseTrainer):
    """
    Trainer for Group Relative Policy Optimization.

    Designed for training Mistral-7B (or similar) with tool use.

    The training loop:
    1. Sample prompts
    2. Generate group_size responses per prompt
    3. Compute rewards for each response (via MCP tools)
    4. Update policy using group-relative advantages

    Usage:
        trainer = GRPOTrainer(
            policy_model=mistral,
            reference_model=ref_mistral,
            tokenizer=tokenizer,
            reward_fn=compute_reward,
            config=config
        )
        trainer.train(prompt_dataset)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        reward_fn: Callable[[str, str], float],
        config: GRPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        config = config or GRPOTrainerConfig()
        super().__init__(policy_model, tokenizer, config, optimizer)

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_fn = reward_fn

        # Freeze reference model
        self.reference_model.freeze()

        # GRPO-specific state
        self.iteration = 0
        self.best_reward = float("-inf")

    @property
    def grpo_config(self) -> GRPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """
        Compute GRPO loss. Not used directly - GRPO has custom training loop.
        """
        raise NotImplementedError("GRPO uses custom training loop via train()")

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Not used - GRPO generates samples on the fly."""
        raise NotImplementedError("GRPO generates samples on the fly")

    def train(self, prompt_source: Callable[[], list[str]]):
        """
        Run GRPO training.

        Args:
            prompt_source: Function that returns a batch of prompts
        """
        logger.info(f"Starting GRPO training for {self.grpo_config.num_iterations} iterations")
        logger.info(f"Group size: {self.grpo_config.grpo.group_size}")
        logger.info(f"Prompts per iteration: {self.grpo_config.prompts_per_iteration}")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()

        for self.iteration in range(1, self.grpo_config.num_iterations + 1):
            # Get prompts
            prompts = prompt_source()[: self.grpo_config.prompts_per_iteration]

            # Generate samples and compute rewards
            batch = self._generate_grpo_batch(prompts)

            # Compute GRPO update
            metrics = self._grpo_update(batch)

            # Logging
            if self.iteration % self.config.log_interval == 0:
                elapsed = time.time() - self._start_time
                logger.info(
                    f"Iter {self.iteration} | "
                    f"Mean Reward: {metrics['mean_reward']:.4f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"KL: {metrics['kl_penalty']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

                self.metrics_history.append({"iteration": self.iteration, **metrics})

            # Checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"iter_{self.iteration}")

            # Early stopping
            if self.grpo_config.target_reward is not None:
                if metrics["mean_reward"] >= self.grpo_config.target_reward:
                    logger.info(f"Target reward reached: {metrics['mean_reward']:.4f}")
                    break

            # Track best
            if metrics["mean_reward"] > self.best_reward:
                self.best_reward = metrics["mean_reward"]
                self.save_checkpoint("best")

        self.save_checkpoint("final")
        logger.info(f"Training complete. Iterations: {self.iteration}")

    def _generate_grpo_batch(self, prompts: list[str]) -> GRPOBatch:
        """Generate responses and compute rewards for a batch of prompts."""
        batch = GRPOBatch(self.grpo_config.grpo.group_size)

        for prompt in prompts:
            responses = []
            rewards = []

            # Generate group_size responses
            for _ in range(self.grpo_config.grpo.group_size):
                response = self._generate_response(prompt)
                responses.append(response)

                # Compute reward (this is where MCP tools get called)
                reward = self.reward_fn(prompt, response)
                rewards.append(reward)

            batch.add_prompt_group(prompt, responses, rewards)

        return batch

    def _generate_response(self, prompt: str) -> str:
        """Generate a single response from the policy model."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        generated = list(input_ids)

        # Set model to inference mode if applicable
        if hasattr(self.policy_model, "set_mode"):
            self.policy_model.set_mode("INFERENCE")

        max_new_tokens = self.grpo_config.max_response_length

        for _ in range(max_new_tokens):
            input_tensor = mx.array([generated])

            with mx.stop_gradient():
                logits, _ = self.policy_model(input_tensor)

            # Get logits for last position
            next_logits = logits[0, -1, :] / self.grpo_config.temperature

            # Sample
            probs = mx.softmax(next_logits)
            next_token = self._sample_token(probs)

            generated.append(int(next_token))

            # Check for EOS
            if hasattr(self.tokenizer, "eos_token_id"):
                if int(next_token) == self.tokenizer.eos_token_id:
                    break

        # Decode response (exclude prompt)
        response_ids = generated[len(input_ids) :]
        response = self.tokenizer.decode(response_ids)

        return response

    def _grpo_update(self, batch: GRPOBatch) -> dict[str, float]:
        """Perform GRPO update on the batch."""
        # Get all sequences
        sequences = batch.get_all_sequences()
        rewards = batch.get_flat_rewards()

        # Tokenize all sequences
        all_input_ids = []
        all_masks = []
        max_len = 0

        for seq in sequences:
            tokens = self.tokenizer.encode(seq)
            all_input_ids.append(tokens)
            max_len = max(max_len, len(tokens))

        # Pad sequences
        pad_id = self.pad_token_id
        for i, tokens in enumerate(all_input_ids):
            padding = [pad_id] * (max_len - len(tokens))
            mask = [1.0] * len(tokens) + [0.0] * len(padding)
            all_input_ids[i] = tokens + padding
            all_masks.append(mask)

        input_ids = mx.array(all_input_ids)
        attention_mask = mx.array(all_masks)

        # Get log probs from policy
        policy_log_probs, _ = extract_log_probs(self.policy_model, input_ids, attention_mask)

        # Get log probs from reference
        with mx.stop_gradient():
            ref_log_probs, _ = extract_log_probs(self.reference_model, input_ids, attention_mask)

        # Sum to sequence level
        mask_shifted = attention_mask[:, 1:]
        policy_seq_log_probs = compute_sequence_log_prob(policy_log_probs, mask_shifted)
        ref_seq_log_probs = compute_sequence_log_prob(ref_log_probs, mask_shifted)

        # Compute loss
        def loss_fn():
            p_log_probs, _ = extract_log_probs(self.policy_model, input_ids, attention_mask)
            p_seq = compute_sequence_log_prob(p_log_probs, mask_shifted)

            loss, metrics = grpo_loss(
                log_probs=p_seq,
                ref_log_probs=ref_seq_log_probs,
                rewards=rewards,
                group_size=self.grpo_config.grpo.group_size,
                config=self.grpo_config.grpo,
            )
            return loss

        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            grads = self.clip_gradients(grads, self.config.max_grad_norm)

        # Update
        self.optimizer.update(self.policy_model, grads)
        mx.eval(self.policy_model.parameters())

        # Compute metrics
        _, metrics = grpo_loss(
            log_probs=policy_seq_log_probs,
            ref_log_probs=ref_seq_log_probs,
            rewards=rewards,
            group_size=self.grpo_config.grpo.group_size,
            config=self.grpo_config.grpo,
        )

        return {k: float(v) for k, v in metrics.items()}

    def _sample_token(self, probs: mx.array) -> mx.array:
        """Sample from probability distribution."""
        u = mx.random.uniform(probs.shape)
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
        return mx.argmax(mx.log(probs + 1e-10) + gumbel)

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
