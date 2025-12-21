"""
PPO Trainer - Proximal Policy Optimization training loop.

This trainer handles:
- Environment interaction (rollout collection)
- PPO updates with clipped objective
- Support for both Mistral (LLM) and RNN experts
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ...data import RolloutBuffer
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.ppo_loss import PPOConfig, ppo_loss

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainerConfig(BaseTrainerConfig):
    """Configuration for PPO training."""

    # PPO hyperparameters
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Rollout settings
    rollout_steps: int = 2048  # Steps per rollout
    num_envs: int = 1  # Parallel environments
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Update settings
    num_epochs_per_rollout: int = 4  # PPO epochs per rollout
    batch_size: int = 64  # Minibatch size
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    # Training settings
    total_timesteps: int = 1_000_000
    warmup_steps: int = 0

    # Logging and checkpoints
    log_interval: int = 1  # Log every N rollouts
    checkpoint_interval: int = 10  # Checkpoint every N rollouts
    checkpoint_dir: str = "./checkpoints/ppo"

    # Early stopping
    max_steps: int | None = None
    target_reward: float | None = None


class PPOTrainer(BaseTrainer):
    """
    Trainer for Proximal Policy Optimization.

    Works with:
    - RNN experts (continuous/discrete control)
    - Future: LLM policies (Mistral)

    Usage:
        trainer = PPOTrainer(
            policy=expert,
            env=env,
            config=config
        )
        trainer.train()
    """

    def __init__(
        self,
        policy: nn.Module,
        env: Any,  # Environment interface
        config: PPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        config = config or PPOTrainerConfig()
        super().__init__(policy, None, config, optimizer)  # No tokenizer for PPO

        self.policy = policy
        self.env = env

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.rollout_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            num_envs=config.num_envs,
        )

        # PPO-specific state
        self.num_rollouts = 0
        self.best_reward = float("-inf")

    @property
    def ppo_config(self) -> PPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Not used directly - PPO has custom training loop."""
        raise NotImplementedError("PPO uses custom training loop")

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Not used - PPO generates rollouts from environment."""
        raise NotImplementedError("PPO generates rollouts from environment")

    def train(self):
        """Run PPO training loop."""
        logger.info(f"Starting PPO training for {self.ppo_config.total_timesteps} timesteps")
        logger.info(f"Config: {self.config}")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        obs = self.env.reset()

        while self.global_step < self.ppo_config.total_timesteps:
            # Collect rollout
            obs = self._collect_rollout(obs)
            self.num_rollouts += 1

            # Get episode stats before update
            episode_stats = self.buffer.get_episode_stats()

            # Compute advantages
            # Get value estimate for last observation
            last_obs = mx.array([self._obs_to_tensor(obs)])
            result = self.policy(last_obs, deterministic=True)
            last_value = result.get("value", mx.zeros((1,)))
            mx.eval(last_value)

            self.buffer.compute_advantages(last_value)

            # PPO updates
            update_metrics = self._ppo_update()

            # Logging
            if self.num_rollouts % self.config.log_interval == 0:
                elapsed = time.time() - self._start_time
                fps = self.global_step / elapsed

                logger.info(
                    f"Rollout {self.num_rollouts} | "
                    f"Steps: {self.global_step} | "
                    f"Mean Reward: {episode_stats['mean_reward']:.2f} | "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                    f"Value Loss: {update_metrics['value_loss']:.4f} | "
                    f"FPS: {fps:.0f}"
                )

                self.metrics_history.append(
                    {
                        "rollout": self.num_rollouts,
                        "step": self.global_step,
                        **episode_stats,
                        **update_metrics,
                    }
                )

            # Checkpoint
            if self.num_rollouts % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"rollout_{self.num_rollouts}")

            # Check for target reward
            if self.ppo_config.target_reward is not None:
                if episode_stats["mean_reward"] >= self.ppo_config.target_reward:
                    logger.info(f"Target reward reached: {episode_stats['mean_reward']:.2f}")
                    break

            # Update best reward
            if episode_stats["mean_reward"] > self.best_reward:
                self.best_reward = episode_stats["mean_reward"]
                self.save_checkpoint("best")

            # Reset buffer for next rollout
            self.buffer.reset()

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def _collect_rollout(self, initial_obs):
        """Collect rollout_steps of experience."""
        obs = initial_obs

        # Reset hidden state for new rollout
        if hasattr(self.policy, "reset_hidden"):
            self.policy.reset_hidden(batch_size=1)

        for _ in range(self.ppo_config.rollout_steps):
            # Get action from policy
            obs_tensor = mx.array([self._obs_to_tensor(obs)])

            # Forward pass (no gradients needed for rollout collection)
            result = self.policy(obs_tensor, deterministic=False)
            mx.eval(result["action"], result["log_prob"], result.get("value"))

            action = result["action"][0]
            log_prob = float(result["log_prob"][0])
            value = float(result["value"][0]) if result["value"] is not None else 0.0

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Store in buffer
            self.buffer.add(
                observation=obs,
                action=action.tolist() if isinstance(action, mx.array) else action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            self.global_step += 1

            # Reset if done
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        return obs

    def _ppo_update(self) -> dict[str, float]:
        """Perform PPO update on collected rollout."""
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(self.ppo_config.num_epochs_per_rollout):
            for batch in self.buffer.get_batches(
                batch_size=self.ppo_config.batch_size, shuffle=True
            ):
                # Reset hidden state for batch processing (stateless for PPO updates)
                batch_size = len(batch["observations"])
                if hasattr(self.policy, "reset_hidden"):
                    self.policy.reset_hidden(batch_size=batch_size)

                # Convert batch observations to tensors
                obs_tensors = mx.array(
                    [self._obs_to_tensor(o) for o in batch["observations"]], dtype=mx.float32
                )
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Define loss function for gradients
                def loss_fn(model):
                    result = model.get_action_and_value(obs_tensors, action=actions)
                    loss, _ = ppo_loss(
                        log_probs=result["log_prob"],
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        values=result["value"],
                        returns=returns,
                        entropy=result["entropy"],
                        config=self.ppo_config.ppo,
                    )
                    return loss

                # Compute loss and gradients
                loss, grads = nn.value_and_grad(self.policy, loss_fn)(self.policy)

                # Clip gradients
                grads = self.clip_gradients(grads, self.ppo_config.ppo.max_grad_norm)

                # Update
                self.optimizer.update(self.policy, grads)
                mx.eval(self.policy.parameters())

                # Compute metrics separately (for logging)
                result = self.policy.get_action_and_value(obs_tensors, action=actions)
                _, metrics = ppo_loss(
                    log_probs=result["log_prob"],
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    values=result["value"],
                    returns=returns,
                    entropy=result["entropy"],
                    config=self.ppo_config.ppo,
                )

                # Track metrics
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(float(metrics[key]))

                # Early stopping on KL
                if float(metrics["approx_kl"]) > 1.5 * self.ppo_config.ppo.target_kl:
                    logger.debug(f"Early stopping at epoch {epoch} due to high KL")
                    break

        # Average metrics
        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _obs_to_tensor(self, obs) -> list[float]:
        """Convert observation to tensor format."""
        if isinstance(obs, (list, tuple)):
            return list(obs)
        elif isinstance(obs, mx.array):
            return obs.tolist()
        elif isinstance(obs, dict):
            # Flatten dict observation
            flat = []
            for v in obs.values():
                if isinstance(v, (int, float)):
                    flat.append(float(v))
                elif isinstance(v, (list, tuple)):
                    flat.extend([float(x) for x in v])
            return flat
        else:
            return [0.0] * 10  # Default

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / f"{name}.safetensors"
        weights = dict(self.policy.parameters())
        flat_weights = self._flatten_params(weights)
        mx.save_safetensors(str(path), flat_weights)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        weights = mx.load(path)
        self.policy.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
