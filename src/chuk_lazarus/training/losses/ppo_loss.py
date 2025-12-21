"""
Proximal Policy Optimization (PPO) Loss

PPO is a policy gradient method that uses a clipped surrogate objective
to prevent too-large policy updates, making training more stable.

Key components:
- Clipped surrogate objective (policy loss)
- Value function loss
- Entropy bonus (for exploration)

Paper: https://arxiv.org/abs/1707.06347
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    clip_epsilon: float = 0.2  # Clipping parameter
    value_loss_coef: float = 0.5  # Value loss weight
    entropy_coef: float = 0.01  # Entropy bonus weight
    max_grad_norm: float = 0.5  # Gradient clipping
    target_kl: float = 0.01  # KL target for early stopping
    normalize_advantages: bool = True  # Normalize advantages


def ppo_loss(
    log_probs: mx.array,
    old_log_probs: mx.array,
    advantages: mx.array,
    values: mx.array,
    returns: mx.array,
    entropy: mx.array,
    config: PPOConfig = None,
) -> tuple[mx.array, dict]:
    """
    Compute PPO loss for a batch of trajectories.

    Args:
        log_probs: Current policy log probs, shape (batch,)
        old_log_probs: Old policy log probs (from rollout), shape (batch,)
        advantages: GAE advantages, shape (batch,)
        values: Value estimates, shape (batch,)
        returns: Value targets, shape (batch,)
        entropy: Policy entropy, shape (batch,)
        config: PPO configuration

    Returns:
        total_loss: Combined PPO loss
        metrics: Dict with component losses and diagnostics
    """
    if config is None:
        config = PPOConfig()

    # Normalize advantages
    if config.normalize_advantages:
        adv_mean = mx.mean(advantages)
        adv_std = mx.sqrt(mx.var(advantages) + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

    # Policy loss (clipped surrogate objective)
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)

    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = mx.mean(mx.maximum(policy_loss_unclipped, policy_loss_clipped))

    # Value loss (clipped as well for stability)
    value_loss = mx.mean((values - returns) ** 2)

    # Entropy bonus (negative because we want to maximize entropy)
    entropy_loss = -mx.mean(entropy)

    # Combined loss
    total_loss = (
        policy_loss + config.value_loss_coef * value_loss + config.entropy_coef * entropy_loss
    )

    # Compute diagnostics (no gradients needed)
    approx_kl = mx.mean((ratio - 1) - (log_probs - old_log_probs))
    clip_fraction = mx.mean((mx.abs(ratio - 1) > config.clip_epsilon).astype(mx.float32))

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "entropy": mx.mean(entropy),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": 1 - mx.var(returns - values) / (mx.var(returns) + 1e-8),
    }

    return total_loss, metrics


def compute_ppo_loss_for_batch(
    policy_model: nn.Module,
    value_model: nn.Module,
    batch: dict[str, mx.array],
    config: PPOConfig = None,
) -> tuple[mx.array, dict]:
    """
    Compute PPO loss for a batch from the rollout buffer.

    This is the main entry point for PPO training updates.

    Args:
        policy_model: The policy network
        value_model: The value network (can be same as policy with separate head)
        batch: Dict containing:
            - observations: (batch, obs_dim)
            - actions: (batch, action_dim) or (batch,) for discrete
            - old_log_probs: (batch,)
            - advantages: (batch,)
            - returns: (batch,)
        config: PPO configuration

    Returns:
        loss: Total PPO loss
        metrics: Training metrics
    """
    if config is None:
        config = PPOConfig()

    observations = batch["observations"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    # Get current policy outputs
    policy_output = policy_model.get_action_and_value(observations, action=actions)

    log_probs = policy_output["log_prob"]
    entropy = policy_output["entropy"]

    # Get value estimates
    if value_model is policy_model:
        values = policy_output["value"]
    else:
        values = value_model(observations)

    return ppo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
        entropy=entropy,
        config=config,
    )
