"""
Group Relative Policy Optimization (GRPO) Loss

GRPO is a simpler alternative to PPO that:
- Generates multiple responses per prompt (a "group")
- Computes rewards for each response
- Uses group-relative advantages (no value function needed!)
- Updates policy to prefer higher-reward responses

This is inspired by DeepSeek's approach and is particularly
well-suited for LLM training where:
- Generating multiple samples is easy
- Value estimation for text is hard
- Relative comparisons are more meaningful than absolute values

Key benefits:
- No value function required (simpler architecture)
- Natural exploration through group sampling
- Works well with sparse rewards
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    group_size: int = 4  # Responses per prompt
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.1  # KL penalty coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    normalize_advantages: bool = True  # Normalize within group
    temperature: float = 1.0  # Sampling temperature


def grpo_loss(
    log_probs: mx.array,
    ref_log_probs: mx.array,
    rewards: mx.array,
    group_size: int,
    config: GRPOConfig = None,
) -> tuple[mx.array, dict]:
    """
    Compute GRPO loss.

    Args:
        log_probs: Current policy log probs, shape (batch,)
            where batch = num_prompts * group_size
        ref_log_probs: Reference policy log probs, shape (batch,)
        rewards: Rewards for each response, shape (batch,)
        group_size: Number of responses per prompt
        config: GRPO configuration

    Returns:
        loss: GRPO loss
        metrics: Training metrics
    """
    if config is None:
        config = GRPOConfig()

    batch_size = log_probs.shape[0]
    num_prompts = batch_size // group_size

    # Reshape to (num_prompts, group_size)
    rewards_grouped = rewards.reshape(num_prompts, group_size)

    # Compute group-relative advantages
    # For each prompt, advantage = reward - mean(rewards in group)
    group_mean = mx.mean(rewards_grouped, axis=1, keepdims=True)
    advantages = rewards_grouped - group_mean

    if config.normalize_advantages:
        group_std = mx.sqrt(mx.var(rewards_grouped, axis=1, keepdims=True) + 1e-8)
        advantages = advantages / group_std

    # Flatten back
    advantages = advantages.reshape(-1)

    # Policy gradient with clipping (like PPO)
    ratio = mx.exp(log_probs - ref_log_probs)
    clipped_ratio = mx.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)

    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = mx.mean(mx.maximum(policy_loss_unclipped, policy_loss_clipped))

    # KL penalty (stay close to reference)
    kl_penalty = mx.mean(log_probs - ref_log_probs)

    # Total loss
    total_loss = policy_loss + config.kl_coef * kl_penalty

    # Metrics
    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "mean_reward": mx.mean(rewards),
        "reward_std": mx.sqrt(mx.var(rewards)),
        "mean_advantage": mx.mean(advantages),
        "clip_fraction": mx.mean((mx.abs(ratio - 1) > config.clip_epsilon).astype(mx.float32)),
    }

    return total_loss, metrics


def compute_grpo_advantages(rewards: mx.array, group_size: int, normalize: bool = True) -> mx.array:
    """
    Compute group-relative advantages.

    Args:
        rewards: Shape (batch,) where batch = num_prompts * group_size
        group_size: Number of responses per prompt
        normalize: Whether to normalize by group std

    Returns:
        advantages: Shape (batch,)
    """
    batch_size = rewards.shape[0]
    num_prompts = batch_size // group_size

    rewards_grouped = rewards.reshape(num_prompts, group_size)
    group_mean = mx.mean(rewards_grouped, axis=1, keepdims=True)
    advantages = rewards_grouped - group_mean

    if normalize:
        group_std = mx.sqrt(mx.var(rewards_grouped, axis=1, keepdims=True) + 1e-8)
        advantages = advantages / group_std

    return advantages.reshape(-1)


class GRPOBatch:
    """
    Helper class for organizing GRPO batches.

    Ensures proper grouping of responses for the same prompt.
    """

    def __init__(self, group_size: int):
        self.group_size = group_size
        self.prompts: list[str] = []
        self.responses: list[list[str]] = []  # List of response groups
        self.rewards: list[list[float]] = []

    def add_prompt_group(self, prompt: str, responses: list[str], rewards: list[float]):
        """Add a prompt with its group of responses and rewards."""
        assert len(responses) == self.group_size
        assert len(rewards) == self.group_size

        self.prompts.append(prompt)
        self.responses.append(responses)
        self.rewards.append(rewards)

    def get_flat_rewards(self) -> mx.array:
        """Get flattened rewards array."""
        flat = []
        for group in self.rewards:
            flat.extend(group)
        return mx.array(flat, dtype=mx.float32)

    def get_all_sequences(self) -> list[str]:
        """Get all prompt+response sequences for tokenization."""
        sequences = []
        for prompt, response_group in zip(self.prompts, self.responses):
            for response in response_group:
                sequences.append(prompt + response)
        return sequences

    def __len__(self) -> int:
        return len(self.prompts)
