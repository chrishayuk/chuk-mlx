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

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 4                 # Responses per prompt
    clip_epsilon: float = 0.2           # PPO-style clipping
    kl_coef: float = 0.1               # KL penalty coefficient
    entropy_coef: float = 0.01          # Entropy bonus
    normalize_advantages: bool = True    # Normalize within group
    temperature: float = 1.0            # Sampling temperature


def grpo_loss(
    log_probs: mx.array,
    ref_log_probs: mx.array,
    rewards: mx.array,
    group_size: int,
    config: GRPOConfig = None
) -> Tuple[mx.array, Dict]:
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
    log_probs_grouped = log_probs.reshape(num_prompts, group_size)
    ref_log_probs_grouped = ref_log_probs.reshape(num_prompts, group_size)
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


def compute_grpo_advantages(
    rewards: mx.array,
    group_size: int,
    normalize: bool = True
) -> mx.array:
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
        self.prompts: List[str] = []
        self.responses: List[List[str]] = []  # List of response groups
        self.rewards: List[List[float]] = []

    def add_prompt_group(
        self,
        prompt: str,
        responses: List[str],
        rewards: List[float]
    ):
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

    def get_all_sequences(self) -> List[str]:
        """Get all prompt+response sequences for tokenization."""
        sequences = []
        for prompt, response_group in zip(self.prompts, self.responses):
            for response in response_group:
                sequences.append(prompt + response)
        return sequences

    def __len__(self) -> int:
        return len(self.prompts)


def generate_grpo_samples(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    reward_fn,
    config: GRPOConfig = None,
    max_length: int = 512
) -> GRPOBatch:
    """
    Generate GRPO training samples.

    For each prompt:
    1. Generate group_size responses
    2. Compute reward for each
    3. Package into GRPOBatch

    Args:
        model: The policy model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompts to generate from
        reward_fn: Function (prompt, response) -> reward
        config: GRPO configuration
        max_length: Maximum generation length

    Returns:
        GRPOBatch ready for training
    """
    if config is None:
        config = GRPOConfig()

    batch = GRPOBatch(config.group_size)

    for prompt in prompts:
        responses = []
        rewards = []

        # Generate group_size responses
        for _ in range(config.group_size):
            response = _generate_response(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=config.temperature
            )
            responses.append(response)

            # Compute reward
            reward = reward_fn(prompt, response)
            rewards.append(reward)

        batch.add_prompt_group(prompt, responses, rewards)

    return batch


def _generate_response(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 1.0
) -> str:
    """Generate a single response from the model."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = mx.array([input_ids])

    # Generate (simplified - in practice use proper generation)
    generated = input_ids.copy()

    for _ in range(max_length - len(input_ids)):
        logits, _ = model(mx.array([generated]))
        next_token_logits = logits[0, -1, :] / temperature

        # Sample
        probs = mx.softmax(next_token_logits)
        next_token = _sample_token(probs)

        generated.append(int(next_token))

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

    # Decode response (exclude prompt)
    response_ids = generated[len(input_ids):]
    response = tokenizer.decode(response_ids)

    return response


def _sample_token(probs: mx.array) -> mx.array:
    """Sample a token from probability distribution."""
    # Gumbel-max trick
    u = mx.random.uniform(probs.shape)
    gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
    return mx.argmax(mx.log(probs + 1e-10) + gumbel)
