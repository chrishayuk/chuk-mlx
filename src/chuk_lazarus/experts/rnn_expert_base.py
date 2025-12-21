"""
Base class for Tiny RNN Experts.

These are small recurrent models (~10-100M params) that handle
specific iterative control tasks. They:
- Take compact numerical observations (not text)
- Maintain hidden state across steps
- Output actions (tool calls or decisions)
- Are trained with RL independently or jointly with the main LLM

Examples:
- Physics controller (adjust angle/velocity iteratively)
- Schedule optimizer (refine time allocations)
- Route planner (iteratively improve paths)
- ARC grid solver (step through transformations)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ExpertConfig:
    """Configuration for an RNN expert."""

    name: str  # Unique identifier (e.g., "physics_controller")
    obs_dim: int  # Observation dimension
    action_dim: int  # Action/output dimension
    hidden_dim: int = 128  # RNN hidden state size
    num_layers: int = 2  # Number of RNN layers
    dropout: float = 0.0  # Dropout rate

    # For discrete actions
    discrete_actions: bool = False
    num_actions: int = 0  # If discrete

    # For continuous actions
    action_low: float = -1.0  # Action bounds
    action_high: float = 1.0

    # Value function (for PPO)
    use_value_head: bool = True


class RNNExpertBase(nn.Module, ABC):
    """
    Base class for tiny RNN experts.

    These are "muscle memory" networks that handle iterative
    control loops without language understanding.
    """

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.hidden_state = None

        # Input projection
        self.input_proj = nn.Linear(config.obs_dim, config.hidden_dim)

        # RNN layers (implemented by subclass)
        self._build_rnn_layers()

        # Policy head
        if config.discrete_actions:
            self.policy_head = nn.Linear(config.hidden_dim, config.num_actions)
        else:
            # For continuous: output mean (and optionally log_std)
            self.policy_mean = nn.Linear(config.hidden_dim, config.action_dim)
            self.policy_log_std = nn.Linear(config.hidden_dim, config.action_dim)
            # Initialize to output center of action range
            self.policy_mean.weight = mx.zeros_like(self.policy_mean.weight)
            self.policy_mean.bias = mx.zeros_like(self.policy_mean.bias)
            self.policy_log_std.weight = mx.zeros_like(self.policy_log_std.weight)
            self.policy_log_std.bias = mx.full(self.policy_log_std.bias.shape, -1.0)  # std ~ 0.37

        # Value head (optional, for PPO)
        if config.use_value_head:
            self.value_head = nn.Linear(config.hidden_dim, 1)
        else:
            self.value_head = None

    @abstractmethod
    def _build_rnn_layers(self):
        """Build the recurrent layers. Implemented by GRU/LSTM subclasses."""
        pass

    @abstractmethod
    def _forward_rnn(self, x: mx.array, hidden: Any | None) -> tuple[mx.array, Any]:
        """Forward through RNN layers. Returns (output, new_hidden)."""
        pass

    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden state for new episode."""
        self.hidden_state = None

    def __call__(
        self, obs: mx.array, hidden: Any | None = None, deterministic: bool = False
    ) -> dict:
        """
        Forward pass through the expert.

        Args:
            obs: Observation tensor, shape (batch, obs_dim)
            hidden: Optional hidden state from previous step
            deterministic: If True, use mean action (no sampling)

        Returns:
            dict with:
                - action: Selected action
                - log_prob: Log probability of action
                - value: Value estimate (if use_value_head)
                - hidden: New hidden state
                - entropy: Action entropy (for exploration bonus)
        """
        batch_size = obs.shape[0]

        # Project input
        x = self.input_proj(obs)
        x = nn.relu(x)

        # Forward through RNN
        if hidden is None:
            hidden = self.hidden_state
        rnn_out, new_hidden = self._forward_rnn(x, hidden)
        self.hidden_state = new_hidden

        # Compute policy
        if self.config.discrete_actions:
            logits = self.policy_head(rnn_out)
            probs = mx.softmax(logits, axis=-1)
            log_probs = mx.log(probs + 1e-10)

            if deterministic:
                action = mx.argmax(logits, axis=-1)
            else:
                # Sample from categorical
                action = self._sample_categorical(probs)

            action_log_prob = log_probs[mx.arange(batch_size), action]
            entropy = -mx.sum(probs * log_probs, axis=-1)
        else:
            # Continuous action
            mean_raw = self.policy_mean(rnn_out)
            log_std = self.policy_log_std(rnn_out)
            # Tighter bounds: min std ~ 0.1 (log=-2.3), max std ~ 1.0 (log=0)
            log_std = mx.clip(log_std, -2, 0)
            std = mx.exp(log_std)

            # Use tanh to squash mean to valid range
            action_range = (self.config.action_high - self.config.action_low) / 2
            action_center = (self.config.action_high + self.config.action_low) / 2
            mean = mx.tanh(mean_raw) * action_range + action_center

            if deterministic:
                action = mean
            else:
                # Sample from Gaussian
                noise = mx.random.normal(mean.shape)
                action = mean + std * noise * action_range * 0.5

            # Clip to action bounds
            action = mx.clip(action, self.config.action_low, self.config.action_high)

            # Log probability of Gaussian
            action_log_prob = self._gaussian_log_prob(action, mean, std)
            entropy = mx.sum(log_std + 0.5 * mx.log(2 * mx.pi * mx.e), axis=-1)

        # Value estimate
        value = None
        if self.value_head is not None:
            value = self.value_head(rnn_out).squeeze(-1)

        return {
            "action": action,
            "log_prob": action_log_prob,
            "value": value,
            "hidden": new_hidden,
            "entropy": entropy,
        }

    def _sample_categorical(self, probs: mx.array) -> mx.array:
        """Sample from categorical distribution."""
        # Use Gumbel-max trick for sampling
        u = mx.random.uniform(probs.shape)
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
        return mx.argmax(mx.log(probs + 1e-10) + gumbel, axis=-1)

    def _gaussian_log_prob(self, action: mx.array, mean: mx.array, std: mx.array) -> mx.array:
        """Compute log probability under Gaussian."""
        var = std**2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / var + mx.log(var) + mx.log(mx.array(2 * 3.14159265))
        )
        return mx.sum(log_prob, axis=-1)

    def get_action_and_value(
        self, obs: mx.array, action: mx.array = None, hidden: Any | None = None
    ) -> dict:
        """
        Get action log prob and value for given observation and action.
        Used during PPO training when we need log_prob of a specific action.
        """
        result = self(obs, hidden, deterministic=False)

        if action is not None:
            # Recompute log prob for provided action
            batch_size = obs.shape[0]
            x = self.input_proj(obs)
            x = nn.relu(x)
            rnn_out, _ = self._forward_rnn(x, hidden)

            if self.config.discrete_actions:
                logits = self.policy_head(rnn_out)
                log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
                action_log_prob = log_probs[mx.arange(batch_size), action]
            else:
                mean_raw = self.policy_mean(rnn_out)
                log_std = self.policy_log_std(rnn_out)
                std = mx.exp(mx.clip(log_std, -2, 0))
                action_range = (self.config.action_high - self.config.action_low) / 2
                action_center = (self.config.action_high + self.config.action_low) / 2
                mean = mx.tanh(mean_raw) * action_range + action_center
                action_log_prob = self._gaussian_log_prob(action, mean, std)

            result["log_prob"] = action_log_prob

        return result
