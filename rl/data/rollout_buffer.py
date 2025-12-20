"""
Rollout Buffer for storing trajectories during RL training.

The rollout buffer collects experience during environment interaction
and prepares batches for policy updates.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
import random

import mlx.core as mx

from ..utils.advantage import compute_gae, normalize_advantages

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single transition in the environment."""
    observation: Any                    # State observation
    action: Any                         # Action taken
    reward: float                       # Reward received
    done: bool                          # Episode terminated
    log_prob: float                     # Log probability of action
    value: Optional[float] = None       # Value estimate (for PPO)
    hidden_state: Optional[Any] = None  # RNN hidden state


@dataclass
class Episode:
    """A complete episode trajectory."""
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    def add(self, transition: Transition):
        """Add a transition to the episode."""
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.length += 1

    def get_arrays(self) -> Dict[str, mx.array]:
        """Convert episode to MLX arrays."""
        return {
            "rewards": mx.array([t.reward for t in self.transitions]),
            "dones": mx.array([float(t.done) for t in self.transitions]),
            "log_probs": mx.array([t.log_prob for t in self.transitions]),
            "values": mx.array([t.value for t in self.transitions if t.value is not None]),
        }


class RolloutBuffer:
    """
    Buffer for storing rollout data during RL training.

    Supports:
    - Collecting transitions from multiple parallel environments
    - Computing advantages (GAE)
    - Generating training batches
    - Mixed storage of LLM and RNN expert transitions
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_envs: int = 1
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_envs = num_envs

        # Storage
        self.observations: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.hidden_states: List[Any] = []

        # Computed after collection
        self.advantages: Optional[mx.array] = None
        self.returns: Optional[mx.array] = None

        # Episode tracking
        self.episodes: List[Episode] = []
        self.current_episodes: List[Episode] = [Episode() for _ in range(num_envs)]

        # Pointer
        self.ptr = 0
        self.full = False

    def reset(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.hidden_states = []
        self.advantages = None
        self.returns = None
        self.episodes = []
        self.current_episodes = [Episode() for _ in range(self.num_envs)]
        self.ptr = 0
        self.full = False

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float = None,
        hidden_state: Any = None,
        env_idx: int = 0
    ):
        """
        Add a transition to the buffer.

        Args:
            observation: State observation
            action: Action taken
            reward: Reward received
            done: Episode terminated
            log_prob: Log probability of action
            value: Value estimate (optional, for PPO)
            hidden_state: RNN hidden state (optional)
            env_idx: Environment index (for parallel envs)
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value if value is not None else 0.0)
        self.hidden_states.append(hidden_state)

        # Track episode
        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            hidden_state=hidden_state
        )
        self.current_episodes[env_idx].add(transition)

        if done:
            self.episodes.append(self.current_episodes[env_idx])
            self.current_episodes[env_idx] = Episode()

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def add_batch(
        self,
        observations: List[Any],
        actions: List[Any],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[float],
        values: List[float] = None,
        hidden_states: List[Any] = None
    ):
        """Add a batch of transitions (from parallel envs)."""
        if values is None:
            values = [None] * len(observations)
        if hidden_states is None:
            hidden_states = [None] * len(observations)

        for i, (obs, action, reward, done, log_prob, value, hidden) in enumerate(
            zip(observations, actions, rewards, dones, log_probs, values, hidden_states)
        ):
            self.add(obs, action, reward, done, log_prob, value, hidden, env_idx=i % self.num_envs)

    def compute_advantages(self, last_values: mx.array = None):
        """
        Compute advantages and returns using GAE.

        Args:
            last_values: Value estimates for final states (for bootstrapping)
        """
        rewards = mx.array(self.rewards, dtype=mx.float32)
        values = mx.array(self.values, dtype=mx.float32)
        dones = mx.array(self.dones, dtype=mx.float32)

        if last_values is None:
            last_values = mx.zeros((1,))

        # Compute GAE using Python list then convert to array
        n = len(self.rewards)
        advantages_list = [0.0] * n
        last_gae = 0.0

        for t in range(n - 1, -1, -1):
            if t == n - 1:
                next_value = float(last_values[0]) if last_values.shape[0] > 0 else 0.0
            else:
                next_value = float(values[t + 1])

            next_non_terminal = 1.0 - float(dones[t])

            delta = float(rewards[t]) + self.gamma * next_value * next_non_terminal - float(values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages_list[t] = last_gae

        self.advantages = mx.array(advantages_list, dtype=mx.float32)
        self.returns = self.advantages + values

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Iterator[Dict[str, mx.array]]:
        """
        Generate training batches from the buffer.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data

        Yields:
            Dict with batch data
        """
        if self.advantages is None:
            self.compute_advantages()

        n = len(self.observations)
        indices = list(range(n))

        if shuffle:
            random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]

            yield {
                "observations": [self.observations[i] for i in batch_indices],
                "actions": mx.array([self.actions[i] for i in batch_indices]),
                "old_log_probs": mx.array([self.log_probs[i] for i in batch_indices]),
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
                "values": mx.array([self.values[i] for i in batch_indices]),
            }

    def get_all(self) -> Dict[str, mx.array]:
        """Get all data as a single batch."""
        if self.advantages is None:
            self.compute_advantages()

        return {
            "observations": self.observations,
            "actions": mx.array(self.actions),
            "old_log_probs": mx.array(self.log_probs),
            "advantages": self.advantages,
            "returns": self.returns,
            "values": mx.array(self.values),
        }

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics about collected episodes."""
        if not self.episodes:
            return {
                "num_episodes": 0,
                "mean_reward": 0.0,
                "mean_length": 0.0,
            }

        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.length for ep in self.episodes]

        return {
            "num_episodes": len(self.episodes),
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_length": sum(lengths) / len(lengths),
        }

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def is_full(self) -> bool:
        return self.full or len(self.observations) >= self.buffer_size
