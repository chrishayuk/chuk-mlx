"""
Episode and Transition data structures for RL.

These structures are used throughout the RL pipeline for:
- Collecting experience
- Storing trajectories
- Training data organization
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic

import mlx.core as mx


T = TypeVar('T')  # Generic type for observations


@dataclass
class Transition:
    """
    A single transition (s, a, r, s', done).

    This is the fundamental unit of RL experience.
    """
    observation: Any                    # State s
    action: Any                         # Action a
    reward: float                       # Reward r
    next_observation: Any = None        # Next state s'
    done: bool = False                  # Terminal flag
    truncated: bool = False             # Truncation flag (time limit)

    # Policy info (for off-policy / importance sampling)
    log_prob: float = 0.0               # log π(a|s)
    value: float = 0.0                  # V(s) estimate
    entropy: float = 0.0                # Policy entropy

    # For RNN policies
    hidden_state: Any = None            # h_t
    next_hidden_state: Any = None       # h_{t+1}

    # Metadata
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "done": self.done,
            "truncated": self.truncated,
            "log_prob": self.log_prob,
            "value": self.value,
            "entropy": self.entropy,
            "info": self.info,
        }


@dataclass
class Episode:
    """
    A complete episode (trajectory) of transitions.

    Contains metadata about the episode and methods for
    computing returns, advantages, etc.
    """
    transitions: List[Transition] = field(default_factory=list)

    # Episode metadata
    episode_id: Optional[str] = None
    task: Optional[Dict] = None
    info: Dict[str, Any] = field(default_factory=dict)

    # Cached computations
    _returns: Optional[mx.array] = None
    _advantages: Optional[mx.array] = None

    def add(self, transition: Transition):
        """Add a transition to the episode."""
        self.transitions.append(transition)
        # Invalidate cache
        self._returns = None
        self._advantages = None

    def extend(self, transitions: List[Transition]):
        """Add multiple transitions."""
        self.transitions.extend(transitions)
        self._returns = None
        self._advantages = None

    @property
    def length(self) -> int:
        """Episode length."""
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        """Sum of rewards in episode."""
        return sum(t.reward for t in self.transitions)

    @property
    def rewards(self) -> mx.array:
        """Get rewards as array."""
        return mx.array([t.reward for t in self.transitions])

    @property
    def dones(self) -> mx.array:
        """Get done flags as array."""
        return mx.array([float(t.done) for t in self.transitions])

    @property
    def log_probs(self) -> mx.array:
        """Get log probabilities as array."""
        return mx.array([t.log_prob for t in self.transitions])

    @property
    def values(self) -> mx.array:
        """Get value estimates as array."""
        return mx.array([t.value for t in self.transitions])

    def compute_returns(self, gamma: float = 0.99) -> mx.array:
        """
        Compute discounted returns (reward-to-go).

        Args:
            gamma: Discount factor

        Returns:
            Returns array of shape (episode_length,)
        """
        if self._returns is not None:
            return self._returns

        n = len(self.transitions)
        returns = mx.zeros((n,))
        running_return = 0.0

        for t in range(n - 1, -1, -1):
            running_return = self.transitions[t].reward + gamma * running_return
            returns = returns.at[t].set(running_return)

        self._returns = returns
        return returns

    def compute_advantages(
        self,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> mx.array:
        """
        Compute GAE advantages.

        Args:
            gamma: Discount factor
            lam: GAE lambda

        Returns:
            Advantages array of shape (episode_length,)
        """
        if self._advantages is not None:
            return self._advantages

        n = len(self.transitions)
        advantages = mx.zeros((n,))
        last_gae = 0.0

        for t in range(n - 1, -1, -1):
            if t == n - 1:
                next_value = 0.0  # Terminal
            else:
                next_value = self.transitions[t + 1].value

            delta = (
                self.transitions[t].reward
                + gamma * next_value * (1 - float(self.transitions[t].done))
                - self.transitions[t].value
            )
            last_gae = delta + gamma * lam * (1 - float(self.transitions[t].done)) * last_gae
            advantages = advantages.at[t].set(last_gae)

        self._advantages = advantages
        return advantages

    def get_batch(self) -> Dict[str, mx.array]:
        """Get episode data as a batch dict."""
        returns = self.compute_returns()
        advantages = self.compute_advantages()

        return {
            "observations": [t.observation for t in self.transitions],
            "actions": mx.array([t.action for t in self.transitions]),
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
        }

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self):
        return iter(self.transitions)

    def __getitem__(self, idx) -> Transition:
        return self.transitions[idx]


@dataclass
class EpisodeBatch:
    """
    A batch of episodes for training.

    Useful for training on multiple complete episodes at once.
    """
    episodes: List[Episode] = field(default_factory=list)

    def add(self, episode: Episode):
        """Add an episode to the batch."""
        self.episodes.append(episode)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_transitions(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.total_reward for ep in self.episodes) / len(self.episodes)

    @property
    def mean_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(len(ep) for ep in self.episodes) / len(self.episodes)

    def get_flat_batch(self) -> Dict[str, Any]:
        """
        Flatten all episodes into a single batch.

        Returns a dict with all transitions concatenated.
        """
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        all_returns = []
        all_advantages = []

        for ep in self.episodes:
            batch = ep.get_batch()
            all_obs.extend(batch["observations"])
            all_actions.append(batch["actions"])
            all_rewards.append(batch["rewards"])
            all_dones.append(batch["dones"])
            all_log_probs.append(batch["log_probs"])
            all_values.append(batch["values"])
            all_returns.append(batch["returns"])
            all_advantages.append(batch["advantages"])

        return {
            "observations": all_obs,
            "actions": mx.concatenate(all_actions),
            "rewards": mx.concatenate(all_rewards),
            "dones": mx.concatenate(all_dones),
            "log_probs": mx.concatenate(all_log_probs),
            "values": mx.concatenate(all_values),
            "returns": mx.concatenate(all_returns),
            "advantages": mx.concatenate(all_advantages),
        }

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)
