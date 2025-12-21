"""
Advantage estimation for policy gradient methods.

GAE (Generalized Advantage Estimation) provides a balance between
bias and variance in advantage estimates.
"""

import mlx.core as mx


def compute_returns(rewards: mx.array, dones: mx.array, gamma: float = 0.99) -> mx.array:
    """
    Compute discounted returns (reward-to-go).

    Args:
        rewards: Shape (batch, timesteps) - rewards at each step
        dones: Shape (batch, timesteps) - episode termination flags
        gamma: Discount factor

    Returns:
        returns: Shape (batch, timesteps) - discounted returns
    """
    batch_size, timesteps = rewards.shape
    returns = mx.zeros_like(rewards)

    # Work backwards from the end
    running_return = mx.zeros((batch_size,))

    for t in range(timesteps - 1, -1, -1):
        # Reset return on episode boundary
        running_return = rewards[:, t] + gamma * running_return * (1 - dones[:, t])
        returns = returns.at[:, t].set(running_return)

    return returns


def compute_gae(
    rewards: mx.array, values: mx.array, dones: mx.array, gamma: float = 0.99, lam: float = 0.95
) -> tuple[mx.array, mx.array]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE provides a balance between:
    - TD(0) advantages (low variance, high bias)
    - Monte Carlo advantages (high variance, low bias)

    Args:
        rewards: Shape (batch, timesteps) - rewards at each step
        values: Shape (batch, timesteps) - value estimates from critic
        dones: Shape (batch, timesteps) - episode termination flags
        gamma: Discount factor
        lam: GAE lambda (0 = TD(0), 1 = Monte Carlo)

    Returns:
        advantages: Shape (batch, timesteps) - GAE advantages
        returns: Shape (batch, timesteps) - value targets (advantages + values)
    """
    batch_size, timesteps = rewards.shape
    advantages = mx.zeros_like(rewards)

    # Bootstrap value for last timestep (assume 0 if episode ended)
    last_value = mx.zeros((batch_size,))
    last_gae = mx.zeros((batch_size,))

    # Work backwards
    for t in range(timesteps - 1, -1, -1):
        if t == timesteps - 1:
            next_value = last_value
        else:
            next_value = values[:, t + 1]

        # TD error: r + gamma * V(s') - V(s)
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]

        # GAE: sum of discounted TD errors
        last_gae = delta + gamma * lam * (1 - dones[:, t]) * last_gae
        advantages = advantages.at[:, t].set(last_gae)

    # Returns = advantages + values (the target for value function)
    returns = advantages + values

    return advantages, returns


def normalize_advantages(advantages: mx.array, eps: float = 1e-8) -> mx.array:
    """
    Normalize advantages to have zero mean and unit variance.

    This helps stabilize training across different reward scales.

    Args:
        advantages: Shape (batch, timesteps) or (batch,)
        eps: Small constant for numerical stability

    Returns:
        normalized: Normalized advantages
    """
    mean = mx.mean(advantages)
    std = mx.sqrt(mx.var(advantages) + eps)
    return (advantages - mean) / std
