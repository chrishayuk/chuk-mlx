"""Tests for advantage estimation utilities."""

import mlx.core as mx
import pytest

from chuk_lazarus.training.utils.advantage import (
    compute_gae,
    compute_returns,
    normalize_advantages,
)


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_basic_returns(self):
        """Test basic return computation."""
        rewards = mx.array([[1.0, 1.0, 1.0, 1.0]])
        dones = mx.zeros((1, 4))
        gamma = 0.99

        returns = compute_returns(rewards, dones, gamma)

        # Expected: working backwards
        # r3 = 1
        # r2 = 1 + 0.99 * 1 = 1.99
        # r1 = 1 + 0.99 * 1.99 = 2.9701
        # r0 = 1 + 0.99 * 2.9701 = 3.940399
        assert returns[0, 3].item() == pytest.approx(1.0, rel=1e-3)
        assert returns[0, 2].item() == pytest.approx(1.99, rel=1e-3)
        assert returns[0, 1].item() == pytest.approx(2.9701, rel=1e-3)
        assert returns[0, 0].item() == pytest.approx(3.940399, rel=1e-3)

    def test_returns_with_episode_boundary(self):
        """Test returns reset at episode boundaries."""
        rewards = mx.array([[1.0, 1.0, 1.0, 1.0]])
        dones = mx.array([[0.0, 1.0, 0.0, 0.0]])  # Episode ends after position 1
        gamma = 0.99

        returns = compute_returns(rewards, dones, gamma)

        # After done, return should reset
        # Position 1 is terminal, so return is just the reward
        assert returns[0, 1].item() == pytest.approx(1.0, rel=1e-3)

    def test_returns_batch_processing(self):
        """Test returns work with batch dimension."""
        batch_size = 3
        timesteps = 5
        rewards = mx.ones((batch_size, timesteps))
        dones = mx.zeros((batch_size, timesteps))
        gamma = 0.9

        returns = compute_returns(rewards, dones, gamma)

        assert returns.shape == (batch_size, timesteps)
        # All batches should have same returns since rewards and dones are same
        for b in range(batch_size):
            assert returns[b, -1].item() == pytest.approx(1.0, rel=1e-3)

    def test_returns_with_zero_gamma(self):
        """Test returns with gamma=0 are just immediate rewards."""
        rewards = mx.array([[1.0, 2.0, 3.0]])
        dones = mx.zeros((1, 3))
        gamma = 0.0

        returns = compute_returns(rewards, dones, gamma)

        # With gamma=0, return is just the immediate reward
        assert returns[0, 0].item() == pytest.approx(1.0, rel=1e-3)
        assert returns[0, 1].item() == pytest.approx(2.0, rel=1e-3)
        assert returns[0, 2].item() == pytest.approx(3.0, rel=1e-3)


class TestComputeGAE:
    """Tests for compute_gae function."""

    def test_basic_gae(self):
        """Test basic GAE computation."""
        rewards = mx.array([[1.0, 1.0, 1.0]])
        values = mx.array([[0.5, 0.5, 0.5]])
        dones = mx.zeros((1, 3))
        gamma = 0.99
        lam = 0.95

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        assert advantages.shape == (1, 3)
        assert returns.shape == (1, 3)
        # Returns should be advantages + values
        for i in range(3):
            expected_return = advantages[0, i].item() + values[0, i].item()
            assert returns[0, i].item() == pytest.approx(expected_return, rel=1e-3)

    def test_gae_with_lambda_zero_is_td0(self):
        """Test GAE with lambda=0 is equivalent to TD(0)."""
        rewards = mx.array([[1.0, 1.0, 1.0]])
        values = mx.array([[0.5, 0.6, 0.7]])
        dones = mx.zeros((1, 3))
        gamma = 0.99
        lam = 0.0  # TD(0)

        advantages, _ = compute_gae(rewards, values, dones, gamma, lam)

        # TD(0) advantage = r + gamma * V(s') - V(s)
        # For last position, V(s') = 0 (bootstrap)
        expected_adv_2 = 1.0 + gamma * 0.0 - 0.7
        assert advantages[0, 2].item() == pytest.approx(expected_adv_2, rel=1e-3)

    def test_gae_episode_boundary(self):
        """Test GAE resets at episode boundaries."""
        rewards = mx.array([[1.0, 1.0, 1.0]])
        values = mx.array([[0.5, 0.5, 0.5]])
        dones = mx.array([[0.0, 1.0, 0.0]])  # Episode ends at position 1
        gamma = 0.99
        lam = 0.95

        advantages, _ = compute_gae(rewards, values, dones, gamma, lam)

        # At done position, next value is effectively 0
        assert advantages.shape == (1, 3)

    def test_gae_batch_processing(self):
        """Test GAE works with batch dimension."""
        batch_size = 2
        timesteps = 4
        rewards = mx.ones((batch_size, timesteps))
        values = mx.ones((batch_size, timesteps)) * 0.5
        dones = mx.zeros((batch_size, timesteps))
        gamma = 0.99
        lam = 0.95

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        assert advantages.shape == (batch_size, timesteps)
        assert returns.shape == (batch_size, timesteps)


class TestNormalizeAdvantages:
    """Tests for normalize_advantages function."""

    def test_basic_normalization(self):
        """Test basic normalization."""
        advantages = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        normalized = normalize_advantages(advantages)

        # Mean should be approximately 0
        mean = mx.mean(normalized).item()
        assert mean == pytest.approx(0.0, abs=1e-5)

        # Std should be approximately 1
        std = mx.sqrt(mx.var(normalized)).item()
        assert std == pytest.approx(1.0, rel=1e-2)

    def test_normalization_preserves_shape(self):
        """Test normalization preserves input shape."""
        advantages = mx.random.uniform(shape=(4, 8))

        normalized = normalize_advantages(advantages)

        assert normalized.shape == advantages.shape

    def test_normalization_constant_input(self):
        """Test normalization handles constant input."""
        advantages = mx.ones((3, 3)) * 5.0

        normalized = normalize_advantages(advantages)

        # With constant input, std is 0, so normalized should be 0
        assert mx.all(mx.abs(normalized) < 1e-5).item()

    def test_normalization_with_negative_values(self):
        """Test normalization works with negative values."""
        advantages = mx.array([[-5.0, -2.0, 0.0, 2.0, 5.0]])

        normalized = normalize_advantages(advantages)

        # Mean should be approximately 0
        mean = mx.mean(normalized).item()
        assert mean == pytest.approx(0.0, abs=1e-5)
