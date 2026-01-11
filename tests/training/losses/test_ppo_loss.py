"""Tests for PPO loss."""

import mlx.core as mx
import pytest

from chuk_lazarus.training.losses.ppo_loss import (
    PPOConfig,
    ppo_loss,
)


class TestPPOConfig:
    """Tests for PPOConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PPOConfig()

        assert config.clip_epsilon == 0.2
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.max_grad_norm == 0.5
        assert config.target_kl == 0.01
        assert config.normalize_advantages is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PPOConfig(
            clip_epsilon=0.3,
            value_loss_coef=0.25,
            entropy_coef=0.02,
            max_grad_norm=1.0,
            target_kl=0.02,
            normalize_advantages=False,
        )

        assert config.clip_epsilon == 0.3
        assert config.value_loss_coef == 0.25
        assert config.entropy_coef == 0.02
        assert config.max_grad_norm == 1.0
        assert config.target_kl == 0.02
        assert config.normalize_advantages is False


class TestPPOLoss:
    """Tests for ppo_loss function."""

    def test_basic_loss(self):
        """Test basic PPO loss computation."""
        batch_size = 16

        log_probs = mx.random.normal((batch_size,)) * 0.1
        old_log_probs = mx.random.normal((batch_size,)) * 0.1
        advantages = mx.random.normal((batch_size,))
        values = mx.random.normal((batch_size,))
        returns = mx.random.normal((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.5

        loss, metrics = ppo_loss(log_probs, old_log_probs, advantages, values, returns, entropy)

        assert loss.shape == ()
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics
        assert "explained_variance" in metrics

    def test_loss_with_config(self):
        """Test PPO loss with custom config."""
        batch_size = 16

        log_probs = mx.random.normal((batch_size,)) * 0.1
        old_log_probs = mx.random.normal((batch_size,)) * 0.1
        advantages = mx.random.normal((batch_size,))
        values = mx.random.normal((batch_size,))
        returns = mx.random.normal((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.5

        config = PPOConfig(
            clip_epsilon=0.1,
            value_loss_coef=1.0,
            entropy_coef=0.05,
            normalize_advantages=True,
        )

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config
        )

        assert loss.shape == ()

    def test_loss_without_normalization(self):
        """Test PPO loss without advantage normalization."""
        batch_size = 16

        log_probs = mx.random.normal((batch_size,)) * 0.1
        old_log_probs = mx.random.normal((batch_size,)) * 0.1
        advantages = mx.random.normal((batch_size,))
        values = mx.random.normal((batch_size,))
        returns = mx.random.normal((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.5

        config = PPOConfig(normalize_advantages=False)

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config
        )

        assert loss.shape == ()

    def test_loss_default_config(self):
        """Test PPO loss with None config (uses defaults)."""
        batch_size = 16

        log_probs = mx.random.normal((batch_size,)) * 0.1
        old_log_probs = mx.random.normal((batch_size,)) * 0.1
        advantages = mx.random.normal((batch_size,))
        values = mx.random.normal((batch_size,))
        returns = mx.random.normal((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.5

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config=None
        )

        assert loss.shape == ()

    def test_clipping_behavior(self):
        """Test that clipping affects the loss."""
        batch_size = 16

        # Create a scenario where ratio is significantly different from 1
        log_probs = mx.ones((batch_size,)) * 0.5
        old_log_probs = mx.ones((batch_size,)) * (-0.5)  # ratio = exp(1) ~ 2.7
        advantages = mx.ones((batch_size,))
        values = mx.random.normal((batch_size,))
        returns = mx.random.normal((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.5

        config = PPOConfig(clip_epsilon=0.2)

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config
        )

        # Clip fraction should be non-zero since ratio is far from 1
        assert float(metrics["clip_fraction"]) > 0

    def test_value_loss_component(self):
        """Test value loss component."""
        batch_size = 16

        log_probs = mx.zeros((batch_size,))
        old_log_probs = mx.zeros((batch_size,))
        advantages = mx.zeros((batch_size,))
        values = mx.ones((batch_size,)) * 2.0
        returns = mx.ones((batch_size,)) * 1.0  # Different from values
        entropy = mx.ones((batch_size,)) * 0.5

        config = PPOConfig(value_loss_coef=1.0, entropy_coef=0.0)

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config
        )

        # Value loss should be (2.0 - 1.0)^2 = 1.0
        assert float(metrics["value_loss"]) == pytest.approx(1.0, rel=0.01)

    def test_entropy_bonus(self):
        """Test entropy bonus contribution."""
        batch_size = 16

        log_probs = mx.zeros((batch_size,))
        old_log_probs = mx.zeros((batch_size,))
        advantages = mx.zeros((batch_size,))
        values = mx.zeros((batch_size,))
        returns = mx.zeros((batch_size,))
        entropy = mx.ones((batch_size,)) * 0.8

        config = PPOConfig(value_loss_coef=0.0, entropy_coef=1.0)

        loss, metrics = ppo_loss(
            log_probs, old_log_probs, advantages, values, returns, entropy, config
        )

        # Entropy loss is -mean(entropy), so with coef 1.0:
        # total_loss should include -0.8 (negative because we maximize entropy)
        assert float(metrics["entropy"]) == pytest.approx(0.8, rel=0.01)
        assert float(metrics["entropy_loss"]) == pytest.approx(-0.8, rel=0.01)

    def test_explained_variance(self):
        """Test explained variance metric."""
        batch_size = 16

        log_probs = mx.zeros((batch_size,))
        old_log_probs = mx.zeros((batch_size,))
        advantages = mx.zeros((batch_size,))
        # Perfect value prediction
        values = mx.random.normal((batch_size,))
        returns = values  # Values perfectly predict returns
        entropy = mx.ones((batch_size,)) * 0.5

        loss, metrics = ppo_loss(log_probs, old_log_probs, advantages, values, returns, entropy)

        # With perfect prediction, explained variance should be ~1.0
        # (1 - var(returns - values) / var(returns))
        # Since returns == values, var(returns - values) = 0
        assert float(metrics["explained_variance"]) == pytest.approx(1.0, rel=0.01)
