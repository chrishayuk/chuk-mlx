"""Tests for GRPO loss."""

import mlx.core as mx
import pytest

from chuk_lazarus.training.losses.grpo_loss import (
    GRPOBatch,
    GRPOConfig,
    compute_grpo_advantages,
    grpo_loss,
)


class TestGRPOConfig:
    """Tests for GRPOConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GRPOConfig()

        assert config.group_size == 4
        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.1
        assert config.entropy_coef == 0.01
        assert config.normalize_advantages is True
        assert config.temperature == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = GRPOConfig(
            group_size=8,
            clip_epsilon=0.3,
            kl_coef=0.05,
            entropy_coef=0.02,
            normalize_advantages=False,
            temperature=0.8,
        )

        assert config.group_size == 8
        assert config.clip_epsilon == 0.3
        assert config.kl_coef == 0.05
        assert config.entropy_coef == 0.02
        assert config.normalize_advantages is False
        assert config.temperature == 0.8


class TestGRPOLoss:
    """Tests for grpo_loss function."""

    def test_basic_loss(self):
        """Test basic GRPO loss computation."""
        group_size = 4
        num_prompts = 2
        batch_size = num_prompts * group_size

        log_probs = mx.random.normal((batch_size,)) * 0.1
        ref_log_probs = mx.random.normal((batch_size,)) * 0.1
        rewards = mx.array([1.0, 0.5, -0.5, -1.0, 0.8, 0.3, -0.3, -0.8])

        loss, metrics = grpo_loss(log_probs, ref_log_probs, rewards, group_size)

        assert loss.shape == ()
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "kl_penalty" in metrics
        assert "mean_reward" in metrics
        assert "reward_std" in metrics
        assert "mean_advantage" in metrics
        assert "clip_fraction" in metrics

    def test_loss_with_config(self):
        """Test GRPO loss with custom config."""
        group_size = 4
        num_prompts = 2
        batch_size = num_prompts * group_size

        log_probs = mx.random.normal((batch_size,)) * 0.1
        ref_log_probs = mx.random.normal((batch_size,)) * 0.1
        rewards = mx.array([1.0, 0.5, -0.5, -1.0, 0.8, 0.3, -0.3, -0.8])

        config = GRPOConfig(
            group_size=group_size,
            clip_epsilon=0.1,
            kl_coef=0.2,
            normalize_advantages=True,
        )

        loss, metrics = grpo_loss(log_probs, ref_log_probs, rewards, group_size, config)

        assert loss.shape == ()

    def test_loss_without_normalization(self):
        """Test GRPO loss without advantage normalization."""
        group_size = 4
        num_prompts = 2
        batch_size = num_prompts * group_size

        log_probs = mx.random.normal((batch_size,)) * 0.1
        ref_log_probs = mx.random.normal((batch_size,)) * 0.1
        rewards = mx.array([1.0, 0.5, -0.5, -1.0, 0.8, 0.3, -0.3, -0.8])

        config = GRPOConfig(normalize_advantages=False)

        loss, metrics = grpo_loss(log_probs, ref_log_probs, rewards, group_size, config)

        assert loss.shape == ()

    def test_loss_default_config(self):
        """Test GRPO loss with None config (uses defaults)."""
        group_size = 4
        num_prompts = 2
        batch_size = num_prompts * group_size

        log_probs = mx.random.normal((batch_size,)) * 0.1
        ref_log_probs = mx.random.normal((batch_size,)) * 0.1
        rewards = mx.random.normal((batch_size,))

        loss, metrics = grpo_loss(log_probs, ref_log_probs, rewards, group_size, config=None)

        assert loss.shape == ()


class TestComputeGRPOAdvantages:
    """Tests for compute_grpo_advantages function."""

    def test_basic_advantages(self):
        """Test basic advantage computation."""
        group_size = 4
        rewards = mx.array([1.0, 0.5, -0.5, -1.0, 0.8, 0.3, -0.3, -0.8])

        advantages = compute_grpo_advantages(rewards, group_size, normalize=False)

        assert advantages.shape == (8,)
        # First group mean is 0.0, so advantages = rewards - 0 = rewards
        # Second group mean is 0.0, same

    def test_normalized_advantages(self):
        """Test normalized advantage computation."""
        group_size = 4
        rewards = mx.array([1.0, 0.5, -0.5, -1.0, 0.8, 0.3, -0.3, -0.8])

        advantages = compute_grpo_advantages(rewards, group_size, normalize=True)

        assert advantages.shape == (8,)
        # Normalized advantages should have roughly mean 0 and std 1 within groups

    def test_advantages_single_group(self):
        """Test advantages with single group."""
        group_size = 4
        rewards = mx.array([1.0, 2.0, 3.0, 4.0])

        advantages = compute_grpo_advantages(rewards, group_size, normalize=False)

        # Mean is 2.5, so advantages are [-1.5, -0.5, 0.5, 1.5]
        expected = mx.array([-1.5, -0.5, 0.5, 1.5])
        assert mx.allclose(advantages, expected).item()


class TestGRPOBatch:
    """Tests for GRPOBatch class."""

    def test_init(self):
        """Test batch initialization."""
        batch = GRPOBatch(group_size=4)

        assert batch.group_size == 4
        assert batch.prompts == []
        assert batch.responses == []
        assert batch.rewards == []

    def test_add_prompt_group(self):
        """Test adding a prompt group."""
        batch = GRPOBatch(group_size=4)

        prompt = "What is 2+2?"
        responses = ["4", "four", "3", "5"]
        rewards = [1.0, 0.8, -0.5, -0.5]

        batch.add_prompt_group(prompt, responses, rewards)

        assert len(batch) == 1
        assert batch.prompts[0] == prompt
        assert batch.responses[0] == responses
        assert batch.rewards[0] == rewards

    def test_add_multiple_groups(self):
        """Test adding multiple prompt groups."""
        batch = GRPOBatch(group_size=2)

        batch.add_prompt_group("Q1", ["A1", "A2"], [1.0, 0.5])
        batch.add_prompt_group("Q2", ["B1", "B2"], [0.8, -0.2])

        assert len(batch) == 2

    def test_get_flat_rewards(self):
        """Test getting flattened rewards."""
        batch = GRPOBatch(group_size=2)

        batch.add_prompt_group("Q1", ["A1", "A2"], [1.0, 0.5])
        batch.add_prompt_group("Q2", ["B1", "B2"], [0.8, -0.2])

        flat_rewards = batch.get_flat_rewards()

        expected = mx.array([1.0, 0.5, 0.8, -0.2], dtype=mx.float32)
        assert mx.allclose(flat_rewards, expected).item()

    def test_get_all_sequences(self):
        """Test getting all sequences."""
        batch = GRPOBatch(group_size=2)

        batch.add_prompt_group("Q1:", ["A1", "A2"], [1.0, 0.5])
        batch.add_prompt_group("Q2:", ["B1", "B2"], [0.8, -0.2])

        sequences = batch.get_all_sequences()

        expected = ["Q1:A1", "Q1:A2", "Q2:B1", "Q2:B2"]
        assert sequences == expected

    def test_add_prompt_group_wrong_size_responses(self):
        """Test that adding wrong size responses raises error."""
        batch = GRPOBatch(group_size=4)

        with pytest.raises(AssertionError):
            batch.add_prompt_group("Q1", ["A1", "A2"], [1.0, 0.5, 0.3, 0.2])

    def test_add_prompt_group_wrong_size_rewards(self):
        """Test that adding wrong size rewards raises error."""
        batch = GRPOBatch(group_size=4)

        with pytest.raises(AssertionError):
            batch.add_prompt_group("Q1", ["A1", "A2", "A3", "A4"], [1.0, 0.5])

    def test_len(self):
        """Test __len__ method."""
        batch = GRPOBatch(group_size=2)

        assert len(batch) == 0

        batch.add_prompt_group("Q1", ["A1", "A2"], [1.0, 0.5])
        assert len(batch) == 1

        batch.add_prompt_group("Q2", ["B1", "B2"], [0.8, -0.2])
        assert len(batch) == 2
