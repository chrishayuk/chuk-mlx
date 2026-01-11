"""Tests for training losses."""

import pytest

import mlx.core as mx


class TestDualRewardLoss:
    """Tests for dual reward loss."""

    def test_import(self):
        """Test dual reward loss can be imported."""
        from chuk_lazarus.training.losses.dual_reward_loss import (
            DualRewardLossConfig,
            dual_reward_loss,
            classification_only_loss,
        )

        assert dual_reward_loss is not None
        assert DualRewardLossConfig is not None
        assert classification_only_loss is not None

    def test_config_defaults(self):
        """Test config defaults."""
        from chuk_lazarus.training.losses.dual_reward_loss import DualRewardLossConfig

        config = DualRewardLossConfig()
        assert config.classifier_layer == -1
        assert config.classifier_weight == 0.4
        assert config.use_softmax is True


class TestGRPOLoss:
    """Tests for GRPO loss."""

    def test_import(self):
        """Test GRPO loss can be imported."""
        from chuk_lazarus.training.losses.grpo_loss import grpo_loss

        assert grpo_loss is not None


class TestPPOLoss:
    """Tests for PPO loss."""

    def test_import(self):
        """Test PPO loss can be imported."""
        from chuk_lazarus.training.losses.ppo_loss import ppo_loss

        assert ppo_loss is not None
