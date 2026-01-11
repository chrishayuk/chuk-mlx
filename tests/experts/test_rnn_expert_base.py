"""Tests for RNN expert base module."""

import pytest

from chuk_lazarus.experts.rnn_expert_base import ExpertConfig


class TestExpertConfig:
    """Tests for ExpertConfig dataclass."""

    def test_expert_config_minimal(self):
        """Test minimal expert config."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=5,
        )
        assert config.name == "test_expert"
        assert config.obs_dim == 10
        assert config.action_dim == 5
        assert config.hidden_dim == 128  # default
        assert config.num_layers == 2  # default
        assert config.dropout == 0.0  # default
        assert config.discrete_actions is False  # default
        assert config.use_value_head is True  # default

    def test_expert_config_full(self):
        """Test expert config with all options."""
        config = ExpertConfig(
            name="full_expert",
            obs_dim=100,
            action_dim=10,
            hidden_dim=256,
            num_layers=4,
            dropout=0.1,
            discrete_actions=True,
            num_actions=8,
            action_low=-2.0,
            action_high=2.0,
            use_value_head=False,
        )
        assert config.name == "full_expert"
        assert config.obs_dim == 100
        assert config.action_dim == 10
        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.dropout == 0.1
        assert config.discrete_actions is True
        assert config.num_actions == 8
        assert config.action_low == -2.0
        assert config.action_high == 2.0
        assert config.use_value_head is False

    def test_expert_config_continuous_action_bounds(self):
        """Test continuous action bounds."""
        config = ExpertConfig(
            name="continuous_expert",
            obs_dim=10,
            action_dim=5,
            action_low=-5.0,
            action_high=5.0,
        )
        assert config.action_low == -5.0
        assert config.action_high == 5.0
        assert config.discrete_actions is False
