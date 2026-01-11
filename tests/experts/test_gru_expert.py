"""Tests for GRU expert."""

import mlx.core as mx

from chuk_lazarus.experts.gru_expert import (
    GRUCell,
    GRUExpert,
    create_physics_controller,
    create_scheduler_expert,
)
from chuk_lazarus.experts.rnn_expert_base import ExpertConfig


class TestGRUCell:
    """Tests for GRUCell class."""

    def test_init(self):
        """Test GRUCell initialization."""
        cell = GRUCell(input_dim=10, hidden_dim=32)

        assert cell.hidden_dim == 32
        assert hasattr(cell, "W_r")  # Reset gate
        assert hasattr(cell, "W_z")  # Update gate
        assert hasattr(cell, "W_h")  # Candidate hidden

    def test_forward_without_hidden(self):
        """Test forward pass without initial hidden state."""
        cell = GRUCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))  # Batch of 4
        output = cell(x)

        assert output.shape == (4, 32)

    def test_forward_with_hidden(self):
        """Test forward pass with initial hidden state."""
        cell = GRUCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))
        h = mx.random.normal((4, 32))
        output = cell(x, h)

        assert output.shape == (4, 32)

    def test_forward_hidden_broadcast(self):
        """Test forward pass with hidden state that needs broadcasting."""
        cell = GRUCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))  # Batch of 4
        h = mx.random.normal((1, 32))  # Single hidden state
        output = cell(x, h)

        assert output.shape == (4, 32)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        cell = GRUCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((1, 10))
        output = cell(x)

        assert output.shape == (1, 32)


class TestGRUExpert:
    """Tests for GRUExpert class."""

    def test_init(self):
        """Test GRUExpert initialization."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        assert expert.config == config
        assert len(expert.gru_layers) == 2

    def test_build_rnn_layers(self):
        """Test _build_rnn_layers creates correct number of layers."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=64,
            num_layers=3,
        )
        expert = GRUExpert(config)

        assert len(expert.gru_layers) == 3
        for cell in expert.gru_layers:
            assert isinstance(cell, GRUCell)
            assert cell.hidden_dim == 64

    def test_forward_basic(self):
        """Test basic forward pass."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        obs = mx.random.normal((4, 10))
        result = expert(obs)

        assert "action" in result
        assert result["action"].shape[0] == 4
        assert result["action"].shape[1] == 2

    def test_forward_with_value_head(self):
        """Test forward pass with value head enabled."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
            use_value_head=True,
        )
        expert = GRUExpert(config)

        obs = mx.random.normal((4, 10))
        result = expert(obs)

        assert "action" in result
        assert "value" in result
        # Value shape is (batch_size,) not (batch_size, 1)
        assert result["value"].shape == (4,)

    def test_forward_without_value_head(self):
        """Test forward pass without value head."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
            use_value_head=False,
        )
        expert = GRUExpert(config)

        obs = mx.random.normal((4, 10))
        result = expert(obs)

        assert "action" in result
        # Value is in output but set to None when use_value_head is False
        assert result["value"] is None

    def test_reset_hidden(self):
        """Test hidden state reset."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        # First forward pass stores hidden state
        obs = mx.random.normal((4, 10))
        result = expert(obs)
        assert "hidden" in result

        # Reset hidden - just verify it doesn't raise
        expert.reset_hidden(batch_size=2)

        # After reset, next forward should work fresh
        obs2 = mx.random.normal((2, 10))
        result2 = expert(obs2)
        assert "hidden" in result2

    def test_forward_rnn_without_hidden(self):
        """Test _forward_rnn without hidden state."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        x = mx.random.normal((4, 32))
        output, new_hidden = expert._forward_rnn(x, hidden=None)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 2

    def test_forward_rnn_with_hidden(self):
        """Test _forward_rnn with hidden state."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        x = mx.random.normal((4, 32))
        hidden = [mx.random.normal((4, 32)), mx.random.normal((4, 32))]
        output, new_hidden = expert._forward_rnn(x, hidden=hidden)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 2

    def test_forward_rnn_with_single_hidden(self):
        """Test _forward_rnn when hidden is not a list."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = GRUExpert(config)

        x = mx.random.normal((4, 32))
        hidden = mx.random.normal((4, 32))  # Not a list
        output, new_hidden = expert._forward_rnn(x, hidden=hidden)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 2

    def test_forward_rnn_with_dropout(self):
        """Test _forward_rnn with dropout."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=3,
            dropout=0.1,
        )
        expert = GRUExpert(config)

        x = mx.random.normal((4, 32))
        output, new_hidden = expert._forward_rnn(x, hidden=None)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 3


class TestCreatePhysicsController:
    """Tests for create_physics_controller function."""

    def test_default_params(self):
        """Test with default parameters."""
        expert = create_physics_controller()

        assert expert.config.name == "physics_controller"
        assert expert.config.obs_dim == 10
        assert expert.config.action_dim == 2
        assert expert.config.hidden_dim == 64
        assert expert.config.num_layers == 2
        assert expert.config.discrete_actions is False
        assert expert.config.use_value_head is True

    def test_custom_params(self):
        """Test with custom parameters."""
        expert = create_physics_controller(obs_dim=20, action_dim=4, hidden_dim=128)

        assert expert.config.obs_dim == 20
        assert expert.config.action_dim == 4
        assert expert.config.hidden_dim == 128

    def test_action_bounds(self):
        """Test action bounds configuration."""
        expert = create_physics_controller()

        assert expert.config.action_low == -1.0
        assert expert.config.action_high == 1.0


class TestCreateSchedulerExpert:
    """Tests for create_scheduler_expert function."""

    def test_default_params(self):
        """Test with default parameters."""
        expert = create_scheduler_expert()

        assert expert.config.name == "scheduler"
        assert expert.config.obs_dim == 10 * 3 + 5  # num_tasks * 3 + 5
        assert expert.config.action_dim == 10 + 2  # num_tasks + 2
        assert expert.config.hidden_dim == 128

    def test_custom_params(self):
        """Test with custom parameters."""
        expert = create_scheduler_expert(num_tasks=20, hidden_dim=256)

        assert expert.config.obs_dim == 20 * 3 + 5
        assert expert.config.action_dim == 20 + 2
        assert expert.config.hidden_dim == 256

    def test_discrete_actions(self):
        """Test that discrete_actions is False."""
        expert = create_scheduler_expert()

        assert expert.config.discrete_actions is False

    def test_value_head(self):
        """Test that value head is enabled."""
        expert = create_scheduler_expert()

        assert expert.config.use_value_head is True
