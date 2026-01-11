"""Tests for LSTM expert."""

import mlx.core as mx

from chuk_lazarus.experts.lstm_expert import (
    LSTMCell,
    LSTMExpert,
    create_arc_solver_expert,
    create_planning_expert,
)
from chuk_lazarus.experts.rnn_expert_base import ExpertConfig


class TestLSTMCell:
    """Tests for LSTMCell class."""

    def test_init(self):
        """Test LSTMCell initialization."""
        cell = LSTMCell(input_dim=10, hidden_dim=32)

        assert cell.hidden_dim == 32
        assert hasattr(cell, "gates")

    def test_forward_without_state(self):
        """Test forward pass without initial state."""
        cell = LSTMCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))  # Batch of 4
        output, state = cell(x)

        assert output.shape == (4, 32)
        assert len(state) == 2  # (h, c)
        assert state[0].shape == (4, 32)  # h
        assert state[1].shape == (4, 32)  # c

    def test_forward_with_state(self):
        """Test forward pass with initial state."""
        cell = LSTMCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))
        h = mx.random.normal((4, 32))
        c = mx.random.normal((4, 32))
        state = (h, c)

        output, new_state = cell(x, state)

        assert output.shape == (4, 32)
        assert len(new_state) == 2
        assert new_state[0].shape == (4, 32)
        assert new_state[1].shape == (4, 32)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        cell = LSTMCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((1, 10))
        output, state = cell(x)

        assert output.shape == (1, 32)
        assert state[0].shape == (1, 32)
        assert state[1].shape == (1, 32)

    def test_output_matches_hidden(self):
        """Test that output equals h in state."""
        cell = LSTMCell(input_dim=10, hidden_dim=32)

        x = mx.random.normal((4, 10))
        output, state = cell(x)

        # Output should be the same as h (new hidden state)
        assert mx.allclose(output, state[0]).item()


class TestLSTMExpert:
    """Tests for LSTMExpert class."""

    def test_init(self):
        """Test LSTMExpert initialization."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = LSTMExpert(config)

        assert expert.config == config
        assert len(expert.lstm_layers) == 2

    def test_build_rnn_layers(self):
        """Test _build_rnn_layers creates correct number of layers."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=64,
            num_layers=3,
        )
        expert = LSTMExpert(config)

        assert len(expert.lstm_layers) == 3
        for cell in expert.lstm_layers:
            assert isinstance(cell, LSTMCell)
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
        expert = LSTMExpert(config)

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
        expert = LSTMExpert(config)

        obs = mx.random.normal((4, 10))
        result = expert(obs)

        assert "action" in result
        assert "value" in result
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
        expert = LSTMExpert(config)

        obs = mx.random.normal((4, 10))
        result = expert(obs)

        assert "action" in result
        assert result["value"] is None

    def test_forward_rnn_without_hidden(self):
        """Test _forward_rnn without hidden state."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = LSTMExpert(config)

        x = mx.random.normal((4, 32))
        output, new_hidden = expert._forward_rnn(x, hidden=None)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 2
        # Each hidden state is a tuple (h, c)
        for state in new_hidden:
            assert len(state) == 2
            assert state[0].shape == (4, 32)
            assert state[1].shape == (4, 32)

    def test_forward_rnn_with_hidden(self):
        """Test _forward_rnn with hidden state."""
        config = ExpertConfig(
            name="test_expert",
            obs_dim=10,
            action_dim=2,
            hidden_dim=32,
            num_layers=2,
        )
        expert = LSTMExpert(config)

        x = mx.random.normal((4, 32))
        hidden = [
            (mx.random.normal((4, 32)), mx.random.normal((4, 32))),
            (mx.random.normal((4, 32)), mx.random.normal((4, 32))),
        ]
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
        expert = LSTMExpert(config)

        x = mx.random.normal((4, 32))
        output, new_hidden = expert._forward_rnn(x, hidden=None)

        assert output.shape == (4, 32)
        assert len(new_hidden) == 3


class TestCreatePlanningExpert:
    """Tests for create_planning_expert function."""

    def test_default_params(self):
        """Test with default parameters."""
        expert = create_planning_expert()

        assert expert.config.name == "planner"
        assert expert.config.obs_dim == 20
        assert expert.config.action_dim == 5
        assert expert.config.hidden_dim == 128
        assert expert.config.num_layers == 3
        assert expert.config.discrete_actions is False
        assert expert.config.use_value_head is True

    def test_custom_params(self):
        """Test with custom parameters."""
        expert = create_planning_expert(state_dim=50, action_dim=10, hidden_dim=256)

        assert expert.config.obs_dim == 50
        assert expert.config.action_dim == 10
        assert expert.config.hidden_dim == 256


class TestCreateArcSolverExpert:
    """Tests for create_arc_solver_expert function."""

    def test_default_params(self):
        """Test with default parameters."""
        expert = create_arc_solver_expert()

        assert expert.config.name == "arc_solver"
        # Default grid_size=30, so obs_dim = 30*30*2 + 5 = 1805
        assert expert.config.obs_dim == 30 * 30 * 2 + 5
        # Default num_actions=20, so action_dim = 20
        assert expert.config.action_dim == 20
        assert expert.config.hidden_dim == 256
        assert expert.config.num_layers == 3
        assert expert.config.discrete_actions is True
        assert expert.config.use_value_head is True

    def test_custom_params(self):
        """Test with custom parameters."""
        expert = create_arc_solver_expert(grid_size=10, num_actions=10, hidden_dim=128)

        assert expert.config.obs_dim == 10 * 10 * 2 + 5
        assert expert.config.action_dim == 10
        assert expert.config.hidden_dim == 128

    def test_discrete_actions(self):
        """Test that discrete_actions is True."""
        expert = create_arc_solver_expert()

        assert expert.config.discrete_actions is True
        assert expert.config.num_actions == 20
