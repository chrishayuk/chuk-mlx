"""
LSTM-based Tiny Expert.

LSTM has an additional cell state for longer-term memory.
Use when tasks require remembering information over many steps.
"""

import mlx.core as mx
import mlx.nn as nn

from .rnn_expert_base import ExpertConfig, RNNExpertBase


class LSTMCell(nn.Module):
    """Single LSTM cell for MLX."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # All four gates in one projection (more efficient)
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def __call__(
        self, x: mx.array, state: tuple[mx.array, mx.array] | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, input_dim)
            state: Tuple of (h, c) where:
                h: Hidden state, shape (batch, hidden_dim)
                c: Cell state, shape (batch, hidden_dim)

        Returns:
            new_h: New hidden state
            (new_h, new_c): New state tuple
        """
        batch_size = x.shape[0]

        if state is None:
            h = mx.zeros((batch_size, self.hidden_dim))
            c = mx.zeros((batch_size, self.hidden_dim))
        else:
            h, c = state

        # Concatenate input and hidden
        combined = mx.concatenate([x, h], axis=-1)

        # Compute all gates at once
        gates = self.gates(combined)

        # Split into individual gates
        i, f, g, o = mx.split(gates, 4, axis=-1)

        # Apply activations
        i = mx.sigmoid(i)  # Input gate
        f = mx.sigmoid(f)  # Forget gate
        g = mx.tanh(g)  # Cell gate (candidate)
        o = mx.sigmoid(o)  # Output gate

        # Update cell state
        new_c = f * c + i * g

        # Compute new hidden state
        new_h = o * mx.tanh(new_c)

        return new_h, (new_h, new_c)


class LSTMExpert(RNNExpertBase):
    """
    LSTM-based expert for tasks requiring longer memory.

    The cell state provides a "highway" for gradients and
    allows remembering information over many timesteps.

    Good for:
        - Multi-step planning
        - Tasks with delayed rewards
        - Remembering constraints across many actions
    """

    def _build_rnn_layers(self):
        """Build LSTM layers."""
        self.lstm_layers = []
        for i in range(self.config.num_layers):
            input_dim = self.config.hidden_dim
            self.lstm_layers.append(LSTMCell(input_dim, self.config.hidden_dim))

        self._lstm_modules = self.lstm_layers

    def _forward_rnn(self, x: mx.array, hidden: list | None = None) -> tuple[mx.array, list]:
        """
        Forward through LSTM layers.

        Args:
            x: Input after projection, shape (batch, hidden_dim)
            hidden: List of (h, c) tuples per layer

        Returns:
            output: Final layer output (h)
            new_hidden: List of new (h, c) state tuples
        """
        if hidden is None:
            hidden = [None] * self.config.num_layers

        new_hidden = []
        current = x

        for i, lstm_cell in enumerate(self.lstm_layers):
            state = hidden[i] if i < len(hidden) else None
            current, new_state = lstm_cell(current, state)
            new_hidden.append(new_state)

            # Dropout between layers
            if self.config.dropout > 0 and i < self.config.num_layers - 1:
                current = nn.Dropout(self.config.dropout)(current)

        return current, new_hidden


def create_planning_expert(
    state_dim: int = 20, action_dim: int = 5, hidden_dim: int = 128
) -> LSTMExpert:
    """
    Create an LSTM expert for multi-step planning tasks.

    LSTM is better here because planning requires:
    - Remembering the goal over many steps
    - Tracking which sub-goals have been achieved
    - Maintaining constraints across decisions
    """
    config = ExpertConfig(
        name="planner",
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=3,  # Deeper for planning
        discrete_actions=False,
        use_value_head=True,
    )
    return LSTMExpert(config)


def create_arc_solver_expert(
    grid_size: int = 30,
    num_actions: int = 20,  # Transform types
    hidden_dim: int = 256,
) -> LSTMExpert:
    """
    Create an LSTM expert for ARC-style grid puzzles.

    Observations: flattened grid + goal grid + step count
    Actions: discrete transformation selection
    """
    config = ExpertConfig(
        name="arc_solver",
        obs_dim=grid_size * grid_size * 2 + 5,  # current + goal + metadata
        action_dim=num_actions,
        hidden_dim=hidden_dim,
        num_layers=3,
        discrete_actions=True,
        num_actions=num_actions,
        use_value_head=True,
    )
    return LSTMExpert(config)
