"""
GRU-based Tiny Expert.

GRU is simpler than LSTM (2 gates vs 3) and often works just as well.
Good default choice for control tasks.
"""

import mlx.core as mx
import mlx.nn as nn

from .rnn_expert_base import ExpertConfig, RNNExpertBase


class GRUCell(nn.Module):
    """Single GRU cell for MLX."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Reset gate
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Update gate
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Candidate hidden
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def __call__(self, x: mx.array, h: mx.array | None = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, input_dim)
            h: Hidden state, shape (batch, hidden_dim)

        Returns:
            new_h: New hidden state, shape (batch, hidden_dim)
        """
        batch_size = x.shape[0]

        if h is None:
            h = mx.zeros((batch_size, self.hidden_dim))
        elif h.shape[0] != batch_size:
            # Expand hidden state to match batch size
            h = mx.broadcast_to(h, (batch_size, self.hidden_dim))

        # Concatenate input and hidden
        combined = mx.concatenate([x, h], axis=-1)

        # Gates
        r = mx.sigmoid(self.W_r(combined))  # Reset gate
        z = mx.sigmoid(self.W_z(combined))  # Update gate

        # Candidate hidden (with reset gate applied)
        combined_reset = mx.concatenate([x, r * h], axis=-1)
        h_tilde = mx.tanh(self.W_h(combined_reset))

        # New hidden state
        new_h = (1 - z) * h + z * h_tilde

        return new_h


class GRUExpert(RNNExpertBase):
    """
    GRU-based expert for control tasks.

    Example usage:
        config = ExpertConfig(
            name="physics_controller",
            obs_dim=10,  # goal, error, attempts, wind, etc.
            action_dim=2,  # angle, velocity
            hidden_dim=64,
            num_layers=2,
        )
        expert = GRUExpert(config)

        # Reset at episode start
        expert.reset_hidden(batch_size=1)

        # Step through environment
        obs = mx.array([[target_dist, error, attempts, wind_x, wind_y, ...]])
        result = expert(obs)
        action = result["action"]  # e.g., [angle, velocity]
    """

    def _build_rnn_layers(self):
        """Build GRU layers."""
        self.gru_layers = []
        for i in range(self.config.num_layers):
            input_dim = self.config.hidden_dim  # After input projection
            self.gru_layers.append(GRUCell(input_dim, self.config.hidden_dim))

        # Register as module list for parameter tracking
        self._gru_modules = self.gru_layers

    def _forward_rnn(self, x: mx.array, hidden: list | None = None) -> tuple[mx.array, list]:
        """
        Forward through GRU layers.

        Args:
            x: Input after projection, shape (batch, hidden_dim)
            hidden: List of hidden states per layer

        Returns:
            output: Final layer output
            new_hidden: List of new hidden states
        """
        if hidden is None:
            hidden = [None] * self.config.num_layers

        # Ensure hidden is a list
        if not isinstance(hidden, list):
            hidden = [hidden] + [None] * (self.config.num_layers - 1)

        new_hidden = []
        current = x

        for i, gru_cell in enumerate(self.gru_layers):
            h = hidden[i] if i < len(hidden) else None
            current = gru_cell(current, h)
            new_hidden.append(current)

            # Optional dropout between layers (not on last)
            if self.config.dropout > 0 and i < self.config.num_layers - 1:
                current = nn.Dropout(self.config.dropout)(current)

        return current, new_hidden


def create_physics_controller(
    obs_dim: int = 10, action_dim: int = 2, hidden_dim: int = 64
) -> GRUExpert:
    """
    Create a GRU expert for physics control tasks.

    Default observation space:
        - target_distance
        - current_error
        - attempts_remaining
        - wind_x, wind_y
        - last_angle, last_velocity
        - constraint flags (3)

    Default action space:
        - angle (continuous)
        - velocity (continuous)
    """
    config = ExpertConfig(
        name="physics_controller",
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        discrete_actions=False,
        action_low=-1.0,
        action_high=1.0,
        use_value_head=True,
    )
    return GRUExpert(config)


def create_scheduler_expert(num_tasks: int = 10, hidden_dim: int = 128) -> GRUExpert:
    """
    Create a GRU expert for scheduling optimization.

    Observation: task features, constraints, current schedule quality
    Action: which task to move / how to adjust
    """
    config = ExpertConfig(
        name="scheduler",
        obs_dim=num_tasks * 3 + 5,  # task features + global state
        action_dim=num_tasks + 2,  # task selection + adjustment
        hidden_dim=hidden_dim,
        num_layers=2,
        discrete_actions=False,
        use_value_head=True,
    )
    return GRUExpert(config)
