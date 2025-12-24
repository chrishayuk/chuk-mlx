"""
Gated Recurrent Unit (GRU).

Simplified version of LSTM with fewer gates.
Often faster with similar performance.

Reference: https://arxiv.org/abs/1406.1078
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GRUCell(nn.Module):
    """
    Single GRU cell.

    Processes one timestep and returns updated hidden state.

    Equations:
        z = sigmoid(W_z @ [h, x] + b_z)     # update gate
        r = sigmoid(W_r @ [h, x] + b_r)     # reset gate
        h_tilde = tanh(W_h @ [r*h, x] + b_h)  # candidate
        h' = (1 - z) * h + z * h_tilde       # new hidden state

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        bias: Whether to use bias

    Example:
        >>> cell = GRUCell(input_size=128, hidden_size=256)
        >>> x = mx.random.normal((32, 128))
        >>> h = mx.zeros((32, 256))
        >>> h_new = cell(x, h)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined projection for z, r gates
        self.W_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        h: mx.array,
    ) -> mx.array:
        """
        Forward pass for single timestep.

        Args:
            x: Input, shape (batch, input_size)
            h: Hidden state, shape (batch, hidden_size)

        Returns:
            New hidden state, shape (batch, hidden_size)
        """
        # Compute gates together
        gates_i = self.W_ih(x)  # (batch, 3 * hidden_size)
        gates_h = self.W_hh(h)

        # Split input projections
        z_i, r_i, n_i = mx.split(gates_i, 3, axis=-1)
        z_h, r_h, n_h = mx.split(gates_h, 3, axis=-1)

        # Update and reset gates
        z = mx.sigmoid(z_i + z_h)
        r = mx.sigmoid(r_i + r_h)

        # Candidate (with reset gate applied to hidden)
        n = mx.tanh(n_i + r * n_h)

        # New hidden state
        h_new = (1 - z) * h + z * n

        return h_new

    def init_state(self, batch_size: int) -> mx.array:
        """Initialize hidden state to zeros."""
        return mx.zeros((batch_size, self.hidden_size))


class GRU(nn.Module):
    """
    Full GRU layer processing sequences.

    Wraps GRUCell to process complete sequences.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of stacked GRU layers
        bias: Whether to use bias
        dropout: Dropout between layers (if num_layers > 1)
        bidirectional: Use bidirectional GRU

    Example:
        >>> gru = GRU(input_size=128, hidden_size=256, num_layers=2)
        >>> x = mx.random.normal((32, 100, 128))
        >>> output, h = gru(x)  # output: (32, 100, 256)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create cells for each layer
        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(GRUCell(layer_input_size, hidden_size, bias=bias))

        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def __call__(
        self,
        x: mx.array,
        h: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass through GRU.

        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h: Optional initial hidden state, shape
               (num_layers * num_directions, batch, hidden_size)

        Returns:
            - Output, shape (batch, seq_len, hidden_size * num_directions)
            - Final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state if not provided
        if h is None:
            h = self.init_state(batch_size)

        h_list = list(mx.split(h, self.num_layers * self.num_directions, axis=0))
        h_list = [hi.squeeze(0) for hi in h_list]

        output = x

        for layer in range(self.num_layers):
            # Forward direction
            cell_idx = layer * self.num_directions
            cell = self.cells[cell_idx]
            h_state = h_list[cell_idx]

            forward_outputs = []
            for t in range(seq_len):
                h_state = cell(output[:, t], h_state)
                forward_outputs.append(h_state)

            h_list[cell_idx] = h_state
            forward_out = mx.stack(forward_outputs, axis=1)

            if self.bidirectional:
                # Backward direction
                cell_idx = layer * self.num_directions + 1
                cell = self.cells[cell_idx]
                h_state = h_list[cell_idx]

                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h_state = cell(output[:, t], h_state)
                    backward_outputs.append(h_state)

                h_list[cell_idx] = h_state
                backward_out = mx.stack(backward_outputs[::-1], axis=1)

                output = mx.concatenate([forward_out, backward_out], axis=-1)
            else:
                output = forward_out

            # Apply dropout between layers
            if self.dropout is not None and layer < self.num_layers - 1:
                output = self.dropout(output)

        # Stack final states
        h_n = mx.stack(h_list, axis=0)

        return output, h_n

    def init_state(self, batch_size: int) -> mx.array:
        """Initialize states for all layers."""
        num_states = self.num_layers * self.num_directions
        return mx.zeros((num_states, batch_size, self.hidden_size))


def create_gru(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
) -> GRU:
    """Factory function for GRU."""
    return GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
