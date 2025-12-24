"""
Long Short-Term Memory (LSTM).

Classic recurrent architecture with forget, input, and output gates.
Good for learning long-range dependencies.

Reference: https://www.bioinf.jku.at/publications/older/2604.pdf
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LSTMCell(nn.Module):
    """
    Single LSTM cell.

    Processes one timestep and returns updated hidden/cell states.

    Equations:
        f = sigmoid(W_f @ [h, x] + b_f)     # forget gate
        i = sigmoid(W_i @ [h, x] + b_i)     # input gate
        g = tanh(W_g @ [h, x] + b_g)        # candidate cell
        o = sigmoid(W_o @ [h, x] + b_o)     # output gate
        c' = f * c + i * g                   # new cell state
        h' = o * tanh(c')                    # new hidden state

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        bias: Whether to use bias

    Example:
        >>> cell = LSTMCell(input_size=128, hidden_size=256)
        >>> x = mx.random.normal((32, 128))
        >>> h = mx.zeros((32, 256))
        >>> c = mx.zeros((32, 256))
        >>> h_new, c_new = cell(x, (h, c))
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

        # Combined projection for efficiency: [f, i, g, o]
        # Input projection
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        # Hidden projection
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass for single timestep.

        Args:
            x: Input, shape (batch, input_size)
            state: (hidden, cell) tuple, each shape (batch, hidden_size)

        Returns:
            (new_hidden, new_cell) tuple
        """
        h, c = state

        # Combined gates computation
        gates = self.W_ih(x) + self.W_hh(h)  # (batch, 4 * hidden_size)

        # Split into individual gates
        f, i, g, o = mx.split(gates, 4, axis=-1)

        # Apply activations
        f = mx.sigmoid(f)  # forget gate
        i = mx.sigmoid(i)  # input gate
        g = mx.tanh(g)  # candidate
        o = mx.sigmoid(o)  # output gate

        # Update cell state
        c_new = f * c + i * g

        # Update hidden state
        h_new = o * mx.tanh(c_new)

        return h_new, c_new

    def init_state(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Initialize hidden and cell states to zeros."""
        h = mx.zeros((batch_size, self.hidden_size))
        c = mx.zeros((batch_size, self.hidden_size))
        return h, c


class LSTM(nn.Module):
    """
    Full LSTM layer processing sequences.

    Wraps LSTMCell to process complete sequences.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of stacked LSTM layers
        bias: Whether to use bias
        dropout: Dropout between layers (if num_layers > 1)
        bidirectional: Use bidirectional LSTM

    Example:
        >>> lstm = LSTM(input_size=128, hidden_size=256, num_layers=2)
        >>> x = mx.random.normal((32, 100, 128))
        >>> output, (h, c) = lstm(x)  # output: (32, 100, 256)
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
                self.cells.append(LSTMCell(layer_input_size, hidden_size, bias=bias))

        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def __call__(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass through LSTM.

        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            state: Optional initial (h, c), each shape
                   (num_layers * num_directions, batch, hidden_size)

        Returns:
            - Output, shape (batch, seq_len, hidden_size * num_directions)
            - Final (h, c) states
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state if not provided
        if state is None:
            state = self.init_state(batch_size)

        h_n, c_n = state
        h_list = list(mx.split(h_n, self.num_layers * self.num_directions, axis=0))
        c_list = list(mx.split(c_n, self.num_layers * self.num_directions, axis=0))

        # Squeeze the split dimension
        h_list = [h.squeeze(0) for h in h_list]
        c_list = [c.squeeze(0) for c in c_list]

        output = x

        for layer in range(self.num_layers):
            # Forward direction
            cell_idx = layer * self.num_directions
            cell = self.cells[cell_idx]
            h, c = h_list[cell_idx], c_list[cell_idx]

            forward_outputs = []
            for t in range(seq_len):
                h, c = cell(output[:, t], (h, c))
                forward_outputs.append(h)

            h_list[cell_idx] = h
            c_list[cell_idx] = c
            forward_out = mx.stack(forward_outputs, axis=1)

            if self.bidirectional:
                # Backward direction
                cell_idx = layer * self.num_directions + 1
                cell = self.cells[cell_idx]
                h, c = h_list[cell_idx], c_list[cell_idx]

                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h, c = cell(output[:, t], (h, c))
                    backward_outputs.append(h)

                h_list[cell_idx] = h
                c_list[cell_idx] = c
                backward_out = mx.stack(backward_outputs[::-1], axis=1)

                output = mx.concatenate([forward_out, backward_out], axis=-1)
            else:
                output = forward_out

            # Apply dropout between layers
            if self.dropout is not None and layer < self.num_layers - 1:
                output = self.dropout(output)

        # Stack final states
        h_n = mx.stack(h_list, axis=0)
        c_n = mx.stack(c_list, axis=0)

        return output, (h_n, c_n)

    def init_state(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Initialize states for all layers."""
        num_states = self.num_layers * self.num_directions
        h = mx.zeros((num_states, batch_size, self.hidden_size))
        c = mx.zeros((num_states, batch_size, self.hidden_size))
        return h, c


def create_lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bidirectional: bool = False,
) -> LSTM:
    """Factory function for LSTM."""
    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
