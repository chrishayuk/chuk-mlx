"""
Minimal GRU (minGRU).

Simplified GRU with only one gate, making it faster while
maintaining good performance. Good for efficiency-focused use cases.

This is a minimalist variant that can be parallelized more easily.

Reference: Inspired by "Simplifying Recurrent Neural Networks"
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MinGRUCell(nn.Module):
    """
    Minimal GRU cell with single gate.

    Simplified equation:
        z = sigmoid(W_z @ x + U_z @ h + b_z)  # single gate
        h' = (1 - z) * h + z * tanh(W_h @ x)  # no reset gate

    This removes the reset gate and simplifies computation.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        bias: Whether to use bias

    Example:
        >>> cell = MinGRUCell(input_size=128, hidden_size=256)
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

        # Gate projection (z)
        self.W_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate projection
        self.W_h = nn.Linear(input_size, hidden_size, bias=bias)

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
        # Single update gate
        z = mx.sigmoid(self.W_z(x) + self.U_z(h))

        # Candidate (no reset gate)
        h_tilde = mx.tanh(self.W_h(x))

        # Update hidden state
        h_new = (1 - z) * h + z * h_tilde

        return h_new

    def init_state(self, batch_size: int) -> mx.array:
        """Initialize hidden state to zeros."""
        return mx.zeros((batch_size, self.hidden_size))


class MinGRU(nn.Module):
    """
    Full MinGRU layer processing sequences.

    Can be run in parallel mode for training (all timesteps at once)
    or sequential mode for inference (one timestep at a time).

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of stacked layers
        bias: Whether to use bias
        dropout: Dropout between layers

    Example:
        >>> mingru = MinGRU(input_size=128, hidden_size=256)
        >>> x = mx.random.normal((32, 100, 128))
        >>> output, h = mingru(x)  # output: (32, 100, 256)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create cells for each layer
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(MinGRUCell(layer_input_size, hidden_size, bias=bias))

        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def __call__(
        self,
        x: mx.array,
        h: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass through MinGRU.

        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h: Optional initial hidden state, shape (num_layers, batch, hidden_size)

        Returns:
            - Output, shape (batch, seq_len, hidden_size)
            - Final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state if not provided
        if h is None:
            h = self.init_state(batch_size)

        h_list = list(mx.split(h, self.num_layers, axis=0))
        h_list = [hi.squeeze(0) for hi in h_list]

        output = x

        for layer in range(self.num_layers):
            cell = self.cells[layer]
            h_state = h_list[layer]

            layer_outputs = []
            for t in range(seq_len):
                h_state = cell(output[:, t], h_state)
                layer_outputs.append(h_state)

            h_list[layer] = h_state
            output = mx.stack(layer_outputs, axis=1)

            # Apply dropout between layers
            if self.dropout is not None and layer < self.num_layers - 1:
                output = self.dropout(output)

        # Stack final states
        h_n = mx.stack(h_list, axis=0)

        return output, h_n

    def parallel_forward(
        self,
        x: mx.array,
        h: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Parallel forward pass (experimental).

        For training, we can compute all gates in parallel and then
        use a parallel scan. This is an experimental optimization.

        Args:
            x: Input sequence, shape (batch, seq_len, input_size)
            h: Optional initial hidden state

        Returns:
            - Output, shape (batch, seq_len, hidden_size)
            - Final hidden state
        """
        # For now, fall back to sequential
        # A true parallel implementation would use associative scan
        return self(x, h)

    def init_state(self, batch_size: int) -> mx.array:
        """Initialize states for all layers."""
        return mx.zeros((self.num_layers, batch_size, self.hidden_size))


class MinGRUBlock(nn.Module):
    """
    MinGRU block with residual connection and normalization.

    Standard block for use in sequence models:
        output = x + MinGRU(norm(x))

    Args:
        d_model: Model dimension
        num_layers: Number of MinGRU layers
        norm_eps: Epsilon for RMSNorm

    Example:
        >>> block = MinGRUBlock(d_model=768)
        >>> x = mx.random.normal((32, 100, 768))
        >>> y, h = block(x)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.d_model = d_model

        # Pre-norm
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)

        # MinGRU
        self.mingru = MinGRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
        )

    def __call__(
        self,
        x: mx.array,
        h: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass with residual.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            h: Optional hidden state

        Returns:
            - Output with residual
            - New hidden state
        """
        residual = x
        x = self.norm(x)
        x, h_new = self.mingru(x, h)
        x = residual + x

        return x, h_new

    def init_state(self, batch_size: int) -> mx.array:
        """Initialize state."""
        return self.mingru.init_state(batch_size)


def create_mingru(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
) -> MinGRU:
    """Factory function for MinGRU."""
    return MinGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


def create_mingru_block(
    d_model: int,
    num_layers: int = 1,
    norm_eps: float = 1e-5,
) -> MinGRUBlock:
    """Factory function for MinGRUBlock."""
    return MinGRUBlock(
        d_model=d_model,
        num_layers=num_layers,
        norm_eps=norm_eps,
    )
