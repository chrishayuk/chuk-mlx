"""
Standard MLP (Multi-Layer Perceptron).

Two-layer feed-forward network with configurable activation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import FFNConfig
from ...core.enums import ActivationType


class MLP(nn.Module):
    """
    Standard two-layer MLP.

    Architecture: Linear -> Activation -> Linear

    Args:
        config: FFN configuration

    Example:
        >>> config = FFNConfig(hidden_size=4096, intermediate_size=11008, activation=ActivationType.GELU)
        >>> mlp = MLP(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = mlp(x)  # (2, 10, 4096)
    """

    def __init__(self, config: FFNConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Projections
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

        # Activation
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, activation: ActivationType):
        """Get activation function."""
        activations = {
            ActivationType.RELU: nn.relu,
            ActivationType.GELU: nn.gelu,
            ActivationType.GELU_APPROX: nn.gelu_approx,
            ActivationType.SILU: nn.silu,
            ActivationType.TANH: mx.tanh,
            ActivationType.SIGMOID: mx.sigmoid,
            ActivationType.NONE: lambda x: x,
        }
        return activations.get(activation, nn.gelu)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            Output, shape (batch, seq_len, hidden_size)
        """
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


def create_mlp(
    hidden_size: int,
    intermediate_size: int,
    activation: ActivationType = ActivationType.GELU,
    bias: bool = False,
) -> MLP:
    """
    Factory function for MLP.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        activation: Activation function
        bias: Use bias in linear layers

    Returns:
        MLP instance
    """
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        bias=bias,
    )
    return MLP(config)
