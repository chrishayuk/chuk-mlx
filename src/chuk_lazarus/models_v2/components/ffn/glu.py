"""
Gated Linear Unit (GLU) Feed-Forward Network.

Gated MLP with configurable activation function.
Supports SwiGLU (SiLU), GEGLU (GELU), and other gated variants.

Architecture: (Activation(xW_gate) * xW_up) @ W_down

Reference: https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import FFNConfig
from ...core.enums import ActivationType


class GLU(nn.Module):
    """
    Gated Linear Unit Feed-Forward Network.

    Gated MLP with configurable activation:
    output = activation(x @ W_gate) * (x @ W_up) @ W_down

    Common variants:
    - SwiGLU: activation=SILU (used by Llama, Mistral)
    - GEGLU: activation=GELU
    - ReGLU: activation=RELU

    Args:
        config: FFN configuration

    Example:
        >>> # SwiGLU (Llama-style)
        >>> config = FFNConfig(hidden_size=4096, intermediate_size=11008, activation=ActivationType.SILU)
        >>> ffn = GLU(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = ffn(x)  # (2, 10, 4096)

        >>> # GEGLU
        >>> config = FFNConfig(hidden_size=4096, intermediate_size=11008, activation=ActivationType.GELU)
        >>> ffn = GLU(config)
    """

    def __init__(self, config: FFNConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate, up, and down projections
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

        # Activation function for gating
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, activation: ActivationType):
        """Get activation function."""
        activations = {
            ActivationType.SILU: nn.silu,
            ActivationType.GELU: nn.gelu,
            ActivationType.GELU_APPROX: nn.gelu_approx,
            ActivationType.GELU_TANH: nn.gelu_approx,
            ActivationType.RELU: nn.relu,
            ActivationType.TANH: mx.tanh,
            ActivationType.SIGMOID: mx.sigmoid,
            ActivationType.NONE: lambda x: x,
        }
        return activations.get(activation, nn.silu)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            Output, shape (batch, seq_len, hidden_size)
        """
        # Compute gate and up projections
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)

        # Gated output
        hidden = gate * up

        # Down projection
        return self.down_proj(hidden)


# Convenience aliases
SwiGLU = GLU  # SwiGLU is just GLU with SILU activation (default)
GEGLU = GLU  # GEGLU is just GLU with GELU activation


def create_glu(
    hidden_size: int,
    intermediate_size: int,
    activation: ActivationType = ActivationType.SILU,
    bias: bool = False,
) -> GLU:
    """
    Factory function for GLU.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        activation: Activation function (SILU for SwiGLU, GELU for GEGLU)
        bias: Use bias in linear layers

    Returns:
        GLU instance
    """
    config = FFNConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        bias=bias,
    )
    return GLU(config)


def create_swiglu(
    hidden_size: int,
    intermediate_size: int,
    bias: bool = False,
) -> GLU:
    """Create SwiGLU (GLU with SiLU activation)."""
    return create_glu(hidden_size, intermediate_size, ActivationType.SILU, bias)


def create_geglu(
    hidden_size: int,
    intermediate_size: int,
    bias: bool = False,
) -> GLU:
    """Create GEGLU (GLU with GELU activation)."""
    return create_glu(hidden_size, intermediate_size, ActivationType.GELU, bias)
