"""
GEGLU Feed-Forward Network.

Gated Linear Unit with GELU activation.
Similar to SwiGLU but uses GELU instead of SiLU.

Architecture: (GELU(xW_gate) * xW_up) @ W_down

Reference: https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import FFNConfig


class GEGLU(nn.Module):
    """
    GEGLU Feed-Forward Network.

    Gated MLP with GELU activation:
    output = GELU(x @ W_gate) * (x @ W_up) @ W_down

    Args:
        config: FFN configuration

    Example:
        >>> config = FFNConfig(hidden_size=4096, intermediate_size=11008)
        >>> ffn = GEGLU(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = ffn(x)  # (2, 10, 4096)
    """

    def __init__(self, config: FFNConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate, up, and down projections
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            Output, shape (batch, seq_len, hidden_size)
        """
        # Compute gate and up projections
        gate = nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)

        # Gated output
        hidden = gate * up

        # Down projection
        return self.down_proj(hidden)


def create_geglu(
    hidden_size: int,
    intermediate_size: int,
    bias: bool = False,
) -> GEGLU:
    """
    Factory function for GEGLU.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        bias: Use bias in linear layers

    Returns:
        GEGLU instance
    """
    from ...core.enums import FFNType

    config = FFNConfig(
        ffn_type=FFNType.GEGLU,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=bias,
    )
    return GEGLU(config)
