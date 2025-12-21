"""MLP (Feed-Forward) layers."""

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig


class MLP(nn.Module):
    """
    Standard MLP with gated activation (SwiGLU style).

    Used in Llama, Mistral, and similar architectures.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

        # Activation function
        self.act_fn = self._get_activation(config.hidden_act)

    def _get_activation(self, act_name: str):
        """Get activation function by name."""
        activations = {
            "silu": nn.silu,
            "swish": nn.silu,
            "gelu": nn.gelu,
            "gelu_new": nn.gelu,
            "relu": nn.relu,
            "gelu_pytorch_tanh": lambda x: nn.gelu(x, approx="tanh"),
        }
        return activations.get(act_name, nn.silu)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with gated activation."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GeLUMLP(nn.Module):
    """
    MLP with GELU activation (no gating).

    Used in some older architectures.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))
