"""
Llama 4 Mixture of Experts (MoE).

Llama 4 uses a sparse MoE architecture with:
- A shared expert that is always active
- Routed experts selected by top-k routing
- Sigmoid-based router scores

This implementation uses MLX's mx.gather_mm for efficient sparse computation,
which is critical for memory efficiency with large numbers of experts.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4
"""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .config import Llama4TextConfig


class Llama4MLP(nn.Module):
    """
    SwiGLU MLP for Llama 4 shared expert.

    Same as standard Llama MLP but parameterized for MoE use.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through SwiGLU."""
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SwitchLinear(nn.Module):
    """
    Linear layer with expert selection using mx.gather_mm.

    Stores weights for all experts in a single tensor and uses
    MLX's gather_mm for efficient sparse computation.

    Weight shape: (num_experts, output_dims, input_dims)
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()

        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self) -> int:
        return self.weight.shape[2]

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """
        Forward pass with expert selection.

        Args:
            x: Input tensor, shape (..., 1, 1, input_dims)
            indices: Expert indices, shape (..., k) where k is num_experts_per_tok

        Returns:
            Output tensor, shape (..., k, 1, output_dims)
        """
        # mx.gather_mm: efficient batched matmul with index selection
        # x @ weight[indices].T
        out = mx.gather_mm(
            x,
            self.weight.swapaxes(-1, -2),  # (experts, input, output)
            rhs_indices=indices,
        )

        if "bias" in self:
            # Add bias for selected experts
            out = out + mx.expand_dims(self.bias[indices], -2)

        return out


@partial(mx.compile, shapeless=True)
def swiglu(x: mx.array, gate: mx.array) -> mx.array:
    """SwiGLU activation: silu(gate) * x"""
    return nn.silu(gate) * x


class SwitchGLU(nn.Module):
    """
    SwiGLU MLP with expert selection using mx.gather_mm.

    This is the efficient implementation for routed experts that uses
    fused weight tensors and MLX's native sparse matmul.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """
        Forward pass with expert selection.

        Args:
            x: Input tensor, shape (batch * seq, hidden_size) after pre-weighting
            indices: Expert indices, shape (batch * seq, k)

        Returns:
            Output tensor, shape (batch * seq, k, hidden_size)
        """
        # Expand dims for gather_mm: (..., 1, 1, hidden)
        x = mx.expand_dims(x, (-2, -3))

        # Compute gate and up projections for selected experts
        x_gate = self.gate_proj(x, indices)  # (..., k, 1, intermediate)
        x_up = self.up_proj(x, indices)  # (..., k, 1, intermediate)

        # Apply SwiGLU activation
        x = swiglu(x_up, x_gate)

        # Down projection back to hidden size
        x = self.down_proj(x, indices)  # (..., k, 1, hidden)

        return x.squeeze(-2)  # (..., k, hidden)


class Llama4MoE(nn.Module):
    """
    Llama 4 Mixture of Experts layer using efficient gather_mm.

    Key features:
    - Shared expert: Always active for all tokens (standard MLP)
    - Routed experts: Sparsely activated via top-k routing using SwitchGLU
    - Sigmoid router with top-k selection
    - Uses mx.gather_mm for efficient sparse computation

    Args:
        config: Llama 4 text configuration

    Example:
        >>> config = Llama4TextConfig.tiny()
        >>> moe = Llama4MoE(config)
        >>> x = mx.random.normal((2, 10, 64))
        >>> output = moe(x)  # Shape: (2, 10, 64)
    """

    def __init__(self, config: Llama4TextConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size  # For shared expert
        self.intermediate_size_mlp = config.intermediate_size_mlp  # For routed experts
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        assert self.num_experts_per_tok == 1, "Only 1 expert per token currently supported"

        # Router: projects to num_experts scores
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

        # Shared expert (always active) - standard MLP
        self.shared_expert = Llama4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        # Routed experts using efficient SwitchGLU with gather_mm
        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size_mlp,
            num_experts=config.num_local_experts,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MoE.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor, shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # 1. Compute shared expert output (always active)
        shared_output = self.shared_expert(x)

        # 2. Compute router scores and select top-k experts
        router_logits = self.router(x)  # (batch, seq_len, num_experts)

        # Use sigmoid for scores (Llama 4 style)
        # Get top-k expert indices
        k = self.num_experts_per_tok
        indices = mx.argpartition(-router_logits, kth=k - 1, axis=-1)[..., :k]

        # Gather scores for selected experts and apply sigmoid
        scores = mx.take_along_axis(router_logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

        # 3. Compute routed expert outputs using SwitchGLU
        # Pre-weight input by routing scores
        x_weighted = x * scores  # (batch, seq, hidden) broadcasted with (batch, seq, k)

        # Reshape for expert computation
        x_flat = x_weighted.reshape(-1, hidden_size)  # (batch * seq, hidden)
        indices_flat = indices.reshape(-1, k)  # (batch * seq, k)

        # Compute routed expert output
        routed_output = self.experts(x_flat, indices_flat)  # (batch * seq, k, hidden)

        # Sum over k dimension and reshape back
        routed_output = routed_output.squeeze(1)  # (batch * seq, hidden) for k=1
        routed_output = routed_output.reshape(batch_size, seq_len, hidden_size)

        # 4. Combine shared and routed outputs
        return shared_output + routed_output


def create_llama4_moe(config: Llama4TextConfig) -> nn.Module:
    """
    Factory function for Llama 4 MoE.

    Args:
        config: Llama 4 text configuration

    Returns:
        MoE module using efficient gather_mm implementation
    """
    return Llama4MoE(config)
