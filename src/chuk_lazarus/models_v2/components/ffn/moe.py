"""
Mixture of Experts (MoE).

Routes tokens to a subset of expert MLPs based on learned routing.
Used by Mixtral, DeepSeek-MoE, Qwen-MoE.

Reference: https://arxiv.org/abs/2401.04088 (Mixtral)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import FFNConfig
from .swiglu import SwiGLU


class MoERouter(nn.Module):
    """
    Expert routing module.

    Routes each token to top-k experts based on learned weights.

    Args:
        hidden_size: Input dimension
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to use per token

    Example:
        >>> router = MoERouter(hidden_size=4096, num_experts=8, num_experts_per_tok=2)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> weights, indices = router(x)  # (2, 10, 2), (2, 10, 2)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compute routing weights and indices.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            weights: Routing weights, shape (batch, seq_len, num_experts_per_tok)
            indices: Expert indices, shape (batch, seq_len, num_experts_per_tok)
        """
        # Compute router logits: (batch, seq_len, num_experts)
        logits = self.gate(x)

        # Get top-k experts
        # Note: MLX doesn't have topk, so we use argsort
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok :]
        indices = indices[:, :, ::-1]  # Descending order

        # Get corresponding weights
        weights = mx.softmax(logits, axis=-1)

        # Gather top-k weights
        # Create indexing for gather
        # This is a workaround since MLX gather is different
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)

        # Renormalize weights to sum to 1
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices


class MoE(nn.Module):
    """
    Mixture of Experts layer.

    Contains multiple expert MLPs and a router that selects
    which experts to use for each token.

    Args:
        config: FFN configuration with MoE settings

    Example:
        >>> config = FFNConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     num_experts=8,
        ...     num_experts_per_tok=2,
        ... )
        >>> moe = MoE(config)
        >>> x = mx.random.normal((2, 10, 4096))
        >>> output = moe(x)  # (2, 10, 4096)
    """

    def __init__(self, config: FFNConfig):
        super().__init__()

        if config.num_experts is None:
            raise ValueError("num_experts must be set for MoE")
        if config.num_experts_per_tok is None:
            raise ValueError("num_experts_per_tok must be set for MoE")

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.router = MoERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
        )

        # Experts (each is a SwiGLU MLP)
        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.bias,
        )
        self.experts = [SwiGLU(expert_config) for _ in range(config.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MoE.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            Output, shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Get routing weights and indices
        weights, indices = self.router(x)

        # Flatten for processing
        x_flat = x.reshape(-1, hidden_size)  # (batch * seq_len, hidden_size)
        indices_flat = indices.reshape(-1, self.num_experts_per_tok)  # (batch * seq_len, k)
        weights_flat = weights.reshape(-1, self.num_experts_per_tok)  # (batch * seq_len, k)

        # Compute expert outputs
        output = mx.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            # Shape: (batch * seq_len, k) - bool mask
            expert_mask = indices_flat == expert_idx

            # Get weight for this expert per token: (batch * seq_len,)
            expert_weights = mx.sum(weights_flat * expert_mask.astype(weights_flat.dtype), axis=-1)

            # Only process if any tokens are routed to this expert
            if mx.any(expert_weights > 0):
                # Get expert output for all tokens
                expert_out = expert(x_flat)

                # Weight by routing weight
                output = output + expert_out * expert_weights[:, None]

        # Reshape back
        output = output.reshape(batch_size, seq_len, hidden_size)

        return output


def create_moe(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    bias: bool = False,
) -> MoE:
    """
    Factory function for MoE.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension per expert
        num_experts: Total number of experts
        num_experts_per_tok: Experts activated per token
        bias: Use bias in linear layers

    Returns:
        MoE instance
    """
    from ...core.enums import FFNType

    config = FFNConfig(
        ffn_type=FFNType.MOE,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=bias,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    return MoE(config)
