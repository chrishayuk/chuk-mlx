"""
Mamba layer.

Full Mamba block combining convolution, selective SSM, and gating.
This is the complete layer that can replace attention in a transformer.

Reference: https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import SSMConfig
from .selective_ssm import SelectiveSSM


class Mamba(nn.Module):
    """
    Complete Mamba layer.

    Combines:
    1. Input projection with gating
    2. Depthwise convolution for local context
    3. Selective SSM for long-range dependencies
    4. Output projection

    Unlike attention, Mamba has O(n) complexity and constant memory
    during inference.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        bias: Use bias in linear layers
        conv_bias: Use bias in conv layer

    Example:
        >>> mamba = Mamba(d_model=768, d_state=16, d_conv=4)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y, cache = mamba(x)  # y: (2, 100, 768)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # The core selective SSM does most of the work
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=bias,
            conv_bias=conv_bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            cache: Optional (conv_state, ssm_state) for inference

        Returns:
            - Output, shape (batch, seq_len, d_model)
            - Updated cache if input cache was provided
        """
        return self.ssm(x, cache)

    @classmethod
    def from_config(cls, config: SSMConfig, d_model: int) -> Mamba:
        """Create Mamba layer from SSMConfig."""
        return cls(
            d_model=d_model,
            d_state=config.state_size,
            d_conv=config.conv_kernel_size,
            expand=config.expand_factor,
        )

    def init_cache(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """
        Initialize cache for inference.

        Args:
            batch_size: Batch size

        Returns:
            (conv_state, ssm_state) tuple
        """
        d_inner = self.d_model * self.expand

        # Conv state: last d_conv-1 inputs
        conv_state = mx.zeros((batch_size, d_inner, self.d_conv - 1))

        # SSM state: (batch, d_inner, d_state)
        ssm_state = mx.zeros((batch_size, d_inner, self.d_state))

        return conv_state, ssm_state


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection and normalization.

    This is the standard block used in a Mamba model:
        output = x + Mamba(norm(x))

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        norm_eps: Epsilon for RMSNorm

    Example:
        >>> block = MambaBlock(d_model=768)
        >>> x = mx.random.normal((2, 100, 768))
        >>> y, cache = block(x)  # y: (2, 100, 768)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.d_model = d_model

        # Pre-norm
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)

        # Mamba layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Forward pass with residual.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            cache: Optional cache for inference

        Returns:
            - Output with residual, same shape as input
            - Updated cache
        """
        # Pre-norm + Mamba + residual
        residual = x
        x = self.norm(x)
        x, new_cache = self.mamba(x, cache)
        x = residual + x

        return x, new_cache

    def init_cache(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Initialize cache for inference."""
        return self.mamba.init_cache(batch_size)


def create_mamba(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
) -> Mamba:
    """
    Factory function for Mamba layer.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor

    Returns:
        Mamba layer instance
    """
    return Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


def create_mamba_block(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    norm_eps: float = 1e-5,
) -> MambaBlock:
    """
    Factory function for MambaBlock.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        norm_eps: Normalization epsilon

    Returns:
        MambaBlock instance
    """
    return MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        norm_eps=norm_eps,
    )
