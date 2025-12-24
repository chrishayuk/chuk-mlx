"""
Mamba block wrapper.

Wraps the Mamba SSM component into the Block interface.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..components.normalization import RMSNorm
from ..components.ssm import Mamba
from ..components.ssm import MambaBlock as MambaBlockCore
from ..core.config import SSMConfig
from ..core.enums import BlockType
from .base import Block, BlockOutput


class MambaBlockWrapper(Block):
    """
    Mamba block with Block interface.

    Wraps the core MambaBlock to provide the standard Block interface
    for use in backbones.

    Architecture:
        x -> norm -> Mamba (conv + SSM) -> + residual -> output

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        norm_eps: Epsilon for RMSNorm

    Example:
        >>> block = MambaBlockWrapper(d_model=768, d_state=16)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output = block(x)
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

        self._hidden_size = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Core Mamba block with norm and residual
        self.mamba_block = MambaBlockCore(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            norm_eps=norm_eps,
        )

    @property
    def block_type(self) -> BlockType:
        return BlockType.MAMBA

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,  # Mamba doesn't use masks
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> BlockOutput:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            mask: Ignored (Mamba is inherently causal)
            cache: Optional (conv_state, ssm_state) for inference

        Returns:
            BlockOutput with hidden states and updated cache
        """
        hidden_states, new_cache = self.mamba_block(x, cache)
        return BlockOutput(hidden_states=hidden_states, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,  # Not used for Mamba
    ) -> tuple[mx.array, mx.array]:
        """Initialize Mamba cache."""
        return self.mamba_block.init_cache(batch_size)

    @classmethod
    def from_config(cls, config: SSMConfig, d_model: int) -> MambaBlockWrapper:
        """Create from SSMConfig."""
        return cls(
            d_model=d_model,
            d_state=config.state_size,
            d_conv=config.conv_kernel,
            expand=config.expand,
        )


class MambaWithFFN(Block):
    """
    Mamba block with additional FFN layer.

    Some architectures add an FFN after the Mamba layer for
    additional expressivity. This provides that option.

    Architecture:
        x -> norm1 -> Mamba -> + residual -> norm2 -> FFN -> + residual -> output

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for Mamba
        intermediate_size: FFN intermediate dimension
        norm_eps: Normalization epsilon

    Example:
        >>> block = MambaWithFFN(d_model=768, intermediate_size=2048)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        intermediate_size: int | None = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._hidden_size = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        intermediate_size = intermediate_size or d_model * 4

        # Mamba sublayer
        self.mamba_norm = RMSNorm(d_model, eps=norm_eps)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # FFN sublayer
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, d_model),
        )

    @property
    def block_type(self) -> BlockType:
        return BlockType.MAMBA

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> BlockOutput:
        """Forward pass."""
        # Mamba with residual
        residual = x
        x = self.mamba_norm(x)
        x, new_cache = self.mamba(x, cache)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Initialize Mamba cache."""
        return self.mamba.init_cache(batch_size)


def create_mamba_block_wrapper(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    norm_eps: float = 1e-5,
) -> MambaBlockWrapper:
    """Factory function for MambaBlockWrapper."""
    return MambaBlockWrapper(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        norm_eps=norm_eps,
    )


def create_mamba_with_ffn(
    d_model: int,
    d_state: int = 16,
    intermediate_size: int | None = None,
) -> MambaWithFFN:
    """Factory function for MambaWithFFN."""
    return MambaWithFFN(
        d_model=d_model,
        d_state=d_state,
        intermediate_size=intermediate_size,
    )
