"""
Hybrid blocks combining multiple sequence modeling approaches.

These blocks combine attention, SSM, and/or RNN components within
a single layer for potentially better performance.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..components.attention import GroupedQueryAttention
from ..components.normalization import RMSNorm
from ..components.ssm import Mamba
from ..core.enums import BlockType, HybridCombineMode
from .base import Block, BlockOutput


class HybridBlock(Block):
    """
    Hybrid block combining attention and Mamba.

    Processes input through both attention and Mamba, then combines
    the outputs. This can capture both global (attention) and local
    (Mamba) patterns efficiently.

    Architecture:
        x -> norm -> [attention, mamba] -> combine -> + residual -> norm -> FFN -> + residual

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        d_state: Mamba state dimension
        intermediate_size: FFN intermediate dimension
        combine_mode: How to combine outputs (HybridCombineMode.ADD, CONCAT, or GATE)
        norm_eps: Normalization epsilon

    Example:
        >>> block = HybridBlock(d_model=768, num_heads=12)
        >>> x = mx.random.normal((2, 100, 768))
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        intermediate_size: int | None = None,
        combine_mode: HybridCombineMode | str = HybridCombineMode.ADD,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._hidden_size = d_model
        # Normalize combine_mode to enum
        self.combine_mode = (
            HybridCombineMode(combine_mode) if isinstance(combine_mode, str) else combine_mode
        )
        self.d_state = d_state

        num_kv_heads = num_kv_heads or num_heads
        intermediate_size = intermediate_size or d_model * 4
        head_dim = d_model // num_heads

        # Pre-norm for sequence modeling
        self.input_norm = RMSNorm(d_model, eps=norm_eps)

        # Attention branch
        from ..core.config import AttentionConfig

        attention_config = AttentionConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=d_model,
            head_dim=head_dim,
        )
        self.attention = GroupedQueryAttention(attention_config)

        # Mamba branch
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
        )

        # Combination strategy
        if self.combine_mode == HybridCombineMode.CONCAT:
            # Concat and project back
            self.combine_proj = nn.Linear(d_model * 2, d_model)
        elif self.combine_mode == HybridCombineMode.GATE:
            # Learnable gating
            self.gate = nn.Linear(d_model * 2, 2)
        else:
            # Simple addition (default)
            self.combine_proj = None

        # FFN sublayer
        self.post_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, d_model),
        )

    @property
    def block_type(self) -> BlockType:
        return BlockType.HYBRID

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: dict[str, Any] | None = None,
    ) -> BlockOutput:
        """
        Forward pass.

        Args:
            x: Input, shape (batch, seq_len, d_model)
            mask: Optional attention mask
            cache: Optional {"attn": (k, v), "mamba": (conv, ssm)} cache

        Returns:
            BlockOutput with hidden states and updated cache
        """
        residual = x
        x = self.input_norm(x)

        # Extract caches
        attn_cache = cache.get("attn") if cache else None
        mamba_cache = cache.get("mamba") if cache else None

        # Parallel branches
        attn_out, new_attn_cache = self.attention(x, mask=mask, cache=attn_cache)
        mamba_out, new_mamba_cache = self.mamba(x, mamba_cache)

        # Combine
        if self.combine_mode == HybridCombineMode.CONCAT:
            combined = mx.concatenate([attn_out, mamba_out], axis=-1)
            x = self.combine_proj(combined)
        elif self.combine_mode == HybridCombineMode.GATE:
            # Compute gate weights
            combined = mx.concatenate([attn_out, mamba_out], axis=-1)
            gates = mx.softmax(self.gate(combined), axis=-1)  # (batch, seq, 2)
            x = gates[:, :, 0:1] * attn_out + gates[:, :, 1:2] * mamba_out
        else:
            # Simple addition with averaging
            x = (attn_out + mamba_out) / 2

        # Residual
        x = residual + x

        # FFN
        residual = x
        x = self.post_norm(x)
        x = self.ffn(x)
        x = residual + x

        # Build cache
        new_cache = {
            "attn": new_attn_cache,
            "mamba": new_mamba_cache,
        }

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> dict[str, Any]:
        """Initialize cache for both branches."""
        # Attention cache
        head_dim = self._hidden_size // self.attention.num_heads
        num_kv_heads = self.attention.num_kv_heads
        k_cache = mx.zeros((batch_size, num_kv_heads, max_seq_len, head_dim))
        v_cache = mx.zeros((batch_size, num_kv_heads, max_seq_len, head_dim))

        # Mamba cache
        mamba_cache = self.mamba.init_cache(batch_size)

        return {
            "attn": (k_cache, v_cache),
            "mamba": mamba_cache,
        }


class AlternatingHybrid(Block):
    """
    Block that alternates between attention and Mamba by layer.

    This is a simple hybrid approach where even layers use attention
    and odd layers use Mamba (or vice versa).

    Args:
        d_model: Model dimension
        layer_idx: Layer index (determines which component to use)
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads
        d_state: Mamba state dimension
        use_attention_first: If True, even layers use attention
        norm_eps: Normalization epsilon

    Example:
        >>> # Even layer uses attention
        >>> block = AlternatingHybrid(d_model=768, layer_idx=0, num_heads=12)
        >>> # Odd layer uses Mamba
        >>> block = AlternatingHybrid(d_model=768, layer_idx=1, num_heads=12)
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        intermediate_size: int | None = None,
        use_attention_first: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self._hidden_size = d_model
        self.layer_idx = layer_idx
        self.d_state = d_state

        num_kv_heads = num_kv_heads or num_heads
        intermediate_size = intermediate_size or d_model * 4
        head_dim = d_model // num_heads

        # Determine which component this layer uses
        use_attention = (layer_idx % 2 == 0) == use_attention_first
        self.uses_attention = use_attention

        # Pre-norm
        self.input_norm = RMSNorm(d_model, eps=norm_eps)

        # Sequence modeling component
        if use_attention:
            from ..core.config import AttentionConfig

            attention_config = AttentionConfig(
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                hidden_size=d_model,
                head_dim=head_dim,
            )
            self.seq_model = GroupedQueryAttention(attention_config)
        else:
            self.seq_model = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
            )

        # FFN
        self.post_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, d_model),
        )

    @property
    def block_type(self) -> BlockType:
        return BlockType.HYBRID

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> BlockOutput:
        """Forward pass."""
        residual = x
        x = self.input_norm(x)

        # Sequence modeling (attention or Mamba)
        x, new_cache = (
            self.seq_model(x, mask=mask, cache=cache)
            if self.uses_attention
            else self.seq_model(x, cache)
        )

        x = residual + x

        # FFN
        residual = x
        x = self.post_norm(x)
        x = self.ffn(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> Any:
        """Initialize cache."""
        if self.uses_attention:
            head_dim = self._hidden_size // self.seq_model.num_heads
            num_kv_heads = self.seq_model.num_kv_heads
            k_cache = mx.zeros((batch_size, num_kv_heads, max_seq_len, head_dim))
            v_cache = mx.zeros((batch_size, num_kv_heads, max_seq_len, head_dim))
            return (k_cache, v_cache)
        else:
            return self.seq_model.init_cache(batch_size)


def create_hybrid_block(
    d_model: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    d_state: int = 16,
    combine_mode: HybridCombineMode | str = HybridCombineMode.ADD,
) -> HybridBlock:
    """Factory function for HybridBlock."""
    return HybridBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_state=d_state,
        combine_mode=combine_mode,
    )


def create_alternating_hybrid(
    d_model: int,
    layer_idx: int,
    num_heads: int = 12,
) -> AlternatingHybrid:
    """Factory function for AlternatingHybrid."""
    return AlternatingHybrid(
        d_model=d_model,
        layer_idx=layer_idx,
        num_heads=num_heads,
    )
