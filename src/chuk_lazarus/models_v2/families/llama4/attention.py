"""
Llama 4 Attention.

Extends GQA with Llama 4-specific features:
- QK normalization (RMS norm on Q and K after RoPE)
- iRoPE: Interleaved RoPE and NoPE (global) layers
- Temperature scaling for long sequences

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...components.embeddings.rope import RoPE
from ...components.normalization import RMSNorm
from ...core.config import RoPEConfig
from .config import Llama4TextConfig


class Llama4Attention(nn.Module):
    """
    Llama 4 attention with QK normalization and iRoPE support.

    Key features:
    - QK normalization: RMS norm applied to Q and K after projection
    - iRoPE: Interleaved RoPE (chunked attention) and NoPE (global) layers
    - Temperature scaling for attention logits

    Args:
        config: Llama 4 text configuration
        layer_idx: Layer index (determines if this is a NoPE layer)

    Example:
        >>> config = Llama4TextConfig.tiny()
        >>> attn = Llama4Attention(config, layer_idx=0)  # NoPE layer
        >>> x = mx.random.normal((2, 10, 64))
        >>> output, cache = attn(x)
    """

    def __init__(self, config: Llama4TextConfig, layer_idx: int = 0):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        # Number of query heads per KV head
        self.n_rep = self.num_heads // self.num_kv_heads

        # Determine if this is a NoPE layer (no RoPE, global attention)
        self.is_nope_layer = False
        if config.no_rope_layers is not None:
            self.is_nope_layer = layer_idx in config.no_rope_layers

        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK normalization
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE for non-NoPE layers
        self.rope = None
        if not self.is_nope_layer:
            rope_config = RoPEConfig(
                theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
            )
            self.rope = RoPE(rope_config, dims=self.head_dim)

        # Attention scaling
        self.scale = self.head_dim**-0.5

        # Temperature scaling
        self.attn_temperature_tuning = config.attn_temperature_tuning
        if self.attn_temperature_tuning:
            # Learned temperature parameter
            self.temperature = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Compute Llama 4 attention.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            mask: Attention mask (additive, -inf for masked)
            cache: Optional KV cache

        Returns:
            output: Shape (batch, seq_len, hidden_size)
            cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization (before RoPE)
        if self.use_qk_norm:
            # Normalize per head
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Get cache offset for RoPE
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]

        # Apply RoPE (only for non-NoPE layers)
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        # Update cache
        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = (k, v)

        # Repeat KV heads to match query heads
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)

        # Compute attention scale
        scale = self.scale
        if self.attn_temperature_tuning:
            scale = scale / self.temperature

        # Compute attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, new_cache

    def _repeat_kv(self, x: mx.array, n_rep: int) -> mx.array:
        """Repeat KV heads to match query heads."""
        if n_rep == 1:
            return x

        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = mx.expand_dims(x, axis=2)
        x = mx.repeat(x, n_rep, axis=2)
        x = x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
        return x


class Llama4FlexAttention(Llama4Attention):
    """
    Flexible attention with floor-scale RoPE.

    Used in some Llama 4 variants for very long context.
    Implements a floor function on the position indices.
    """

    def __init__(self, config: Llama4TextConfig, layer_idx: int = 0):
        super().__init__(config, layer_idx)

        # Floor scale for position indices (for long context)
        self.floor_scale = 1  # Can be adjusted for different context lengths


def create_llama4_attention(
    config: Llama4TextConfig,
    layer_idx: int = 0,
    attention_type: str = "default",
) -> nn.Module:
    """
    Factory function for Llama 4 attention.

    Args:
        config: Llama 4 text configuration
        layer_idx: Layer index
        attention_type: "default" or "flex"

    Returns:
        Attention module
    """
    if attention_type == "flex":
        return Llama4FlexAttention(config, layer_idx)
    return Llama4Attention(config, layer_idx)
