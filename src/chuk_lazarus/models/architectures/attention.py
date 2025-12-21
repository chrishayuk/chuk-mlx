"""Attention mechanisms."""

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig


class Attention(nn.Module):
    """
    Multi-head attention with support for:
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - KV caching for efficient generation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # RoPE
        self.rope = self._setup_rope(config)

    def _setup_rope(self, config: ModelConfig) -> nn.RoPE | None:
        """Setup rotary position embeddings."""
        if config.max_position_embeddings:
            rope_scale = 1.0
            if config.rope_scaling:
                rope_scale = 1 / config.rope_scaling.get("factor", 1.0)

            return nn.RoPE(
                dims=self.head_dim,
                traditional=config.rope_traditional,
                base=config.rope_theta,
                scale=rope_scale,
            )
        return None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size)
            mask: Attention mask
            cache: Tuple of (key_cache, value_cache)

        Returns:
            output: Shape (batch, seq_len, hidden_size)
            cache: Updated (key_cache, value_cache)
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE and handle cache
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]

            if self.rope is not None:
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)

            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        else:
            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)

        # Compute attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        return output, (k, v)


class GQAAttention(Attention):
    """
    Grouped Query Attention.

    Same as Attention but with explicit GQA support.
    The base Attention class already supports GQA via num_kv_heads.
    """

    pass
