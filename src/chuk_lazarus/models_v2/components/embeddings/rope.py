"""
Rotary Position Embeddings (RoPE).

RoPE encodes position information by rotating query and key vectors.
This implementation supports:
- Standard RoPE (Llama, Mistral)
- Scaled RoPE for extended context (YaRN, dynamic scaling)
- Traditional vs interleaved ordering

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import RoPEConfig


class RoPE(nn.Module):
    """
    Rotary Position Embeddings.

    Applies rotation to query and key vectors based on position.

    Args:
        config: RoPE configuration
        dims: Dimension to apply RoPE (typically head_dim)

    Example:
        >>> config = RoPEConfig(theta=10000.0, max_position_embeddings=4096)
        >>> rope = RoPE(config, dims=128)
        >>> q = mx.random.normal((2, 32, 10, 128))  # (batch, heads, seq, head_dim)
        >>> q_rotated = rope(q, offset=0)
    """

    def __init__(self, config: RoPEConfig, dims: int):
        super().__init__()

        self.dims = dims
        self.theta = config.theta
        self.traditional = config.traditional
        self.scaling_factor = config.scaling_factor
        self.max_position_embeddings = config.max_position_embeddings

        # Use MLX's built-in RoPE
        self._rope = nn.RoPE(
            dims=dims,
            traditional=config.traditional,
            base=config.theta,
            scale=1.0 / config.scaling_factor if config.scaling_factor != 1.0 else 1.0,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor, shape (batch, heads, seq_len, head_dim)
            offset: Position offset (for KV cache during generation)

        Returns:
            Rotated tensor, same shape as input
        """
        return self._rope(x, offset=offset)

    def rotate_half(self, x: mx.array) -> mx.array:
        """
        Rotate half the hidden dims.

        Used for manual RoPE computation if needed.

        Args:
            x: Input tensor

        Returns:
            Tensor with half dims rotated
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    @classmethod
    def from_config(cls, config: RoPEConfig, head_dim: int) -> RoPE:
        """
        Create RoPE from config.

        Args:
            config: RoPE configuration
            head_dim: Dimension per attention head

        Returns:
            RoPE instance
        """
        return cls(config, dims=head_dim)


def compute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """
    Compute RoPE frequency tensors.

    Args:
        dim: Embedding dimension (head_dim)
        max_seq_len: Maximum sequence length
        theta: Base frequency
        scaling_factor: Scaling factor for extended context

    Returns:
        Tuple of (cos, sin) tensors, each shape (max_seq_len, dim)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    # Apply scaling
    if scaling_factor != 1.0:
        inv_freq = inv_freq / scaling_factor

    # Compute positions
    positions = mx.arange(max_seq_len).astype(mx.float32)

    # Outer product: (seq_len, dim/2)
    freqs = mx.outer(positions, inv_freq)

    # Expand to full dim: (seq_len, dim)
    freqs = mx.concatenate([freqs, freqs], axis=-1)

    return mx.cos(freqs), mx.sin(freqs)


def apply_rope_manual(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
) -> tuple[mx.array, mx.array]:
    """
    Apply RoPE manually (for custom implementations).

    Args:
        q: Query tensor, shape (batch, heads, seq_len, head_dim)
        k: Key tensor, shape (batch, heads, seq_len, head_dim)
        cos: Cosine frequencies, shape (max_seq_len, head_dim)
        sin: Sine frequencies, shape (max_seq_len, head_dim)
        offset: Position offset

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    seq_len = q.shape[2]

    # Get relevant positions
    cos = cos[offset : offset + seq_len]
    sin = sin[offset : offset + seq_len]

    # Reshape for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Apply rotation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated
