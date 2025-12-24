"""
Sinusoidal position embeddings.

Fixed (non-learnable) position embeddings using sinusoidal functions.
From the original Transformer paper: "Attention Is All You Need".

Reference: https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings.

    Uses sine and cosine functions of different frequencies.
    These are fixed (not learned) during training.

    Args:
        max_position_embeddings: Maximum sequence length
        hidden_size: Embedding dimension (must be even)

    Example:
        >>> pos_embed = SinusoidalPositionEmbedding(
        ...     max_position_embeddings=512,
        ...     hidden_size=768
        ... )
        >>> embeddings = pos_embed(seq_len=10)  # (1, 10, 768)
    """

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
    ):
        super().__init__()

        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even, got {hidden_size}")

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        # Precompute embeddings
        embeddings = self._create_sinusoidal_embeddings(max_position_embeddings, hidden_size)
        self._embeddings = embeddings

    def _create_sinusoidal_embeddings(
        self,
        max_len: int,
        dim: int,
    ) -> mx.array:
        """
        Create sinusoidal embedding table.

        Args:
            max_len: Maximum sequence length
            dim: Embedding dimension

        Returns:
            Embedding table, shape (max_len, dim)
        """
        # Position indices
        positions = mx.arange(max_len).astype(mx.float32)

        # Dimension indices for half the dimensions
        dim_indices = mx.arange(dim // 2).astype(mx.float32)

        # Compute frequencies: 1 / (10000^(2i/dim))
        frequencies = 1.0 / (10000.0 ** (2.0 * dim_indices / dim))

        # Outer product: (max_len, dim/2)
        angles = mx.outer(positions, frequencies)

        # Interleave sin and cos
        sin_embeddings = mx.sin(angles)
        cos_embeddings = mx.cos(angles)

        # Stack: (max_len, dim)
        embeddings = mx.concatenate(
            [sin_embeddings[:, :, None], cos_embeddings[:, :, None]], axis=-1
        ).reshape(max_len, dim)

        return embeddings

    def __call__(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> mx.array:
        """
        Get position embeddings for a sequence.

        Args:
            seq_len: Sequence length
            offset: Starting position (for cached generation)

        Returns:
            Position embeddings, shape (1, seq_len, hidden_size)
        """
        embeddings = self._embeddings[offset : offset + seq_len]
        return embeddings[None, :, :]  # Add batch dimension

    def forward_with_input(
        self,
        input_ids: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """
        Get position embeddings matching input shape.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            offset: Starting position

        Returns:
            Position embeddings, shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape
        embeddings = self._embeddings[offset : offset + seq_len]
        return mx.broadcast_to(embeddings[None, :, :], (batch_size, seq_len, self.hidden_size))


def create_sinusoidal_position_embedding(
    max_position_embeddings: int,
    hidden_size: int,
) -> SinusoidalPositionEmbedding:
    """
    Factory function for sinusoidal position embeddings.

    Args:
        max_position_embeddings: Maximum sequence length
        hidden_size: Embedding dimension

    Returns:
        SinusoidalPositionEmbedding instance
    """
    return SinusoidalPositionEmbedding(
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
    )
