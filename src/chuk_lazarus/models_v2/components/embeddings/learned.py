"""
Learned position embeddings.

Traditional position embeddings that are learned during training.
Used by models like GPT-2, BERT (though BERT also uses segment embeddings).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbedding(nn.Module):
    """
    Learned absolute position embeddings.

    Creates a learnable embedding table for each position.

    Args:
        max_position_embeddings: Maximum sequence length
        hidden_size: Embedding dimension

    Example:
        >>> pos_embed = LearnedPositionEmbedding(max_position_embeddings=512, hidden_size=768)
        >>> embeddings = pos_embed(seq_len=10, offset=0)  # (1, 10, 768)
    """

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
    ):
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        # Learnable position embeddings
        self.weight = nn.Embedding(max_position_embeddings, hidden_size)

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
        positions = mx.arange(offset, offset + seq_len)
        embeddings = self.weight(positions)
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
        positions = mx.arange(offset, offset + seq_len)
        embeddings = self.weight(positions)
        # Broadcast to batch size
        return mx.broadcast_to(embeddings[None, :, :], (batch_size, seq_len, self.hidden_size))


def create_learned_position_embedding(
    max_position_embeddings: int,
    hidden_size: int,
) -> LearnedPositionEmbedding:
    """
    Factory function for learned position embeddings.

    Args:
        max_position_embeddings: Maximum sequence length
        hidden_size: Embedding dimension

    Returns:
        LearnedPositionEmbedding instance
    """
    return LearnedPositionEmbedding(
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
    )
