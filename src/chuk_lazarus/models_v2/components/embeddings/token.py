"""
Token embedding component.

Provides token embeddings with optional scaling (e.g., for Gemma).
Backend-agnostic implementation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...core.config import EmbeddingConfig


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token IDs to dense vectors with optional scaling.

    Args:
        config: Embedding configuration

    Example:
        >>> config = EmbeddingConfig(vocab_size=32000, hidden_size=4096)
        >>> embed = TokenEmbedding(config)
        >>> tokens = mx.array([[1, 2, 3]])
        >>> embeddings = embed(tokens)  # (1, 3, 4096)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.scale_factor = config.scale_factor

        # Create embedding table
        self.weight = nn.Embedding(config.vocab_size, config.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Embed token IDs.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)

        Returns:
            Embeddings, shape (batch, seq_len, hidden_size)
        """
        embeddings = self.weight(input_ids)

        # Apply scaling if configured (e.g., Gemma uses sqrt(hidden_size))
        if self.scale_factor is not None:
            embeddings = embeddings * self.scale_factor

        return embeddings

    def as_linear(self, hidden: mx.array) -> mx.array:
        """
        Use embedding weights as output projection (for tied embeddings).

        Args:
            hidden: Hidden states, shape (batch, seq_len, hidden_size)

        Returns:
            Logits, shape (batch, seq_len, vocab_size)
        """
        return self.weight.as_linear(hidden)

    @classmethod
    def from_pretrained(
        cls,
        weight: mx.array,
        scale_factor: float | None = None,
    ) -> TokenEmbedding:
        """
        Create from pretrained weights.

        Args:
            weight: Embedding weights, shape (vocab_size, hidden_size)
            scale_factor: Optional scaling factor

        Returns:
            TokenEmbedding instance
        """
        vocab_size, hidden_size = weight.shape
        config = EmbeddingConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            scale_factor=scale_factor,
        )
        embed = cls(config)
        embed.weight.weight = weight
        return embed


def create_token_embedding(
    vocab_size: int,
    hidden_size: int,
    scale_factor: float | None = None,
    padding_idx: int | None = None,
) -> TokenEmbedding:
    """
    Factory function to create token embeddings.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Embedding dimension
        scale_factor: Optional scaling (e.g., sqrt(hidden_size) for Gemma)
        padding_idx: Padding token index (unused in MLX, kept for API compat)

    Returns:
        TokenEmbedding instance
    """
    config = EmbeddingConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        scale_factor=scale_factor,
        padding_idx=padding_idx,
    )
    return TokenEmbedding(config)
