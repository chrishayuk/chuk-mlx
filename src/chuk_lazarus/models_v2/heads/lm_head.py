"""
Language modeling head.

Projects hidden states to vocabulary for next token prediction.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .base import Head, HeadOutput


class LMHead(Head):
    """
    Language modeling head for next token prediction.

    Projects hidden states to vocabulary logits. Optionally shares
    weights with the input embedding layer (tied embeddings).

    Args:
        hidden_size: Input hidden dimension
        vocab_size: Output vocabulary size
        bias: Whether to use bias in projection
        tied_embeddings: Optional embedding weights to tie with

    Example:
        >>> head = LMHead(hidden_size=4096, vocab_size=32000)
        >>> hidden = mx.random.normal((2, 100, 4096))
        >>> output = head(hidden)
        >>> output.logits.shape
        (2, 100, 32000)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        tied_embeddings: nn.Module | None = None,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self.tied_embeddings = tied_embeddings

        # Only create projection if not using tied embeddings
        if tied_embeddings is None:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
        else:
            self.lm_head = None

    @property
    def output_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        hidden_states: mx.array,
        labels: mx.array | None = None,
    ) -> HeadOutput:
        """
        Forward pass.

        Args:
            hidden_states: Backbone output, shape (batch, seq_len, hidden_size)
            labels: Optional target token IDs, shape (batch, seq_len)

        Returns:
            HeadOutput with logits and optional cross-entropy loss
        """
        # Project to vocabulary
        if self.tied_embeddings is not None:
            # Use transposed embedding weights
            # Handle both nn.Embedding and our TokenEmbedding wrapper
            if hasattr(self.tied_embeddings, "weight") and hasattr(
                self.tied_embeddings.weight, "weight"
            ):
                # TokenEmbedding wrapper: tied_embeddings.weight is nn.Embedding
                logits = hidden_states @ self.tied_embeddings.weight.weight.T
            elif hasattr(self.tied_embeddings, "weight"):
                # Direct nn.Embedding
                weight = self.tied_embeddings.weight
                if hasattr(weight, "weight"):
                    logits = hidden_states @ weight.weight.T
                else:
                    logits = hidden_states @ weight.T
            else:
                logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            # logits: predict next token
            # labels: actual next token
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Cross-entropy loss
            loss = cross_entropy_loss(shift_logits, shift_labels)

        return HeadOutput(logits=logits, loss=loss)

    def tie_weights(self, embeddings: nn.Module) -> None:
        """Tie weights with embedding layer."""
        self.tied_embeddings = embeddings
        self.lm_head = None


def cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = -100,
) -> mx.array:
    """
    Compute cross-entropy loss.

    Args:
        logits: Predictions, shape (batch, seq_len, vocab_size)
        labels: Targets, shape (batch, seq_len)
        ignore_index: Label value to ignore in loss

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for cross-entropy
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Create mask for valid positions
    mask = labels_flat != ignore_index
    valid_labels = mx.where(mask, labels_flat, 0)

    # Compute log softmax
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)

    # Gather log probs for correct labels
    # Use one-hot for gather since MLX doesn't have advanced indexing
    one_hot = mx.zeros_like(log_probs)
    one_hot = mx.where(
        mx.expand_dims(mx.arange(vocab_size), 0) == mx.expand_dims(valid_labels, 1),
        1.0,
        0.0,
    )
    nll = -mx.sum(log_probs * one_hot, axis=-1)

    # Apply mask and compute mean
    masked_nll = nll * mask
    loss = mx.sum(masked_nll) / mx.maximum(mx.sum(mask), 1.0)

    return loss


def create_lm_head(
    hidden_size: int,
    vocab_size: int,
    bias: bool = False,
) -> LMHead:
    """Factory function for LMHead."""
    return LMHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        bias=bias,
    )
