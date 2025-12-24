"""
Classification heads.

Heads for sequence and token classification tasks.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .base import Head, HeadOutput


class ClassifierHead(Head):
    """
    Classification head for sequence or token classification.

    For sequence classification, uses the last token or pooled representation.
    For token classification, classifies each token independently.

    Args:
        hidden_size: Input hidden dimension
        num_labels: Number of output classes
        pool_strategy: How to pool sequence ("last", "first", "mean", "none")
        dropout: Dropout rate before classification
        bias: Whether to use bias

    Example:
        >>> # Sequence classification
        >>> head = ClassifierHead(hidden_size=768, num_labels=3, pool_strategy="last")
        >>> hidden = mx.random.normal((2, 100, 768))
        >>> output = head(hidden)
        >>> output.logits.shape
        (2, 3)

        >>> # Token classification
        >>> head = ClassifierHead(hidden_size=768, num_labels=9, pool_strategy="none")
        >>> output = head(hidden)
        >>> output.logits.shape
        (2, 100, 9)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        pool_strategy: str = "last",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._num_labels = num_labels
        self.pool_strategy = pool_strategy

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Classification projection
        self.classifier = nn.Linear(hidden_size, num_labels, bias=bias)

    @property
    def output_size(self) -> int:
        return self._num_labels

    def __call__(
        self,
        hidden_states: mx.array,
        labels: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> HeadOutput:
        """
        Forward pass.

        Args:
            hidden_states: Backbone output, shape (batch, seq_len, hidden_size)
            labels: Optional labels for loss
                - Sequence: shape (batch,)
                - Token: shape (batch, seq_len)
            attention_mask: Optional mask for mean pooling

        Returns:
            HeadOutput with logits and optional loss
        """
        # Pool sequence if needed
        if self.pool_strategy == "last":
            # Use last token representation
            pooled = hidden_states[:, -1, :]
        elif self.pool_strategy == "first":
            # Use first token (e.g., [CLS])
            pooled = hidden_states[:, 0, :]
        elif self.pool_strategy == "mean":
            # Mean pool over sequence
            if attention_mask is not None:
                mask = mx.expand_dims(attention_mask, -1)
                pooled = mx.sum(hidden_states * mask, axis=1) / mx.sum(mask, axis=1)
            else:
                pooled = mx.mean(hidden_states, axis=1)
        else:
            # No pooling (token classification)
            pooled = hidden_states

        # Apply dropout
        if self.dropout is not None:
            pooled = self.dropout(pooled)

        # Classify
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.pool_strategy == "none":
                # Token classification: cross-entropy per token
                loss = token_classification_loss(logits, labels)
            else:
                # Sequence classification: single cross-entropy
                loss = sequence_classification_loss(logits, labels)

        return HeadOutput(logits=logits, loss=loss)


class PoolerHead(Head):
    """
    Pooler head with additional transformation.

    Applies a dense layer with activation before classification.
    Similar to BERT's pooler.

    Args:
        hidden_size: Input hidden dimension
        num_labels: Number of output classes
        pool_strategy: Pooling strategy
        activation: Activation function

    Example:
        >>> head = PoolerHead(hidden_size=768, num_labels=2)
        >>> hidden = mx.random.normal((2, 100, 768))
        >>> output = head(hidden)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        pool_strategy: str = "first",
        activation: str = "tanh",
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._num_labels = num_labels
        self.pool_strategy = pool_strategy

        # Pooler dense layer
        self.dense = nn.Linear(hidden_size, hidden_size)

        # Activation
        if activation == "tanh":
            self.activation = mx.tanh
        elif activation == "gelu":
            self.activation = nn.gelu
        else:
            self.activation = lambda x: x

        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_labels)

    @property
    def output_size(self) -> int:
        return self._num_labels

    def __call__(
        self,
        hidden_states: mx.array,
        labels: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> HeadOutput:
        """Forward pass."""
        # Pool
        if self.pool_strategy == "first":
            pooled = hidden_states[:, 0, :]
        else:
            pooled = hidden_states[:, -1, :]

        # Transform
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)

        # Classify
        logits = self.classifier(pooled)

        # Loss
        loss = None
        if labels is not None:
            loss = sequence_classification_loss(logits, labels)

        return HeadOutput(logits=logits, loss=loss)


def sequence_classification_loss(
    logits: mx.array,
    labels: mx.array,
) -> mx.array:
    """
    Cross-entropy loss for sequence classification.

    Args:
        logits: Predictions, shape (batch, num_labels)
        labels: Targets, shape (batch,)

    Returns:
        Scalar loss
    """
    num_labels = logits.shape[-1]

    # Log softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # One-hot encoding
    one_hot = mx.zeros_like(log_probs)
    one_hot = mx.where(
        mx.expand_dims(mx.arange(num_labels), 0) == mx.expand_dims(labels, 1),
        1.0,
        0.0,
    )

    # Cross-entropy
    loss = -mx.sum(log_probs * one_hot, axis=-1)
    return mx.mean(loss)


def token_classification_loss(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = -100,
) -> mx.array:
    """
    Cross-entropy loss for token classification.

    Args:
        logits: Predictions, shape (batch, seq_len, num_labels)
        labels: Targets, shape (batch, seq_len)
        ignore_index: Label to ignore

    Returns:
        Scalar loss
    """
    batch_size, seq_len, num_labels = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, num_labels)
    labels_flat = labels.reshape(-1)

    # Mask
    mask = labels_flat != ignore_index
    valid_labels = mx.where(mask, labels_flat, 0)

    # Log softmax
    log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)

    # One-hot
    one_hot = mx.zeros_like(log_probs)
    one_hot = mx.where(
        mx.expand_dims(mx.arange(num_labels), 0) == mx.expand_dims(valid_labels, 1),
        1.0,
        0.0,
    )

    # Cross-entropy with mask
    nll = -mx.sum(log_probs * one_hot, axis=-1)
    masked_nll = nll * mask
    loss = mx.sum(masked_nll) / mx.maximum(mx.sum(mask), 1.0)

    return loss


def create_classifier_head(
    hidden_size: int,
    num_labels: int,
    pool_strategy: str = "last",
) -> ClassifierHead:
    """Factory function for ClassifierHead."""
    return ClassifierHead(
        hidden_size=hidden_size,
        num_labels=num_labels,
        pool_strategy=pool_strategy,
    )
