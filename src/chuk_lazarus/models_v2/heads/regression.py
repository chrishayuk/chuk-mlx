"""
Regression head.

Head for continuous output tasks (e.g., sentiment scores, similarity).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .base import Head, HeadOutput


class RegressionHead(Head):
    """
    Regression head for continuous output.

    Produces a single or multi-dimensional continuous output.
    Uses MSE loss for training.

    Args:
        hidden_size: Input hidden dimension
        output_dim: Output dimension (default 1 for scalar regression)
        pool_strategy: How to pool sequence
        dropout: Dropout rate
        use_hidden_layer: Add hidden layer before output

    Example:
        >>> head = RegressionHead(hidden_size=768, output_dim=1)
        >>> hidden = mx.random.normal((2, 100, 768))
        >>> output = head(hidden)
        >>> output.logits.shape
        (2, 1)
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int = 1,
        pool_strategy: str = "last",
        dropout: float = 0.0,
        use_hidden_layer: bool = False,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._output_dim = output_dim
        self.pool_strategy = pool_strategy

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Optional hidden layer
        if use_hidden_layer:
            self.hidden = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.gelu
        else:
            self.hidden = None
            self.activation = None

        # Output projection
        self.output = nn.Linear(hidden_size, output_dim)

    @property
    def output_size(self) -> int:
        return self._output_dim

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
            labels: Optional targets, shape (batch, output_dim) or (batch,)
            attention_mask: Optional mask for mean pooling

        Returns:
            HeadOutput with predictions and optional MSE loss
        """
        # Pool sequence
        if self.pool_strategy == "last":
            pooled = hidden_states[:, -1, :]
        elif self.pool_strategy == "first":
            pooled = hidden_states[:, 0, :]
        elif self.pool_strategy == "mean":
            if attention_mask is not None:
                mask = mx.expand_dims(attention_mask, -1)
                pooled = mx.sum(hidden_states * mask, axis=1) / mx.sum(mask, axis=1)
            else:
                pooled = mx.mean(hidden_states, axis=1)
        else:
            # Per-position regression
            pooled = hidden_states

        # Apply dropout
        if self.dropout is not None:
            pooled = self.dropout(pooled)

        # Optional hidden layer
        if self.hidden is not None:
            pooled = self.hidden(pooled)
            pooled = self.activation(pooled)

        # Output projection
        predictions = self.output(pooled)

        # Compute MSE loss if labels provided
        loss = None
        if labels is not None:
            # Ensure labels have same shape as predictions
            if labels.ndim == 1:
                labels = mx.expand_dims(labels, -1)
            loss = mse_loss(predictions, labels)

        return HeadOutput(logits=predictions, loss=loss)


class MultiTaskRegressionHead(Head):
    """
    Multi-task regression head.

    Produces multiple regression outputs with optional task-specific layers.

    Args:
        hidden_size: Input hidden dimension
        task_dims: Dictionary mapping task names to output dimensions
        shared_hidden: Use shared hidden layer before task heads
        pool_strategy: Pooling strategy

    Example:
        >>> head = MultiTaskRegressionHead(
        ...     hidden_size=768,
        ...     task_dims={"sentiment": 1, "rating": 1, "embedding": 128},
        ... )
        >>> hidden = mx.random.normal((2, 100, 768))
        >>> output = head(hidden)
        >>> output.aux_outputs["sentiment"].shape
        (2, 1)
    """

    def __init__(
        self,
        hidden_size: int,
        task_dims: dict[str, int],
        shared_hidden: bool = True,
        pool_strategy: str = "last",
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self.task_dims = task_dims
        self.task_names = list(task_dims.keys())
        self.pool_strategy = pool_strategy

        # Optional shared hidden layer
        if shared_hidden:
            self.shared = nn.Linear(hidden_size, hidden_size)
        else:
            self.shared = None

        # Task-specific heads
        self.task_heads = {name: nn.Linear(hidden_size, dim) for name, dim in task_dims.items()}

    @property
    def output_size(self) -> int:
        return sum(self.task_dims.values())

    def __call__(
        self,
        hidden_states: mx.array,
        labels: dict[str, mx.array] | None = None,
    ) -> HeadOutput:
        """
        Forward pass.

        Args:
            hidden_states: Backbone output
            labels: Optional dict of task labels

        Returns:
            HeadOutput with combined loss and per-task outputs in aux_outputs
        """
        # Pool
        if self.pool_strategy == "last":
            pooled = hidden_states[:, -1, :]
        elif self.pool_strategy == "first":
            pooled = hidden_states[:, 0, :]
        else:
            pooled = mx.mean(hidden_states, axis=1)

        # Shared layer
        if self.shared is not None:
            pooled = nn.gelu(self.shared(pooled))

        # Task outputs
        task_outputs = {name: head(pooled) for name, head in self.task_heads.items()}

        # Combine into single tensor for compatibility
        combined = mx.concatenate(list(task_outputs.values()), axis=-1)

        # Compute losses if labels provided
        loss = None
        if labels is not None:
            losses = []
            for name in self.task_names:
                if name in labels:
                    task_labels = labels[name]
                    if task_labels.ndim == 1:
                        task_labels = mx.expand_dims(task_labels, -1)
                    task_loss = mse_loss(task_outputs[name], task_labels)
                    losses.append(task_loss)

            if losses:
                loss = mx.mean(mx.stack(losses))

        return HeadOutput(
            logits=combined,
            loss=loss,
            aux_outputs=task_outputs,
        )


def mse_loss(predictions: mx.array, targets: mx.array) -> mx.array:
    """
    Mean squared error loss.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Scalar MSE loss
    """
    return mx.mean((predictions - targets) ** 2)


def create_regression_head(
    hidden_size: int,
    output_dim: int = 1,
    pool_strategy: str = "last",
) -> RegressionHead:
    """Factory function for RegressionHead."""
    return RegressionHead(
        hidden_size=hidden_size,
        output_dim=output_dim,
        pool_strategy=pool_strategy,
    )
