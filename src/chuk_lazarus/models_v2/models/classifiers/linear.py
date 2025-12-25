"""
Linear classifier.

Simple linear classification for feature vectors.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LinearClassifier(nn.Module):
    """
    Simple linear classifier.

    Single linear layer for binary or multi-class classification
    on feature vectors.

    Args:
        input_size: Input feature dimension
        num_labels: Number of output classes (default 1 for binary)
        bias: Whether to use bias

    Example:
        >>> clf = LinearClassifier(input_size=768, num_labels=2)
        >>> x = mx.random.normal((32, 768))
        >>> logits = clf(x)
        >>> logits.shape
        (32, 2)
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, num_labels, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input features, shape (batch, input_size)

        Returns:
            Logits, shape (batch, num_labels)
        """
        return self.fc(x)
