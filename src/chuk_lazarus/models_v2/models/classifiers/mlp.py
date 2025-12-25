"""
MLP classifier.

Multi-layer perceptron classification for feature vectors.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...components.ffn import create_mlp
from ...core.enums import ActivationType


class MLPClassifier(nn.Module):
    """
    MLP-based classifier.

    Two-layer feed-forward network for classification on feature vectors.
    Uses the chuk-mlx MLP component internally.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden layer dimension
        num_labels: Number of output classes (default 1 for binary)
        activation: Activation function type
        bias: Whether to use bias

    Example:
        >>> clf = MLPClassifier(input_size=768, hidden_size=256, num_labels=3)
        >>> x = mx.random.normal((32, 768))
        >>> logits = clf(x)
        >>> logits.shape
        (32, 3)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_labels: int = 1,
        activation: ActivationType = ActivationType.GELU,
        bias: bool = True,
    ):
        super().__init__()

        # Hidden layer using chuk-mlx MLP
        # MLP: input_size -> hidden_size -> input_size
        self.mlp = create_mlp(
            hidden_size=input_size,
            intermediate_size=hidden_size,
            activation=activation,
            bias=bias,
        )

        # Output projection
        self.classifier = nn.Linear(input_size, num_labels, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input features, shape (batch, input_size)

        Returns:
            Logits, shape (batch, num_labels)
        """
        x = self.mlp(x)
        return self.classifier(x)
