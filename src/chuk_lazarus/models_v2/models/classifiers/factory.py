"""
Classifier factory functions.

Convenience functions for creating classifiers.
"""

from __future__ import annotations

from ...core.config import ModelConfig
from ...core.enums import ClassificationTask
from ..base import Model
from .sequence import SequenceClassifier
from .token import TokenClassifier


def create_classifier(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_labels: int,
    task: ClassificationTask | str = ClassificationTask.SEQUENCE,
) -> Model:
    """
    Factory function for classifiers.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        num_labels: Number of output classes
        task: ClassificationTask.SEQUENCE or TOKEN

    Returns:
        Classifier model
    """
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
    )

    # Normalize task to enum
    task_enum = ClassificationTask(task) if isinstance(task, str) else task

    if task_enum == ClassificationTask.SEQUENCE:
        return SequenceClassifier(config=config, num_labels=num_labels)
    else:
        return TokenClassifier(config=config, num_labels=num_labels)
