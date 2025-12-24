"""
Output heads for different tasks.

Heads take backbone hidden states and produce task-specific outputs.

Provides:
- LMHead: Language modeling head (next token prediction)
- ClassifierHead: Sequence/token classification
- RegressionHead: Regression output
- PoolerHead: Sequence pooling for classification
"""

from .base import Head, HeadOutput
from .classifier import ClassifierHead, PoolerHead, create_classifier_head
from .lm_head import LMHead, create_lm_head, cross_entropy_loss
from .regression import RegressionHead, create_regression_head

__all__ = [
    # Base
    "Head",
    "HeadOutput",
    # LM
    "LMHead",
    "create_lm_head",
    "cross_entropy_loss",
    # Classification
    "ClassifierHead",
    "PoolerHead",
    "create_classifier_head",
    # Regression
    "RegressionHead",
    "create_regression_head",
]
