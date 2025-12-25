"""
Classification models.

Provides:
- LinearClassifier: Simple linear classifier for feature vectors
- MLPClassifier: MLP-based classifier for feature vectors
- SequenceClassifier: Sequence classification with backbone
- TokenClassifier: Token classification with backbone
"""

from .factory import create_classifier
from .linear import LinearClassifier
from .mlp import MLPClassifier
from .sequence import SequenceClassifier
from .token import TokenClassifier

__all__ = [
    # Simple classifiers
    "LinearClassifier",
    "MLPClassifier",
    # Backbone-based classifiers
    "SequenceClassifier",
    "TokenClassifier",
    # Factory
    "create_classifier",
]
