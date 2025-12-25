"""
Complete models combining backbone + head.

Models are the full end-to-end architecture ready for training/inference.

Provides:
- CausalLM: Causal language model (next token prediction)
- LinearClassifier: Simple linear classifier for feature vectors
- MLPClassifier: MLP-based classifier for feature vectors
- SequenceClassifier: Sequence classification model (with backbone)
- TokenClassifier: Token classification model (with backbone)
"""

from .base import Model, ModelOutput
from .causal_lm import CausalLM, create_causal_lm
from .classifiers import (
    LinearClassifier,
    MLPClassifier,
    SequenceClassifier,
    TokenClassifier,
    create_classifier,
)

__all__ = [
    # Base
    "Model",
    "ModelOutput",
    # Causal LM
    "CausalLM",
    "create_causal_lm",
    # Simple Classifiers
    "LinearClassifier",
    "MLPClassifier",
    # Backbone-based Classifiers
    "SequenceClassifier",
    "TokenClassifier",
    "create_classifier",
]
