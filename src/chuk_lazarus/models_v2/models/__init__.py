"""
Complete models combining backbone + head.

Models are the full end-to-end architecture ready for training/inference.

Provides:
- CausalLM: Causal language model (next token prediction)
- SequenceClassifier: Sequence classification model
- TokenClassifier: Token classification model
"""

from .base import Model, ModelOutput
from .causal_lm import CausalLM, create_causal_lm
from .classifier import SequenceClassifier, TokenClassifier, create_classifier

__all__ = [
    # Base
    "Model",
    "ModelOutput",
    # Causal LM
    "CausalLM",
    "create_causal_lm",
    # Classifiers
    "SequenceClassifier",
    "TokenClassifier",
    "create_classifier",
]
