"""
Training utilities for models_v2.

Provides loss functions and training helpers.
"""

from .loss import compute_lm_loss

__all__ = [
    "compute_lm_loss",
]
