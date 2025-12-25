"""
Loss functions for models_v2.

Pure math: CE, DPO loss, etc. Training loops live in src/chuk_lazarus/training/.
"""

from .loss import compute_lm_loss

__all__ = [
    "compute_lm_loss",
]
