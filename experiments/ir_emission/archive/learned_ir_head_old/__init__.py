"""
Learned IR Head Experiment.

Train a projection head to extract IR structure directly from hidden states,
eliminating text generation and regex parsing.
"""

from .heads import IRHead, IRHeadBinned, IRHeadRegression
from .dataset import IRDataset, create_training_data

__all__ = [
    "IRHead",
    "IRHeadBinned",
    "IRHeadRegression",
    "IRDataset",
    "create_training_data",
]
