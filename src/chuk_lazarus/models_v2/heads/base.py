"""
Base head abstractions.

Defines the common interface for all output heads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HeadOutput:
    """
    Output from a head forward pass.

    Attributes:
        logits: Output logits/predictions
        loss: Optional loss (if labels provided)
        aux_outputs: Optional auxiliary outputs
    """

    logits: mx.array
    loss: mx.array | None = None
    aux_outputs: dict[str, Any] | None = None


class Head(nn.Module, ABC):
    """
    Abstract base class for output heads.

    A head transforms backbone hidden states into task-specific outputs.
    Different heads support different tasks:
    - LMHead: Next token prediction
    - ClassifierHead: Classification
    - RegressionHead: Continuous output
    """

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return the output dimension."""
        pass

    @abstractmethod
    def __call__(
        self,
        hidden_states: mx.array,
        labels: mx.array | None = None,
    ) -> HeadOutput:
        """
        Forward pass.

        Args:
            hidden_states: Backbone output, shape (batch, seq_len, hidden_size)
            labels: Optional labels for loss computation

        Returns:
            HeadOutput with logits and optional loss
        """
        pass
