"""
Base backbone abstractions.

Defines the common interface for all backbone types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BackboneOutput:
    """
    Output from a backbone forward pass.

    Attributes:
        last_hidden_state: Final hidden states, shape (batch, seq_len, hidden_size)
        hidden_states: Optional tuple of all layer hidden states
        cache: Optional cache for inference
        aux_loss: Optional auxiliary loss (e.g., from MoE)
    """

    last_hidden_state: mx.array
    hidden_states: tuple[mx.array, ...] | None = None
    cache: list[Any] | None = None
    aux_loss: mx.array | None = None


class Backbone(nn.Module, ABC):
    """
    Abstract base class for all backbones.

    A backbone is the core of a model, consisting of:
    - Input embeddings (token + position)
    - A stack of blocks (attention, SSM, RNN, etc.)
    - Final normalization

    It produces hidden states that can be processed by different heads.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of layers/blocks."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass

    @abstractmethod
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass through the backbone.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask
            cache: Optional list of per-layer caches
            output_hidden_states: Whether to return all layer hidden states

        Returns:
            BackboneOutput with hidden states and optional cache
        """
        pass

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> list[Any]:
        """
        Initialize cache for all layers.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length

        Returns:
            List of per-layer caches
        """
        return [None] * self.num_layers

    def get_input_embeddings(self) -> nn.Module:
        """Return the input embedding layer."""
        raise NotImplementedError

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        """Set the input embedding layer."""
        raise NotImplementedError
