"""
Base block abstractions.

Defines the common interface for all block types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..core.enums import BlockType


@dataclass
class BlockOutput:
    """
    Output from a block forward pass.

    Attributes:
        hidden_states: Output tensor, shape (batch, seq_len, hidden_size)
        cache: Optional cache for inference (type depends on block)
        aux_loss: Optional auxiliary loss (e.g., from MoE load balancing)
    """

    hidden_states: mx.array
    cache: Any | None = None
    aux_loss: mx.array | None = None


class Block(nn.Module, ABC):
    """
    Abstract base class for all blocks.

    A block is a complete layer that can be stacked to form a backbone.
    It typically combines:
    - A sequence modeling component (attention, SSM, or RNN)
    - A feedforward network
    - Normalization layers
    - Residual connections

    All blocks must implement the same interface for composability.
    """

    @property
    @abstractmethod
    def block_type(self) -> BlockType:
        """Return the type of this block."""
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension."""
        pass

    @abstractmethod
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> BlockOutput:
        """
        Forward pass through the block.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size)
            mask: Optional attention/causal mask
            cache: Optional cache for inference

        Returns:
            BlockOutput with hidden states and optional cache/aux_loss
        """
        pass

    def init_cache(self, batch_size: int, max_seq_len: int) -> Any:
        """
        Initialize cache for inference.

        Default implementation returns None. Override in subclasses
        that support caching.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length

        Returns:
            Initial cache state (type depends on block)
        """
        return None


class SequenceModule(nn.Module, ABC):
    """
    Abstract base for sequence modeling modules.

    This is the core component within a block that processes
    the sequence (attention, SSM, or RNN).
    """

    @abstractmethod
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> tuple[mx.array, Any | None]:
        """
        Process sequence.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            mask: Optional mask
            cache: Optional cache

        Returns:
            - Output tensor
            - Updated cache (or None)
        """
        pass
