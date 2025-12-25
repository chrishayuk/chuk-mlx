"""
Protocols for model blocks.

Blocks are composed of components:
- Block: Base protocol for all block types
- Backbone: Stack of blocks with embeddings
- Head: Task-specific output projections
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Block(Protocol):
    """Protocol for model blocks (transformer, mamba, etc.)."""

    hidden_size: int

    def __call__(
        self,
        x: Any,
        mask: Any | None = None,
        cache: Any = None,
    ) -> tuple[Any, Any]:
        """
        Forward pass through block.

        Args:
            x: Input hidden states, shape (batch, seq_len, hidden_size)
            mask: Optional attention/causal mask
            cache: Optional block-specific cache

        Returns:
            output: Output hidden states, shape (batch, seq_len, hidden_size)
            cache: Updated cache
        """
        ...


@runtime_checkable
class Backbone(Protocol):
    """Protocol for model backbones (stack of blocks)."""

    num_layers: int
    hidden_size: int

    def __call__(
        self,
        x: Any,
        mask: Any | None = None,
        cache: list[Any] | None = None,
    ) -> tuple[Any, list[Any] | None]:
        """
        Forward pass through backbone.

        Args:
            x: Input embeddings, shape (batch, seq_len, hidden_size)
            mask: Optional attention/causal mask
            cache: Optional list of per-layer caches

        Returns:
            hidden: Final hidden states, shape (batch, seq_len, hidden_size)
            cache: Updated cache list or None
        """
        ...


@runtime_checkable
class Head(Protocol):
    """Protocol for output heads."""

    def __call__(self, hidden: Any) -> Any:
        """
        Project hidden states to output space.

        Args:
            hidden: Hidden states, shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor (shape depends on head type)
        """
        ...
