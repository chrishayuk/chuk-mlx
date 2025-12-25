"""
Protocols for complete models.

Models are the top-level abstractions that combine:
- Backbone (embeddings + blocks)
- Head(s) (task-specific outputs)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Protocol for complete models."""

    hidden_size: int

    def __call__(
        self,
        input_ids: Any,
        cache: Any = None,
    ) -> tuple[Any, Any]:
        """
        Full forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            cache: Optional model cache

        Returns:
            output: Model output (logits, etc.)
            cache: Updated cache or None
        """
        ...

    def set_mode(self, mode: str) -> None:
        """Set model mode (train/inference)."""
        ...
