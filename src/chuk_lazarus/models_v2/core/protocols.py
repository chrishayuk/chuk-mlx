"""
Protocol definitions for model components.

Protocols define the interfaces that components must implement.
This enables composition and type-safe dependency injection.

All protocols use abstract Tensor types - actual implementation
depends on the backend (MLX, PyTorch, JAX).
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

# Abstract tensor type - concrete type depends on backend
Tensor = TypeVar("Tensor")


@runtime_checkable
class Embedding(Protocol):
    """Protocol for token embeddings."""

    vocab_size: int
    hidden_size: int

    def __call__(self, input_ids: Any) -> Any:
        """
        Embed token IDs.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)

        Returns:
            Embeddings, shape (batch, seq_len, hidden_size)
        """
        ...

    def as_linear(self, hidden: Any) -> Any:
        """
        Use embedding weights as linear projection (for tied embeddings).

        Args:
            hidden: Hidden states, shape (batch, seq_len, hidden_size)

        Returns:
            Logits, shape (batch, seq_len, vocab_size)
        """
        ...


@runtime_checkable
class PositionEmbedding(Protocol):
    """Protocol for position embeddings."""

    def __call__(
        self,
        x: Any,
        offset: int = 0,
    ) -> Any:
        """
        Apply position embeddings.

        Args:
            x: Input tensor (queries or keys)
            offset: Position offset (for KV cache)

        Returns:
            Position-encoded tensor
        """
        ...


@runtime_checkable
class Norm(Protocol):
    """Protocol for normalization layers."""

    def __call__(self, x: Any) -> Any:
        """
        Apply normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        ...


@runtime_checkable
class Attention(Protocol):
    """Protocol for attention mechanisms."""

    num_heads: int
    head_dim: int

    def __call__(
        self,
        x: Any,
        mask: Any | None = None,
        cache: tuple[Any, Any] | None = None,
    ) -> tuple[Any, tuple[Any, Any] | None]:
        """
        Compute attention.

        Args:
            x: Input hidden states, shape (batch, seq_len, hidden_size)
            mask: Attention mask
            cache: Optional (key_cache, value_cache) tuple

        Returns:
            output: Attention output, shape (batch, seq_len, hidden_size)
            cache: Updated (key_cache, value_cache) or None
        """
        ...


@runtime_checkable
class FFN(Protocol):
    """Protocol for feed-forward networks."""

    hidden_size: int
    intermediate_size: int

    def __call__(self, x: Any) -> Any:
        """
        Apply feed-forward transformation.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)

        Returns:
            Output, shape (batch, seq_len, hidden_size)
        """
        ...


@runtime_checkable
class SSM(Protocol):
    """Protocol for State Space Models."""

    hidden_size: int
    state_size: int

    def __call__(
        self,
        x: Any,
        state: Any | None = None,
    ) -> tuple[Any, Any | None]:
        """
        Apply SSM transformation.

        Args:
            x: Input, shape (batch, seq_len, hidden_size)
            state: Optional SSM state

        Returns:
            output: Output, shape (batch, seq_len, hidden_size)
            state: Updated state or None
        """
        ...


@runtime_checkable
class RecurrentCell(Protocol):
    """Protocol for recurrent cells (LSTM, GRU, etc.)."""

    hidden_size: int

    def __call__(
        self,
        x: Any,
        state: tuple[Any, ...] | None = None,
    ) -> tuple[Any, tuple[Any, ...]]:
        """
        Process one step.

        Args:
            x: Input, shape (batch, input_size)
            state: Optional hidden state tuple

        Returns:
            output: Output, shape (batch, hidden_size)
            state: Updated state tuple
        """
        ...


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


# Type aliases for cache types (backend-agnostic)
AttentionCache = tuple[Any, Any]  # (key_cache, value_cache)
BlockCache = Any  # Block-specific cache type
LayerCache = list[Any]  # List of per-layer caches
