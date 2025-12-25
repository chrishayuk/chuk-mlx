"""
Protocols for model components.

Components are the atomic building blocks:
- Embedding: Token embeddings
- PositionEmbedding: Positional encodings (RoPE, ALiBi, etc.)
- Norm: Normalization layers (LayerNorm, RMSNorm, etc.)
- FFN: Feed-forward networks (MLP, SwiGLU, MoE, etc.)
- Attention: Self-attention mechanisms (MHA, GQA, etc.)
- SSM: State Space Models (Mamba, etc.)
- RecurrentCell: Recurrent cells (LSTM, GRU, etc.)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
