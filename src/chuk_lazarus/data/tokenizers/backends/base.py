"""
Base tokenizer backend protocol and types.

Defines the contract that all tokenizer backends must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class BackendType(str, Enum):
    """Available tokenizer backend types."""

    HUGGINGFACE = "huggingface"  # HuggingFace/SentencePiece compatible
    FAST = "fast"  # MLX Data CharTrie-based

    # Alias for backwards compatibility
    COMPAT = "huggingface"


class TokenizationResult(BaseModel):
    """Result of tokenization operation."""

    token_ids: list[int] = Field(description="Token IDs")
    tokens: list[str] = Field(default_factory=list, description="Token strings (optional)")
    offsets: list[tuple[int, int]] = Field(
        default_factory=list, description="Character offsets (start, end) for each token"
    )


class BatchTokenizationResult(BaseModel):
    """Result of batch tokenization."""

    results: list[TokenizationResult] = Field(description="Per-sample results")
    total_tokens: int = Field(description="Total tokens across all samples")


class BackendInfo(BaseModel):
    """Information about a tokenizer backend."""

    backend_type: BackendType = Field(description="Backend type")
    vocab_size: int = Field(description="Vocabulary size")
    supports_parallel: bool = Field(description="Whether backend supports parallel tokenization")
    supports_offsets: bool = Field(description="Whether backend provides character offsets")
    is_available: bool = Field(description="Whether backend is available in current environment")


@runtime_checkable
class TokenizerBackend(Protocol):
    """Protocol for tokenizer backends."""

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets: bool = False,
    ) -> TokenizationResult:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_offsets: Whether to compute character offsets

        Returns:
            TokenizationResult with token IDs and optional metadata
        """
        ...

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        num_workers: int = 1,
    ) -> BatchTokenizationResult:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            num_workers: Number of parallel workers (for fast backend)

        Returns:
            BatchTokenizationResult with all results
        """
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        ...

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary as token -> id mapping."""
        ...

    def get_info(self) -> BackendInfo:
        """Return information about this backend."""
        ...


class BaseBackend(ABC):
    """Abstract base class for tokenizer backends."""

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets: bool = False,
    ) -> TokenizationResult:
        """Encode text to token IDs."""
        ...

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        num_workers: int = 1,
    ) -> BatchTokenizationResult:
        """
        Default batch encoding (sequential).

        Subclasses can override for parallel processing.
        """
        results = [self.encode(text, add_special_tokens) for text in texts]
        total_tokens = sum(len(r.token_ids) for r in results)
        return BatchTokenizationResult(results=results, total_tokens=total_tokens)

    @abstractmethod
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary."""
        ...

    def get_info(self) -> BackendInfo:
        """Return backend information."""
        return BackendInfo(
            backend_type=self.backend_type,
            vocab_size=self.vocab_size,
            supports_parallel=False,
            supports_offsets=False,
            is_available=True,
        )
