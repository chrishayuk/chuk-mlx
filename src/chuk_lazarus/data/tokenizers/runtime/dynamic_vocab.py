"""Runtime vocabulary extension utilities."""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class VocabExtension(BaseModel):
    """A vocabulary extension entry."""

    token_str: str = Field(description="Token string")
    token_id: int = Field(ge=0, description="Assigned token ID")
    original_tokens: list[int] = Field(
        default_factory=list, description="Original tokenization of the string"
    )
    embedding_initialized: bool = Field(
        default=False, description="Whether embedding has been initialized"
    )
    init_method: str = Field(default="mean", description="Embedding init method")


class DynamicVocab(BaseModel):
    """Dynamic vocabulary manager for runtime token injection."""

    base_vocab_size: int = Field(ge=0, description="Original vocabulary size")
    extensions: dict[str, VocabExtension] = Field(default_factory=dict, description="Added tokens")
    next_id: int = Field(description="Next available token ID")
    frozen_base: bool = Field(default=True, description="Whether base vocab is frozen")

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerProtocol) -> "DynamicVocab":
        """Create from existing tokenizer."""
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        return cls(base_vocab_size=vocab_size, next_id=vocab_size)

    def add_token(
        self,
        token_str: str,
        tokenizer: TokenizerProtocol,
        init_method: str = "mean",
    ) -> VocabExtension:
        """
        Add a new token to the vocabulary.

        Args:
            token_str: Token string to add
            tokenizer: Tokenizer for getting original encoding
            init_method: How to initialize embedding ("mean", "random", "zero")

        Returns:
            VocabExtension entry

        Raises:
            ValueError: If token already exists
        """
        if token_str in self.extensions:
            raise ValueError(f"Token '{token_str}' already added")

        vocab = tokenizer.get_vocab()
        if token_str in vocab:
            raise ValueError(f"Token '{token_str}' already in base vocabulary")

        # Get original tokenization
        original_ids = tokenizer.encode(token_str, add_special_tokens=False)

        extension = VocabExtension(
            token_str=token_str,
            token_id=self.next_id,
            original_tokens=original_ids,
            embedding_initialized=False,
            init_method=init_method,
        )

        self.extensions[token_str] = extension
        self.next_id += 1
        return extension

    def get_all_tokens(self) -> list[VocabExtension]:
        """Get all extension tokens."""
        return list(self.extensions.values())

    def get_token_id(self, token_str: str) -> int | None:
        """Get ID for an extension token."""
        ext = self.extensions.get(token_str)
        return ext.token_id if ext else None

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size including extensions."""
        return self.base_vocab_size + len(self.extensions)


def extend_vocab_runtime(
    vocab: DynamicVocab,
    tokens: list[str],
    tokenizer: TokenizerProtocol,
    init_method: str = "mean",
) -> list[VocabExtension]:
    """
    Extend vocabulary with multiple tokens at runtime.

    Args:
        vocab: Dynamic vocabulary manager
        tokens: List of token strings to add
        tokenizer: Tokenizer for encoding
        init_method: Embedding initialization method

    Returns:
        List of VocabExtension entries for added tokens
    """
    extensions = []
    for token in tokens:
        try:
            ext = vocab.add_token(token, tokenizer, init_method)
            extensions.append(ext)
        except ValueError:
            continue  # Skip duplicates
    return extensions


def create_embedding_slot(
    extension: VocabExtension,
    embedding_dim: int,
    init_method: str | None = None,
) -> list[float]:
    """
    Create initialized embedding for a new token.

    In practice, this would use the model's embedding matrix.
    This is a simplified version that creates placeholder values.

    Args:
        extension: Token extension entry
        embedding_dim: Embedding dimension
        init_method: Override init method from extension

    Returns:
        Initialized embedding as list of floats
    """
    import random

    method = init_method or extension.init_method

    if method == "zero":
        return [0.0] * embedding_dim
    elif method == "random":
        # Small random initialization
        return [random.gauss(0, 0.02) for _ in range(embedding_dim)]
    elif method == "mean":
        # Would average embeddings of original tokens in real implementation
        # Here we just return small random as placeholder
        return [random.gauss(0, 0.01) for _ in range(embedding_dim)]
    else:
        raise ValueError(f"Unknown init method: {method}")
