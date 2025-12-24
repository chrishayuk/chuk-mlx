"""
Bag-of-Words Character Tokenizer for classification tasks.

A tokenizer that converts text into normalized character count vectors.
Useful for simple classification tasks where word order doesn't matter.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable


class BoWTokenizerConfig(BaseModel):
    """Configuration for BoWCharacterTokenizer."""

    model_config = {"frozen": True}

    lowercase: bool = Field(default=True, description="Lowercase input before encoding")
    normalize: bool = Field(default=True, description="L1-normalize the output vectors")


class BoWCharacterTokenizer:
    """
    Bag-of-Words character tokenizer for classification.

    Converts text into fixed-size vectors of character frequencies.
    Each dimension corresponds to a character in the vocabulary.

    Unlike sequence tokenizers that return token IDs, this returns
    float vectors suitable for classification models.

    Example:
        >>> tokenizer = BoWCharacterTokenizer.from_corpus(["cat", "dog"])
        >>> tokenizer.vocab_size
        6  # unique chars: a, c, d, g, o, t

        >>> vec = tokenizer.encode("cat")
        >>> len(vec) == tokenizer.vocab_size
        True

        >>> # Batch encoding
        >>> vecs = tokenizer.encode_batch(["cat", "dog"])
        >>> len(vecs)
        2
    """

    def __init__(
        self,
        charset: str | Iterable[str],
        config: BoWTokenizerConfig | None = None,
    ) -> None:
        """
        Initialize with a character set.

        Args:
            charset: String or iterable of characters for vocabulary.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or BoWTokenizerConfig()

        # Build vocabulary
        self._char_to_id: dict[str, int] = {}
        for i, char in enumerate(sorted(set(charset))):
            self._char_to_id[char] = i

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        texts: Iterable[str],
        config: BoWTokenizerConfig | None = None,
    ) -> BoWCharacterTokenizer:
        """
        Learn vocabulary from a corpus of texts.

        Args:
            texts: Iterable of text strings to extract characters from.
            config: Optional configuration.

        Returns:
            BoWCharacterTokenizer with vocabulary from corpus.
        """
        effective_config = config or BoWTokenizerConfig()

        chars: set[str] = set()
        for text in texts:
            if effective_config.lowercase:
                text = text.lower()
            chars.update(text)

        return cls(charset=chars, config=effective_config)

    @classmethod
    def from_ascii(
        cls,
        config: BoWTokenizerConfig | None = None,
    ) -> BoWCharacterTokenizer:
        """Create tokenizer with printable ASCII characters."""
        return cls(charset=string.printable, config=config)

    @classmethod
    def from_ascii_lowercase(
        cls,
        config: BoWTokenizerConfig | None = None,
    ) -> BoWCharacterTokenizer:
        """Create tokenizer with lowercase ASCII letters and common punctuation."""
        charset = string.ascii_lowercase + string.digits + " .,!?'\"-\n\t"
        effective_config = config or BoWTokenizerConfig(lowercase=True)
        return cls(charset=charset, config=effective_config)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary (number of output dimensions)."""
        return len(self._char_to_id)

    # -------------------------------------------------------------------------
    # Core encoding methods
    # -------------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        """
        Encode text as a bag-of-words vector.

        Args:
            text: Text to encode.

        Returns:
            List of floats representing character frequencies.
            Length equals vocab_size.
        """
        if self.config.lowercase:
            text = text.lower()

        counts = [0.0] * self.vocab_size

        for char in text:
            if char in self._char_to_id:
                counts[self._char_to_id[char]] += 1

        if self.config.normalize:
            total = sum(counts)
            if total > 0:
                counts = [c / total for c in counts]

        return counts

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts.

        Args:
            texts: List of texts to encode.

        Returns:
            List of BoW vectors.
        """
        return [self.encode(text) for text in texts]

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping."""
        return self._char_to_id.copy()

    def get_charset(self) -> str:
        """Get the character set as a string."""
        return "".join(sorted(self._char_to_id.keys()))

    def __contains__(self, char: str) -> bool:
        """Check if a character is in the vocabulary."""
        return char in self._char_to_id

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation."""
        charset = self.get_charset()
        preview = charset[:20] + "..." if len(charset) > 20 else charset
        return f"BoWCharacterTokenizer(vocab_size={self.vocab_size}, charset='{preview}')"
