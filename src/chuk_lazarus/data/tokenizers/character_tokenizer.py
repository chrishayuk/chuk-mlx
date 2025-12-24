"""
Character-level tokenizer for classification and small-scale experiments.

A lightweight tokenizer that maps individual characters to token IDs.
Useful for:
- Classification experiments (sentiment, topic, etc.)
- Character-level language models
- Testing and debugging pipelines
- Domains where subword tokenization is overkill
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from chuk_lazarus.data.tokenizers.types import SpecialTokenName

if TYPE_CHECKING:
    from collections.abc import Iterable


class CharacterTokenizerConfig(BaseModel):
    """Configuration for CharacterTokenizer."""

    model_config = {"frozen": True}

    # Special token IDs (reserved at the start)
    pad_token_id: int = Field(default=0, description="Padding token ID")
    unk_token_id: int = Field(default=1, description="Unknown token ID")
    bos_token_id: int = Field(default=2, description="Beginning of sequence token ID")
    eos_token_id: int = Field(default=3, description="End of sequence token ID")

    # Whether to lowercase input
    lowercase: bool = Field(default=False, description="Lowercase input before tokenizing")


class CharacterTokenizer:
    """
    A simple character-level tokenizer.

    Maps each character to a unique token ID. Supports factory methods
    for common charsets (ASCII, digits, etc.) or learning from a corpus.

    Example:
        >>> tokenizer = CharacterTokenizer.from_ascii()
        >>> tokenizer.encode("hello")
        [2, 104, 101, 108, 108, 111, 3]  # with BOS/EOS

        >>> tokenizer = CharacterTokenizer.from_corpus(["cat", "dog"])
        >>> tokenizer.vocab_size
        10  # 4 special + 6 unique chars (c, a, t, d, o, g)
    """

    def __init__(
        self,
        charset: str | Iterable[str],
        config: CharacterTokenizerConfig | None = None,
    ) -> None:
        """
        Initialize with a character set.

        Args:
            charset: String or iterable of characters to include in vocabulary.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or CharacterTokenizerConfig()

        # Build vocabulary: special tokens first, then characters
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}

        # Reserve special token IDs
        self._special_tokens = {
            SpecialTokenName.PAD.value: self.config.pad_token_id,
            SpecialTokenName.UNK.value: self.config.unk_token_id,
            SpecialTokenName.BOS.value: self.config.bos_token_id,
            SpecialTokenName.EOS.value: self.config.eos_token_id,
        }

        for token, token_id in self._special_tokens.items():
            self._char_to_id[token] = token_id
            self._id_to_char[token_id] = token

        # Add characters starting after special tokens
        next_id = max(self._special_tokens.values()) + 1
        for char in charset:
            if char not in self._char_to_id:
                self._char_to_id[char] = next_id
                self._id_to_char[next_id] = char
                next_id += 1

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_ascii(cls, config: CharacterTokenizerConfig | None = None) -> CharacterTokenizer:
        """
        Create tokenizer with printable ASCII characters.

        Includes: letters, digits, punctuation, whitespace.
        """
        return cls(charset=string.printable, config=config)

    @classmethod
    def from_ascii_lowercase(
        cls,
        config: CharacterTokenizerConfig | None = None,
    ) -> CharacterTokenizer:
        """Create tokenizer with lowercase ASCII letters, digits, and common punctuation."""
        charset = string.ascii_lowercase + string.digits + " .,!?'\"-\n\t"
        effective_config = config or CharacterTokenizerConfig(lowercase=True)
        return cls(charset=charset, config=effective_config)

    @classmethod
    def from_digits(cls, config: CharacterTokenizerConfig | None = None) -> CharacterTokenizer:
        """Create tokenizer with digits only (0-9)."""
        return cls(charset=string.digits, config=config)

    @classmethod
    def from_charset(
        cls,
        charset: str,
        config: CharacterTokenizerConfig | None = None,
    ) -> CharacterTokenizer:
        """Create tokenizer from an explicit character set string."""
        return cls(charset=charset, config=config)

    @classmethod
    def from_corpus(
        cls,
        texts: Iterable[str],
        config: CharacterTokenizerConfig | None = None,
    ) -> CharacterTokenizer:
        """
        Learn vocabulary from a corpus of texts.

        Args:
            texts: Iterable of text strings to extract characters from.
            config: Optional configuration.

        Returns:
            CharacterTokenizer with vocabulary from corpus.
        """
        effective_config = config or CharacterTokenizerConfig()

        # Collect unique characters
        chars: set[str] = set()
        for text in texts:
            if effective_config.lowercase:
                text = text.lower()
            chars.update(text)

        # Sort for deterministic ordering
        sorted_chars = sorted(chars)
        return cls(charset=sorted_chars, config=effective_config)

    # -------------------------------------------------------------------------
    # Properties (TokenizerProtocol compatible)
    # -------------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary including special tokens."""
        return len(self._char_to_id)

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.config.pad_token_id

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.config.unk_token_id

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.config.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.config.eos_token_id

    # -------------------------------------------------------------------------
    # Core methods (TokenizerProtocol compatible)
    # -------------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary."""
        return self._char_to_id.copy()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Optional maximum length (truncates if exceeded).

        Returns:
            List of token IDs.
        """
        if self.config.lowercase:
            text = text.lower()

        # Convert characters to IDs
        ids = [self._char_to_id.get(char, self.config.unk_token_id) for char in text]

        # Add special tokens
        if add_special_tokens:
            ids = [self.config.bos_token_id] + ids + [self.config.eos_token_id]

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            # Ensure EOS at end if we truncated
            if add_special_tokens and ids[-1] != self.config.eos_token_id:
                ids[-1] = self.config.eos_token_id

        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        special_ids = frozenset(self._special_tokens.values())

        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            char = self._id_to_char.get(token_id, SpecialTokenName.UNK.value)
            # Don't output special token strings when skipping
            if skip_special_tokens and char in self._special_tokens:
                continue
            chars.append(char)

        return "".join(chars)

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into individual characters.

        Args:
            text: Text to tokenize.

        Returns:
            List of character strings.
        """
        if self.config.lowercase:
            text = text.lower()
        return list(text)

    # -------------------------------------------------------------------------
    # Batch operations
    # -------------------------------------------------------------------------

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
    ) -> list[list[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of texts to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Optional maximum length.
            padding: Whether to pad to max_length.

        Returns:
            List of token ID lists.
        """
        encoded = [self.encode(text, add_special_tokens, max_length) for text in texts]

        if padding and max_length:
            encoded = [self._pad_sequence(ids, max_length) for ids in encoded]

        return encoded

    def decode_batch(
        self,
        batch: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decode multiple token ID sequences.

        Args:
            batch: List of token ID lists.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded strings.
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    def _pad_sequence(self, ids: list[int], max_length: int) -> list[int]:
        """Pad a sequence to max_length."""
        if len(ids) >= max_length:
            return ids[:max_length]
        return ids + [self.config.pad_token_id] * (max_length - len(ids))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_charset(self) -> str:
        """Get the character set (excluding special tokens)."""
        special_ids = frozenset(self._special_tokens.values())
        chars = [
            char
            for char, token_id in sorted(self._char_to_id.items(), key=lambda x: x[1])
            if token_id not in special_ids
        ]
        return "".join(chars)

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
        return f"CharacterTokenizer(vocab_size={self.vocab_size}, charset='{preview}')"
