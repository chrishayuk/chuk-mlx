"""
Custom tokenizer implementation using Pydantic models and enums.

This is a simple whitespace-based tokenizer for testing and development.
For production use, prefer HuggingFace tokenizers or SentencePiece.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizer

from chuk_lazarus.data.tokenizers.types import SpecialTokenName
from chuk_lazarus.data.tokenizers.vocab_utils import load_vocabulary, save_vocabulary

if TYPE_CHECKING:
    from collections.abc import Sequence


class CustomTokenizer(PreTrainedTokenizer):
    """
    A simple whitespace tokenizer with special token support.

    Uses VocabularyData Pydantic model and SpecialTokenName enum
    to avoid magic strings and dictionary goop.
    """

    def __init__(self, vocab_file: str | Path, **kwargs) -> None:
        """
        Initialize the tokenizer from a vocabulary file.

        Args:
            vocab_file: Path to the vocabulary JSON file.
            **kwargs: Additional arguments passed to PreTrainedTokenizer.

        Raises:
            ValueError: If required special tokens are missing.
        """
        # Load vocabulary using Pydantic model
        self._vocab_data = load_vocabulary(vocab_file)

        # Create merged vocab (regular + special tokens)
        self._merged_vocab: dict[str, int] = {
            **self._vocab_data.vocab,
            **self._vocab_data.special_tokens,
        }

        # Create reverse mapping from IDs to tokens
        self._ids_to_tokens: dict[int, str] = {
            token_id: token for token, token_id in self._merged_vocab.items()
        }

        # Call base class initializer
        super().__init__(**kwargs)

        # Set special token IDs using enum (check both standard and alternative forms)
        self._pad_token_id = self._get_special_token_id(SpecialTokenName.PAD, required=True)
        self._unk_token_id = self._get_special_token_id(SpecialTokenName.UNK, required=True)
        self._bos_token_id = self._get_special_token_id(
            SpecialTokenName.BOS, SpecialTokenName.BOS_ALT, required=True
        )
        self._eos_token_id = self._get_special_token_id(
            SpecialTokenName.EOS, SpecialTokenName.EOS_ALT, required=True
        )

    def _get_special_token_id(
        self,
        *tokens: SpecialTokenName,
        required: bool = False,
    ) -> int | None:
        """
        Get ID for a special token, checking multiple alternatives.

        Args:
            *tokens: Special tokens to check (in order of preference).
            required: If True, raise ValueError when no token found.

        Returns:
            Token ID or None if not found and not required.

        Raises:
            ValueError: If required and no token found.
        """
        for token in tokens:
            token_id = self._vocab_data.get_special_token_id(token)
            if token_id is not None:
                return token_id

        if required:
            token_names = ", ".join(t.value for t in tokens)
            raise ValueError(f"Required special token not found: {token_names}")

        return None

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        """Unknown token ID."""
        return self._unk_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Beginning of sequence token ID."""
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """End of sequence token ID."""
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return len(self._merged_vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary including special tokens."""
        return self._merged_vocab.copy()

    def tokenize(self, text: str) -> list[str]:
        """Split by whitespace to tokenize."""
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID, falling back to UNK if not found."""
        return self._merged_vocab.get(token, self._unk_token_id)  # type: ignore[return-value]

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token, falling back to UNK token string if not found."""
        unk_str = SpecialTokenName.UNK.value
        return self._ids_to_tokens.get(index, unk_str)

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        """Convert a single token or list of tokens to IDs."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(
        self,
        ids: int | Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str | list[str]:
        """Convert a single ID or list of IDs to tokens."""
        special_ids = frozenset(self._vocab_data.special_tokens.values())

        if isinstance(ids, int):
            return self._convert_id_to_token(ids)

        return [
            self._convert_id_to_token(id_)
            for id_ in ids
            if not (skip_special_tokens and id_ in special_ids)
        ]

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        """Add special tokens to a sequence of token IDs."""
        bos = self._bos_token_id
        eos = self._eos_token_id

        if token_ids_1 is None:
            return [bos] + token_ids_0 + [eos]  # type: ignore[list-item]

        return [bos] + token_ids_0 + [eos] + token_ids_1 + [eos]  # type: ignore[list-item]

    def encode(
        self,
        text: str,
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,  # noqa: ARG002
        return_tensors: str | None = None,  # noqa: ARG002
    ) -> list[int]:
        """Tokenize and convert text to input IDs."""
        tokens = self.tokenize(text)
        input_ids = list(self.convert_tokens_to_ids(tokens))  # type: ignore[arg-type]

        if add_special_tokens:
            pair_tokens = self.tokenize(text_pair) if text_pair else None
            pair_ids = (
                list(self.convert_tokens_to_ids(pair_tokens))  # type: ignore[arg-type]
                if pair_tokens
                else None
            )
            input_ids = self.build_inputs_with_special_tokens(input_ids, pair_ids)

        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        if padding and max_length:
            input_ids = self._pad_sequence(input_ids, max_length)

        return input_ids

    def _pad_sequence(
        self,
        sequence: list[int],
        max_length: int,
        pad_to_multiple_of: int | None = None,
    ) -> list[int]:
        """
        Pad a sequence to the specified length.

        Args:
            sequence: Token IDs to pad.
            max_length: Target length.
            pad_to_multiple_of: Round up to multiple of this value.

        Returns:
            Padded sequence.
        """
        target_length = max_length

        if pad_to_multiple_of is not None:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            )

        # Truncate if needed, preserving EOS
        if len(sequence) >= target_length:
            return sequence[: target_length - 1] + [self._eos_token_id]  # type: ignore[list-item]

        # Ensure EOS at end
        if sequence[-1] != self._eos_token_id:
            sequence = sequence + [self._eos_token_id]  # type: ignore[list-item]

        # Pad to target length
        padding_needed = target_length - len(sequence)
        if padding_needed > 0:
            return sequence + [self._pad_token_id] * padding_needed  # type: ignore[list-item]

        return sequence

    def pad(
        self,
        sequence: list[int],
        padding: bool = True,  # noqa: ARG002
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = False,
    ) -> list[int] | tuple[list[int], list[int]]:
        """
        Pad a sequence of token IDs.

        Args:
            sequence: Token IDs to pad.
            padding: Whether to apply padding.
            max_length: Target length (defaults to sequence length).
            pad_to_multiple_of: Round up to multiple of this value.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Padded sequence, or tuple of (sequence, attention_mask).
        """
        if not isinstance(sequence, list) or not all(isinstance(i, int) for i in sequence):
            raise ValueError("Input must be a list of integers.")

        target_length = max_length if max_length is not None else len(sequence)
        padded = self._pad_sequence(sequence, target_length, pad_to_multiple_of)

        if return_attention_mask:
            # 1 for real tokens (including EOS), 0 for padding
            # Count non-padding tokens in the result
            num_real_tokens = sum(1 for t in padded if t != self._pad_token_id)
            attention_mask = [1] * num_real_tokens + [0] * (len(padded) - num_real_tokens)
            return padded, attention_mask

        return padded

    def save_vocabulary(self, save_directory: str | Path) -> tuple[str]:
        """Save the vocabulary to the specified directory."""
        vocab_file = save_vocabulary(self._vocab_data, save_directory)
        return (str(vocab_file),)
