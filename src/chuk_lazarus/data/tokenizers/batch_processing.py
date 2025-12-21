"""Batch processing utilities for tokenizers with Pydantic models."""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class PaddingSide(str, Enum):
    """Side to apply padding."""

    LEFT = "left"
    RIGHT = "right"


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    pad_token_id: int


class BatchResult(BaseModel):
    """Result of batch encoding/padding."""

    input_ids: list[list[int]] = Field(description="Padded token sequences")
    attention_mask: list[list[int]] | None = Field(
        default=None, description="Attention masks (1=real, 0=padding)"
    )


class SequenceStats(BaseModel):
    """Statistics about sequence lengths in a batch."""

    min_length: int = Field(ge=0, description="Minimum sequence length")
    max_length: int = Field(ge=0, description="Maximum sequence length")
    mean_length: float = Field(ge=0.0, description="Mean sequence length")
    total_tokens: int = Field(ge=0, description="Total tokens across all sequences")
    count: int = Field(ge=0, description="Number of sequences")


class ChunkConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(gt=0, description="Maximum tokens per chunk")
    overlap: int = Field(ge=0, default=0, description="Overlapping tokens between chunks")
    add_special_tokens: bool = Field(default=True, description="Add BOS/EOS to each chunk")


def encode_batch(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = True,
) -> list[list[int]]:
    """
    Efficiently encode multiple texts.

    Args:
        texts: List of text strings to encode
        tokenizer: Tokenizer instance with encode method
        add_special_tokens: Whether to add special tokens

    Returns:
        List of token ID lists (variable length)
    """
    return [tokenizer.encode(text, add_special_tokens=add_special_tokens) for text in texts]


def decode_batch(
    token_ids_batch: list[list[int]],
    tokenizer: TokenizerProtocol,
    skip_special_tokens: bool = True,
) -> list[str]:
    """
    Efficiently decode multiple token sequences.

    Args:
        token_ids_batch: List of token ID lists
        tokenizer: Tokenizer instance with decode method
        skip_special_tokens: Whether to skip special tokens in output

    Returns:
        List of decoded strings
    """
    results = []
    for token_ids in token_ids_batch:
        if skip_special_tokens and hasattr(tokenizer, "convert_ids_to_tokens"):
            tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
            if isinstance(tokens, list):
                results.append(" ".join(tokens))
            else:
                results.append(tokenizer.decode(token_ids))
        else:
            results.append(tokenizer.decode(token_ids))
    return results


def pad_batch(
    sequences: list[list[int]],
    pad_token_id: int,
    max_length: int | None = None,
    padding_side: PaddingSide = PaddingSide.RIGHT,
    return_attention_mask: bool = True,
    truncate: bool = False,
) -> BatchResult:
    """
    Pad sequences to uniform length with attention masks.

    Args:
        sequences: List of token ID lists
        pad_token_id: Token ID to use for padding
        max_length: Maximum length (uses longest sequence if None)
        padding_side: LEFT or RIGHT padding
        return_attention_mask: Whether to return attention masks
        truncate: Whether to truncate sequences longer than max_length

    Returns:
        BatchResult with padded sequences and optional masks
    """
    if not sequences:
        return BatchResult(input_ids=[], attention_mask=[] if return_attention_mask else None)

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    attention_masks = []

    for seq in sequences:
        if truncate and len(seq) > max_length:
            seq = seq[:max_length]

        padding_length = max_length - len(seq)

        if padding_length > 0:
            padding = [pad_token_id] * padding_length
            if padding_side == PaddingSide.RIGHT:
                padded_seq = seq + padding
                mask = [1] * len(seq) + [0] * padding_length
            else:
                padded_seq = padding + seq
                mask = [0] * padding_length + [1] * len(seq)
        else:
            padded_seq = seq[:max_length] if truncate else seq
            mask = [1] * len(padded_seq)

        padded_sequences.append(padded_seq)
        attention_masks.append(mask)

    return BatchResult(
        input_ids=padded_sequences,
        attention_mask=attention_masks if return_attention_mask else None,
    )


def create_batch(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_length: int | None = None,
    padding: bool = True,
    truncation: bool = False,
    add_special_tokens: bool = True,
    padding_side: PaddingSide = PaddingSide.RIGHT,
    return_attention_mask: bool = True,
) -> BatchResult:
    """
    Create a batch from texts with encoding, padding, and truncation.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate long sequences
        add_special_tokens: Whether to add special tokens
        padding_side: LEFT or RIGHT padding
        return_attention_mask: Whether to return attention masks

    Returns:
        BatchResult with input_ids and optionally attention_mask
    """
    encoded = encode_batch(texts, tokenizer, add_special_tokens=add_special_tokens)

    if not padding:
        return BatchResult(
            input_ids=encoded,
            attention_mask=[[1] * len(seq) for seq in encoded] if return_attention_mask else None,
        )

    return pad_batch(
        sequences=encoded,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
        padding_side=padding_side,
        return_attention_mask=return_attention_mask,
        truncate=truncation,
    )


def chunk_text(
    text: str,
    tokenizer: TokenizerProtocol,
    config: ChunkConfig,
) -> list[list[int]]:
    """
    Split text into chunks of a specified token size.

    Args:
        text: Text to chunk
        tokenizer: Tokenizer instance
        config: Chunking configuration

    Returns:
        List of token ID chunks
    """
    all_tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(all_tokens) <= config.chunk_size:
        if config.add_special_tokens:
            return [tokenizer.encode(text, add_special_tokens=True)]
        return [all_tokens]

    chunks = []
    step = config.chunk_size - config.overlap
    if step <= 0:
        step = 1

    for i in range(0, len(all_tokens), step):
        chunk = all_tokens[i : i + config.chunk_size]
        if config.add_special_tokens and hasattr(tokenizer, "build_inputs_with_special_tokens"):
            chunk = tokenizer.build_inputs_with_special_tokens(chunk)
        chunks.append(chunk)

        if i + config.chunk_size >= len(all_tokens):
            break

    return chunks


def get_sequence_lengths(sequences: list[list[int]]) -> SequenceStats:
    """
    Get statistics about sequence lengths in a batch.

    Args:
        sequences: List of token ID lists

    Returns:
        SequenceStats with length statistics
    """
    if not sequences:
        return SequenceStats(min_length=0, max_length=0, mean_length=0.0, total_tokens=0, count=0)

    lengths = [len(seq) for seq in sequences]
    return SequenceStats(
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=sum(lengths) / len(lengths),
        total_tokens=sum(lengths),
        count=len(lengths),
    )
