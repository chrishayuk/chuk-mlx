"""Token statistics and analysis utilities with Pydantic models."""

from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class CoverageStats(BaseModel):
    """Statistics about vocabulary coverage."""

    total_tokens: int = Field(description="Total number of tokens")
    known_tokens: int = Field(description="Number of known tokens")
    unknown_tokens: int = Field(description="Number of unknown tokens")
    coverage_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of known tokens")


class CompressionStats(BaseModel):
    """Statistics about tokenization compression."""

    char_count: int = Field(description="Number of characters in original text")
    byte_count: int = Field(description="Number of bytes in UTF-8 encoding")
    token_count: int = Field(description="Number of tokens")
    chars_per_token: float = Field(ge=0.0, description="Average characters per token")
    bytes_per_token: float = Field(ge=0.0, description="Average bytes per token")


class TokenFrequency(BaseModel):
    """Frequency information for a single token."""

    token_id: int = Field(description="Token ID")
    decoded: str = Field(description="Decoded token string")
    frequency: int = Field(ge=0, description="Occurrence count")


class LengthDistribution(BaseModel):
    """Distribution of token lengths."""

    length_counts: dict[int, int] = Field(description="Map of length to count")
    total_tokens: int = Field(description="Total tokens analyzed")
    avg_length: float = Field(ge=0.0, description="Average token length")
    max_length: int = Field(ge=0, description="Maximum token length")
    min_length: int = Field(ge=0, description="Minimum token length")


def get_token_frequencies(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> Counter[int]:
    """
    Count token occurrences across a corpus.

    Args:
        texts: List of text strings to analyze
        tokenizer: Tokenizer instance with encode method
        add_special_tokens: Whether to include special tokens in counts

    Returns:
        Counter mapping token IDs to their frequencies
    """
    counter: Counter[int] = Counter()
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        counter.update(token_ids)
    return counter


def get_vocabulary_coverage(
    text: str,
    tokenizer: TokenizerProtocol,
    unk_token_id: int | None = None,
) -> CoverageStats:
    """
    Calculate what percentage of tokens map to known vs unknown tokens.

    Args:
        text: Text to analyze
        tokenizer: Tokenizer instance
        unk_token_id: ID of the unknown token (auto-detected if None)

    Returns:
        CoverageStats with coverage statistics
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if unk_token_id is None:
        vocab = tokenizer.get_vocab()
        unk_token_id = vocab.get("<unk>", vocab.get("[UNK]", -1))

    total = len(token_ids)
    unknown = sum(1 for tid in token_ids if tid == unk_token_id)
    known = total - unknown

    return CoverageStats(
        total_tokens=total,
        known_tokens=known,
        unknown_tokens=unknown,
        coverage_ratio=known / total if total > 0 else 1.0,
    )


def get_token_length_distribution(
    tokenizer: TokenizerProtocol,
    sample_ids: list[int] | None = None,
) -> LengthDistribution:
    """
    Get distribution of token lengths (in characters when decoded).

    Args:
        tokenizer: Tokenizer instance
        sample_ids: Specific token IDs to analyze (uses full vocab if None)

    Returns:
        LengthDistribution with histogram and stats
    """
    if sample_ids is None:
        vocab = tokenizer.get_vocab()
        sample_ids = list(vocab.values())

    length_counts: dict[int, int] = {}
    for token_id in sample_ids:
        try:
            decoded = tokenizer.decode([token_id])
            length = len(decoded)
            length_counts[length] = length_counts.get(length, 0) + 1
        except Exception:
            continue

    sorted_counts = dict(sorted(length_counts.items()))
    total = sum(sorted_counts.values())
    lengths = list(sorted_counts.keys())

    avg_length = (
        sum(length * count for length, count in sorted_counts.items()) / total if total > 0 else 0.0
    )

    return LengthDistribution(
        length_counts=sorted_counts,
        total_tokens=total,
        avg_length=avg_length,
        max_length=max(lengths) if lengths else 0,
        min_length=min(lengths) if lengths else 0,
    )


def get_top_tokens(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    n: int = 20,
    add_special_tokens: bool = False,
) -> list[TokenFrequency]:
    """
    Get the most frequent tokens in a corpus.

    Args:
        texts: List of text strings to analyze
        tokenizer: Tokenizer instance
        n: Number of top tokens to return
        add_special_tokens: Whether to include special tokens

    Returns:
        List of TokenFrequency for top tokens
    """
    frequencies = get_token_frequencies(texts, tokenizer, add_special_tokens)
    top = frequencies.most_common(n)

    result = []
    for token_id, count in top:
        try:
            decoded = tokenizer.decode([token_id]).strip()
        except Exception:
            decoded = f"<id:{token_id}>"
        result.append(TokenFrequency(token_id=token_id, decoded=decoded, frequency=count))

    return result


def get_rare_tokens(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_freq: int = 1,
    add_special_tokens: bool = False,
) -> list[TokenFrequency]:
    """
    Get tokens that appear rarely in a corpus.

    Args:
        texts: List of text strings to analyze
        tokenizer: Tokenizer instance
        max_freq: Maximum frequency to be considered "rare"
        add_special_tokens: Whether to include special tokens

    Returns:
        List of TokenFrequency for rare tokens
    """
    frequencies = get_token_frequencies(texts, tokenizer, add_special_tokens)

    result = []
    for token_id, count in frequencies.items():
        if count <= max_freq:
            try:
                decoded = tokenizer.decode([token_id]).strip()
            except Exception:
                decoded = f"<id:{token_id}>"
            result.append(TokenFrequency(token_id=token_id, decoded=decoded, frequency=count))

    return sorted(result, key=lambda x: x.frequency)


def calculate_compression_ratio(
    text: str,
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> CompressionStats:
    """
    Calculate the compression ratio of tokenization.

    Args:
        text: Text to analyze
        tokenizer: Tokenizer instance
        add_special_tokens: Whether to include special tokens

    Returns:
        CompressionStats with compression statistics
    """
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    token_count = len(token_ids)

    return CompressionStats(
        char_count=char_count,
        byte_count=byte_count,
        token_count=token_count,
        chars_per_token=char_count / token_count if token_count > 0 else 0.0,
        bytes_per_token=byte_count / token_count if token_count > 0 else 0.0,
    )
