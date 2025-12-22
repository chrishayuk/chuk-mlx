"""
OOV (Out-of-Vocabulary) and rare token reporting.

Identifies tokens that are unknown or rarely used, helping
understand vocabulary coverage and potential issues.
"""

from collections import Counter
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    @property
    def unk_token_id(self) -> int | None: ...


class TokenFrequencyBand(str, Enum):
    """Frequency bands for token analysis."""

    SINGLETON = "singleton"  # Appears exactly once
    RARE = "rare"  # 2-10 occurrences
    UNCOMMON = "uncommon"  # 11-100 occurrences
    COMMON = "common"  # 101-1000 occurrences
    VERY_COMMON = "very_common"  # 1000+ occurrences


class RareTokenInfo(BaseModel):
    """Information about a rare or OOV token."""

    token_id: int = Field(description="Token ID")
    token_str: str = Field(description="Token string representation")
    count: int = Field(description="Number of occurrences")
    band: TokenFrequencyBand = Field(description="Frequency band")
    example_contexts: list[str] = Field(
        default_factory=list, description="Example contexts where token appears"
    )


class OOVReport(BaseModel):
    """Complete OOV and rare token analysis."""

    # Basic stats
    total_tokens: int = Field(description="Total tokens in corpus")
    unique_tokens: int = Field(description="Number of unique tokens")
    vocab_utilization: float = Field(description="Fraction of vocab used (0-1)")

    # OOV stats
    unk_count: int = Field(description="Number of UNK tokens")
    unk_rate: float = Field(description="UNK rate (0-1)")
    unk_contexts: list[str] = Field(default_factory=list, description="Example contexts with UNK")

    # Frequency distribution
    singletons: int = Field(description="Tokens appearing exactly once")
    singleton_rate: float = Field(description="Rate of singletons (0-1)")
    rare_tokens: int = Field(description="Tokens appearing 2-10 times")
    rare_rate: float = Field(description="Rate of rare tokens (0-1)")

    # Detailed rare token info
    top_rare_tokens: list[RareTokenInfo] = Field(
        default_factory=list, description="Most impactful rare tokens"
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Suggestions based on analysis"
    )


def _get_frequency_band(count: int) -> TokenFrequencyBand:
    """Categorize token by frequency."""
    if count == 1:
        return TokenFrequencyBand.SINGLETON
    elif count <= 10:
        return TokenFrequencyBand.RARE
    elif count <= 100:
        return TokenFrequencyBand.UNCOMMON
    elif count <= 1000:
        return TokenFrequencyBand.COMMON
    else:
        return TokenFrequencyBand.VERY_COMMON


def get_frequency_bands(
    texts: list[str],
    tokenizer: TokenizerProtocol,
) -> dict[TokenFrequencyBand, int]:
    """
    Count tokens in each frequency band.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use

    Returns:
        Dictionary mapping band to count of unique tokens in that band
    """
    token_counts: Counter[int] = Counter()

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(token_ids)

    bands: dict[TokenFrequencyBand, int] = dict.fromkeys(TokenFrequencyBand, 0)

    for count in token_counts.values():
        band = _get_frequency_band(count)
        bands[band] += 1

    return bands


def find_rare_tokens(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_frequency: int = 10,
    top_k: int = 20,
    include_contexts: bool = True,
    context_window: int = 50,
) -> list[RareTokenInfo]:
    """
    Find rare tokens in a corpus.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        max_frequency: Maximum frequency to be considered "rare"
        top_k: Number of rare tokens to return
        include_contexts: Whether to include example contexts
        context_window: Characters of context around token

    Returns:
        List of RareTokenInfo for most impactful rare tokens
    """
    token_counts: Counter[int] = Counter()
    token_contexts: dict[int, list[str]] = {}

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(token_ids)

        if include_contexts:
            # Find context for each token
            decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
            pos = 0
            for tid, tok_str in zip(token_ids, decoded_tokens):
                if token_counts[tid] <= max_frequency + 5:  # Only track rare-ish tokens
                    start = max(0, pos - context_window)
                    end = min(len(text), pos + len(tok_str) + context_window)
                    context = text[start:end]
                    if tid not in token_contexts:
                        token_contexts[tid] = []
                    if len(token_contexts[tid]) < 3:  # Max 3 examples
                        token_contexts[tid].append(context)
                pos += len(tok_str)

    # Filter to rare tokens and sort by count (ascending - rarest first)
    rare_tokens = [(tid, count) for tid, count in token_counts.items() if count <= max_frequency]
    rare_tokens.sort(key=lambda x: x[1])

    results: list[RareTokenInfo] = []
    for tid, count in rare_tokens[:top_k]:
        token_str = tokenizer.decode([tid])
        results.append(
            RareTokenInfo(
                token_id=tid,
                token_str=token_str,
                count=count,
                band=_get_frequency_band(count),
                example_contexts=token_contexts.get(tid, []),
            )
        )

    return results


def analyze_oov(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    vocab_size: int | None = None,
    top_rare: int = 10,
) -> OOVReport:
    """
    Comprehensive OOV and rare token analysis.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use
        vocab_size: Vocabulary size (for utilization calc)
        top_rare: Number of rare tokens to include in report

    Returns:
        OOVReport with full analysis
    """
    if not texts:
        return OOVReport(
            total_tokens=0,
            unique_tokens=0,
            vocab_utilization=0,
            unk_count=0,
            unk_rate=0,
            singletons=0,
            singleton_rate=0,
            rare_tokens=0,
            rare_rate=0,
        )

    token_counts: Counter[int] = Counter()
    unk_id = tokenizer.unk_token_id
    unk_contexts: list[str] = []

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(token_ids)

        # Track UNK contexts
        if unk_id is not None and unk_id in token_ids and len(unk_contexts) < 5:
            unk_contexts.append(text[:100] + "..." if len(text) > 100 else text)

    total_tokens = sum(token_counts.values())
    unique_tokens = len(token_counts)

    # UNK stats
    unk_count = token_counts.get(unk_id, 0) if unk_id is not None else 0
    unk_rate = unk_count / total_tokens if total_tokens > 0 else 0

    # Frequency distribution
    singletons = sum(1 for c in token_counts.values() if c == 1)
    rare = sum(1 for c in token_counts.values() if 2 <= c <= 10)

    singleton_rate = singletons / unique_tokens if unique_tokens > 0 else 0
    rare_rate = rare / unique_tokens if unique_tokens > 0 else 0

    # Vocab utilization
    if vocab_size:
        vocab_utilization = unique_tokens / vocab_size
    else:
        vocab_utilization = 0

    # Find top rare tokens
    top_rare_tokens = find_rare_tokens(texts, tokenizer, top_k=top_rare)

    # Generate recommendations
    recommendations: list[str] = []

    if unk_rate > 0.01:
        recommendations.append(
            f"High UNK rate ({unk_rate:.2%}). Consider vocabulary extension or byte fallback."
        )

    if singleton_rate > 0.5:
        recommendations.append(
            f"High singleton rate ({singleton_rate:.2%}). Vocabulary may be undertrained."
        )

    if vocab_utilization > 0 and vocab_utilization < 0.3:
        recommendations.append(
            f"Low vocab utilization ({vocab_utilization:.2%}). Vocabulary may be oversized."
        )

    if rare_rate > 0.3:
        recommendations.append(
            f"Many rare tokens ({rare_rate:.2%}). Consider frequency-based filtering."
        )

    return OOVReport(
        total_tokens=total_tokens,
        unique_tokens=unique_tokens,
        vocab_utilization=vocab_utilization,
        unk_count=unk_count,
        unk_rate=unk_rate,
        unk_contexts=unk_contexts,
        singletons=singletons,
        singleton_rate=singleton_rate,
        rare_tokens=rare,
        rare_rate=rare_rate,
        top_rare_tokens=top_rare_tokens,
        recommendations=recommendations,
    )
