"""Token entropy and distribution analysis."""

import math
from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class TokenDistribution(BaseModel):
    """Token frequency distribution statistics."""

    total_tokens: int = Field(ge=0, description="Total tokens analyzed")
    unique_tokens: int = Field(ge=0, description="Unique token types")
    type_token_ratio: float = Field(ge=0.0, description="Unique/total ratio")
    top_tokens: list[tuple[int, str, int]] = Field(
        default_factory=list, description="Top tokens: (id, decoded, count)"
    )
    singleton_count: int = Field(ge=0, description="Tokens appearing only once")
    singleton_ratio: float = Field(ge=0.0, description="Ratio of singletons")


class EntropyReport(BaseModel):
    """Token entropy analysis report."""

    entropy: float = Field(ge=0.0, description="Shannon entropy in bits")
    normalized_entropy: float = Field(
        ge=0.0, le=1.0, description="Entropy normalized by max possible"
    )
    perplexity: float = Field(ge=1.0, description="Perplexity (2^entropy)")
    distribution: TokenDistribution = Field(description="Token distribution stats")
    uniformity_score: float = Field(ge=0.0, le=1.0, description="How uniform the distribution is")
    concentration_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of tokens in top 10%")


def get_token_distribution(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    top_n: int = 20,
) -> TokenDistribution:
    """
    Analyze token frequency distribution.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance
        top_n: Number of top tokens to return

    Returns:
        TokenDistribution with frequency statistics
    """
    counter: Counter[int] = Counter()

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(token_ids)

    total = sum(counter.values())
    unique = len(counter)

    # Get top tokens with decoded strings
    top_tokens = []
    for tid, count in counter.most_common(top_n):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            decoded = f"<id:{tid}>"
        top_tokens.append((tid, decoded, count))

    # Count singletons
    singletons = sum(1 for count in counter.values() if count == 1)

    return TokenDistribution(
        total_tokens=total,
        unique_tokens=unique,
        type_token_ratio=unique / total if total > 0 else 0.0,
        top_tokens=top_tokens,
        singleton_count=singletons,
        singleton_ratio=singletons / unique if unique > 0 else 0.0,
    )


def calculate_entropy(frequencies: Counter[int]) -> float:
    """
    Calculate Shannon entropy from token frequencies.

    Args:
        frequencies: Counter of token frequencies

    Returns:
        Entropy in bits
    """
    total = sum(frequencies.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in frequencies.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def analyze_entropy(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    top_n: int = 20,
) -> EntropyReport:
    """
    Comprehensive entropy analysis of tokenization.

    High entropy = diverse token usage (good for generalization)
    Low entropy = concentrated usage (may indicate repetitive data)

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance
        top_n: Number of top tokens in distribution

    Returns:
        EntropyReport with entropy metrics
    """
    counter: Counter[int] = Counter()

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(token_ids)

    total = sum(counter.values())
    unique = len(counter)

    # Calculate entropy
    entropy = calculate_entropy(counter)

    # Maximum possible entropy (uniform distribution)
    max_entropy = math.log2(unique) if unique > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Perplexity
    perplexity = 2**entropy if entropy > 0 else 1.0

    # Uniformity: how close to uniform distribution
    # Perfect uniformity = all tokens have same frequency
    if unique > 0 and total > 0:
        expected_freq = total / unique
        variance = sum((count - expected_freq) ** 2 for count in counter.values())
        max_variance = total**2  # Worst case: all in one token
        uniformity = 1.0 - (variance / max_variance) if max_variance > 0 else 1.0
    else:
        uniformity = 0.0

    # Concentration: what fraction of tokens are in top 10%
    sorted_counts = sorted(counter.values(), reverse=True)
    top_10_percent = max(1, len(sorted_counts) // 10)
    top_sum = sum(sorted_counts[:top_10_percent])
    concentration = top_sum / total if total > 0 else 0.0

    # Get distribution
    distribution = get_token_distribution(texts, tokenizer, top_n)

    return EntropyReport(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        perplexity=perplexity,
        distribution=distribution,
        uniformity_score=uniformity,
        concentration_ratio=concentration,
    )
