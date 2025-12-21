"""Tokenizer-dataset compatibility scoring."""

from typing import Protocol

from pydantic import BaseModel, Field

from .coverage import analyze_coverage
from .entropy import analyze_entropy


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class FitScoreConfig(BaseModel):
    """Configuration for fit score calculation."""

    # Weight for each component (should sum to 1.0)
    coverage_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    compression_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    entropy_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    vocab_util_weight: float = Field(default=0.25, ge=0.0, le=1.0)

    # Thresholds for scoring
    ideal_tokens_per_word: float = Field(default=1.3, gt=0.0)
    max_acceptable_tokens_per_word: float = Field(default=2.5, gt=0.0)
    ideal_unk_rate: float = Field(default=0.0, ge=0.0)
    max_acceptable_unk_rate: float = Field(default=0.01, ge=0.0)


class FitScore(BaseModel):
    """Tokenizer-dataset fit score."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall fit score (0-1, higher is better)"
    )
    coverage_score: float = Field(ge=0.0, le=1.0, description="UNK rate score")
    compression_score: float = Field(ge=0.0, le=1.0, description="Tokens per word score")
    entropy_score: float = Field(ge=0.0, le=1.0, description="Token entropy score")
    vocab_utilization_score: float = Field(ge=0.0, le=1.0, description="Vocabulary usage score")
    recommendation: str = Field(description="Human-readable recommendation")
    details: dict = Field(default_factory=dict, description="Detailed metrics")


class TokenizerComparison(BaseModel):
    """Comparison of two tokenizers on same dataset."""

    tokenizer1_name: str = Field(description="First tokenizer name")
    tokenizer2_name: str = Field(description="Second tokenizer name")
    tokenizer1_score: FitScore = Field(description="First tokenizer fit score")
    tokenizer2_score: FitScore = Field(description="Second tokenizer fit score")
    winner: str = Field(description="Better tokenizer name")
    score_delta: float = Field(description="Score difference")
    comparison_notes: list[str] = Field(default_factory=list, description="Comparison notes")


def _score_unk_rate(unk_rate: float, config: FitScoreConfig) -> float:
    """Score UNK rate (lower is better)."""
    if unk_rate <= config.ideal_unk_rate:
        return 1.0
    if unk_rate >= config.max_acceptable_unk_rate:
        return 0.0
    # Linear interpolation
    range_size = config.max_acceptable_unk_rate - config.ideal_unk_rate
    return 1.0 - (unk_rate - config.ideal_unk_rate) / range_size


def _score_tokens_per_word(tpw: float, config: FitScoreConfig) -> float:
    """Score tokens per word (closer to ideal is better)."""
    if tpw <= config.ideal_tokens_per_word:
        return 1.0
    if tpw >= config.max_acceptable_tokens_per_word:
        return 0.0
    # Linear interpolation
    range_size = config.max_acceptable_tokens_per_word - config.ideal_tokens_per_word
    return 1.0 - (tpw - config.ideal_tokens_per_word) / range_size


def _score_entropy(normalized_entropy: float) -> float:
    """Score entropy (moderate is best)."""
    # Very low entropy = repetitive, very high = chaotic
    # Optimal is around 0.6-0.8
    if normalized_entropy < 0.3:
        return normalized_entropy / 0.3 * 0.5  # Scale 0-0.3 to 0-0.5
    if normalized_entropy > 0.9:
        return 1.0 - (normalized_entropy - 0.9) / 0.1 * 0.5  # Scale 0.9-1.0 to 1.0-0.5
    # Sweet spot: 0.3-0.9 maps to 0.5-1.0-0.5 (peak at 0.7)
    if normalized_entropy <= 0.7:
        return 0.5 + (normalized_entropy - 0.3) / 0.4 * 0.5
    return 1.0 - (normalized_entropy - 0.7) / 0.2 * 0.5


def _score_vocab_utilization(util: float) -> float:
    """Score vocabulary utilization."""
    # Very low = wasted parameters, very high = maybe underfitting
    # Sweet spot: 0.2-0.8
    if util < 0.05:
        return util / 0.05 * 0.3  # Very low: 0-0.3
    if util < 0.2:
        return 0.3 + (util - 0.05) / 0.15 * 0.4  # Low: 0.3-0.7
    if util <= 0.8:
        return 0.7 + (util - 0.2) / 0.6 * 0.3  # Optimal: 0.7-1.0
    return 1.0 - (util - 0.8) / 0.2 * 0.2  # Very high: slight penalty


def calculate_fit_score(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: FitScoreConfig | None = None,
) -> FitScore:
    """
    Calculate how well a tokenizer fits a dataset.

    Args:
        texts: Sample texts from the dataset
        tokenizer: Tokenizer to evaluate
        config: Scoring configuration

    Returns:
        FitScore with overall and component scores
    """
    if config is None:
        config = FitScoreConfig()

    # Analyze coverage
    coverage = analyze_coverage(texts, tokenizer, include_fragments=False)

    # Analyze entropy
    entropy = analyze_entropy(texts, tokenizer)

    # Calculate component scores
    coverage_score = _score_unk_rate(coverage.unk_rate, config)
    compression_score = _score_tokens_per_word(coverage.tokens_per_word, config)
    entropy_score = _score_entropy(entropy.normalized_entropy)
    vocab_score = _score_vocab_utilization(coverage.vocab_utilization)

    # Weighted overall score
    overall = (
        coverage_score * config.coverage_weight
        + compression_score * config.compression_weight
        + entropy_score * config.entropy_weight
        + vocab_score * config.vocab_util_weight
    )

    # Generate recommendation
    if overall >= 0.8:
        rec = "Excellent fit. Tokenizer is well-suited for this dataset."
    elif overall >= 0.6:
        rec = "Good fit. Minor optimizations possible."
    elif overall >= 0.4:
        rec = "Moderate fit. Consider domain-specific tokenizer or vocabulary extension."
    else:
        rec = "Poor fit. Recommend different tokenizer or significant vocabulary surgery."

    details = {
        "tokens_per_word": coverage.tokens_per_word,
        "unk_rate": coverage.unk_rate,
        "vocab_utilization": coverage.vocab_utilization,
        "entropy": entropy.entropy,
        "normalized_entropy": entropy.normalized_entropy,
        "total_texts": len(texts),
        "total_tokens": coverage.total_tokens,
    }

    return FitScore(
        overall_score=overall,
        coverage_score=coverage_score,
        compression_score=compression_score,
        entropy_score=entropy_score,
        vocab_utilization_score=vocab_score,
        recommendation=rec,
        details=details,
    )


def compare_tokenizers_for_dataset(
    texts: list[str],
    tokenizer1: TokenizerProtocol,
    tokenizer2: TokenizerProtocol,
    name1: str = "tokenizer1",
    name2: str = "tokenizer2",
    config: FitScoreConfig | None = None,
) -> TokenizerComparison:
    """
    Compare two tokenizers on the same dataset.

    Args:
        texts: Sample texts from the dataset
        tokenizer1: First tokenizer
        tokenizer2: Second tokenizer
        name1: Name for first tokenizer
        name2: Name for second tokenizer
        config: Scoring configuration

    Returns:
        TokenizerComparison with scores and recommendation
    """
    score1 = calculate_fit_score(texts, tokenizer1, config)
    score2 = calculate_fit_score(texts, tokenizer2, config)

    delta = score1.overall_score - score2.overall_score
    winner = name1 if delta > 0 else name2

    notes = []
    if abs(score1.coverage_score - score2.coverage_score) > 0.1:
        better = name1 if score1.coverage_score > score2.coverage_score else name2
        notes.append(f"{better} has better vocabulary coverage")

    if abs(score1.compression_score - score2.compression_score) > 0.1:
        better = name1 if score1.compression_score > score2.compression_score else name2
        notes.append(f"{better} has better compression (fewer tokens per word)")

    if abs(delta) < 0.05:
        notes.append("Tokenizers are roughly equivalent for this dataset")

    return TokenizerComparison(
        tokenizer1_name=name1,
        tokenizer2_name=name2,
        tokenizer1_score=score1,
        tokenizer2_score=score2,
        winner=winner,
        score_delta=abs(delta),
        comparison_notes=notes,
    )
