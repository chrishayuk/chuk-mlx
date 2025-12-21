"""Token coverage and UNK rate analysis."""

import re
from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class FragmentAnalysis(BaseModel):
    """Analysis of token fragments (subword pieces)."""

    total_tokens: int = Field(ge=0, description="Total tokens in analysis")
    fragment_tokens: int = Field(ge=0, description="Tokens that are fragments")
    fragment_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of fragments")
    top_fragments: list[tuple[str, int]] = Field(
        default_factory=list, description="Most common fragment tokens"
    )
    whitespace_tokens: int = Field(ge=0, description="Tokens starting with space marker")
    punctuation_tokens: int = Field(ge=0, description="Pure punctuation tokens")


class CoverageReport(BaseModel):
    """Comprehensive token coverage report."""

    total_texts: int = Field(ge=0, description="Number of texts analyzed")
    total_words: int = Field(ge=0, description="Total words (whitespace split)")
    total_tokens: int = Field(ge=0, description="Total tokens produced")
    tokens_per_word: float = Field(ge=0.0, description="Average tokens per word")
    unk_count: int = Field(ge=0, description="Number of UNK tokens")
    unk_rate: float = Field(ge=0.0, le=1.0, description="Ratio of UNK tokens")
    vocab_utilization: float = Field(ge=0.0, le=1.0, description="Fraction of vocab used")
    unique_tokens_used: int = Field(ge=0, description="Unique token IDs seen")
    vocab_size: int = Field(ge=0, description="Total vocabulary size")
    fragment_analysis: FragmentAnalysis | None = Field(
        default=None, description="Fragment token analysis"
    )
    domain_warnings: list[str] = Field(
        default_factory=list, description="Potential domain mismatch warnings"
    )


def get_unk_rate(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    unk_token_id: int | None = None,
) -> float:
    """
    Calculate the UNK token rate across texts.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance
        unk_token_id: UNK token ID (auto-detected if None)

    Returns:
        Ratio of UNK tokens (0.0 to 1.0)
    """
    if unk_token_id is None:
        vocab = tokenizer.get_vocab()
        unk_token_id = vocab.get("<unk>", vocab.get("[UNK]", -1))

    total_tokens = 0
    unk_count = 0

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        unk_count += sum(1 for tid in token_ids if tid == unk_token_id)

    return unk_count / total_tokens if total_tokens > 0 else 0.0


def get_tokens_per_word(
    texts: list[str],
    tokenizer: TokenizerProtocol,
) -> float:
    """
    Calculate average tokens per word.

    High values (>2.0) indicate poor tokenizer fit for the domain.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance

    Returns:
        Average tokens per whitespace-separated word
    """
    total_words = 0
    total_tokens = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)

    return total_tokens / total_words if total_words > 0 else 0.0


def analyze_fragments(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    top_n: int = 20,
) -> FragmentAnalysis:
    """
    Analyze fragment/subword token patterns.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance
        top_n: Number of top fragments to return

    Returns:
        FragmentAnalysis with fragment statistics
    """
    fragment_counter: Counter[str] = Counter()
    total_tokens = 0
    fragment_count = 0
    whitespace_count = 0
    punctuation_count = 0

    # Common subword markers
    fragment_markers = ("##", "▁", "Ġ", "@@")
    punctuation_pattern = re.compile(r"^[^\w\s]+$")

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)

        for tid in token_ids:
            try:
                token_str = tokenizer.decode([tid])
            except Exception:
                continue

            # Check for fragment markers
            is_fragment = False
            for marker in fragment_markers:
                if marker in token_str:
                    is_fragment = True
                    break

            # Short tokens that aren't complete words are likely fragments
            if len(token_str.strip()) < 3 and not token_str.strip().isalpha():
                is_fragment = True

            if is_fragment:
                fragment_count += 1
                fragment_counter[token_str] += 1

            # Check whitespace prefix (sentencepiece style)
            if token_str.startswith("▁") or token_str.startswith("Ġ"):
                whitespace_count += 1

            # Check pure punctuation
            clean = token_str.strip()
            if clean and punctuation_pattern.match(clean):
                punctuation_count += 1

    top_fragments = fragment_counter.most_common(top_n)

    return FragmentAnalysis(
        total_tokens=total_tokens,
        fragment_tokens=fragment_count,
        fragment_ratio=fragment_count / total_tokens if total_tokens > 0 else 0.0,
        top_fragments=top_fragments,
        whitespace_tokens=whitespace_count,
        punctuation_tokens=punctuation_count,
    )


def analyze_coverage(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    unk_token_id: int | None = None,
    include_fragments: bool = True,
) -> CoverageReport:
    """
    Comprehensive token coverage analysis.

    Args:
        texts: List of texts to analyze
        tokenizer: Tokenizer instance
        unk_token_id: UNK token ID (auto-detected if None)
        include_fragments: Whether to include fragment analysis

    Returns:
        CoverageReport with full analysis
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    if unk_token_id is None:
        unk_token_id = vocab.get("<unk>", vocab.get("[UNK]", -1))

    total_words = 0
    total_tokens = 0
    unk_count = 0
    unique_tokens: set[int] = set()

    for text in texts:
        words = text.split()
        total_words += len(words)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        unique_tokens.update(token_ids)

        unk_count += sum(1 for tid in token_ids if tid == unk_token_id)

    tokens_per_word = total_tokens / total_words if total_words > 0 else 0.0
    unk_rate = unk_count / total_tokens if total_tokens > 0 else 0.0
    vocab_utilization = len(unique_tokens) / vocab_size if vocab_size > 0 else 0.0

    # Generate warnings
    warnings = []
    if tokens_per_word > 2.0:
        warnings.append(f"High tokens/word ({tokens_per_word:.2f}) - tokenizer may not fit domain")
    if unk_rate > 0.01:
        warnings.append(f"High UNK rate ({unk_rate:.2%}) - vocabulary coverage issues")
    if vocab_utilization < 0.1:
        warnings.append(
            f"Low vocab utilization ({vocab_utilization:.2%}) - may be over-parameterized"
        )

    fragment_analysis = None
    if include_fragments:
        fragment_analysis = analyze_fragments(texts, tokenizer)

    return CoverageReport(
        total_texts=len(texts),
        total_words=total_words,
        total_tokens=total_tokens,
        tokens_per_word=tokens_per_word,
        unk_count=unk_count,
        unk_rate=unk_rate,
        vocab_utilization=vocab_utilization,
        unique_tokens_used=len(unique_tokens),
        vocab_size=vocab_size,
        fragment_analysis=fragment_analysis,
        domain_warnings=warnings,
    )
