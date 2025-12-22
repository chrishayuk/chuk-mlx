"""
Before/after vocabulary swap analysis.

Compares tokenization behavior between two tokenizers to understand
the impact of vocabulary changes on a corpus.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class VocabSwapReport(BaseModel):
    """Report comparing two tokenizers on the same corpus."""

    # Tokenizer info
    tokenizer1_name: str = Field(description="First tokenizer name")
    tokenizer2_name: str = Field(description="Second tokenizer name")
    tokenizer1_vocab_size: int = Field(description="First tokenizer vocab size")
    tokenizer2_vocab_size: int = Field(description="Second tokenizer vocab size")

    # Corpus stats
    total_samples: int = Field(description="Total samples analyzed")
    total_chars: int = Field(description="Total characters in corpus")

    # Token counts
    tokens1_total: int = Field(description="Total tokens with tokenizer 1")
    tokens2_total: int = Field(description="Total tokens with tokenizer 2")
    token_count_diff: int = Field(description="Difference (tok2 - tok1)")
    token_count_ratio: float = Field(description="Ratio (tok2 / tok1)")

    # Compression
    chars_per_token1: float = Field(description="Chars per token for tokenizer 1")
    chars_per_token2: float = Field(description="Chars per token for tokenizer 2")
    compression_improvement: float = Field(description="Improvement ratio (>1 = tok2 better)")

    # Per-sample analysis
    samples_improved: int = Field(description="Samples with fewer tokens in tok2")
    samples_same: int = Field(description="Samples with same token count")
    samples_worse: int = Field(description="Samples with more tokens in tok2")

    improvement_rate: float = Field(description="Fraction of samples improved")

    # Biggest changes
    max_improvement: int = Field(description="Best improvement (tokens saved)")
    max_regression: int = Field(description="Worst regression (extra tokens)")
    mean_change: float = Field(description="Mean token count change per sample")

    # Examples
    improved_examples: list[dict] = Field(
        default_factory=list, description="Examples of improved tokenization"
    )
    regressed_examples: list[dict] = Field(
        default_factory=list, description="Examples of worse tokenization"
    )

    # Training impact
    training_speedup: float = Field(
        description="Estimated training speedup factor (>1 = faster with tok2)"
    )
    memory_reduction: float = Field(
        description="Estimated memory reduction (>0 = less memory with tok2)"
    )

    # Recommendations
    recommendations: list[str] = Field(default_factory=list)


def compare_vocab_impact(
    texts: list[str],
    tokenizer1: TokenizerProtocol,
    tokenizer2: TokenizerProtocol,
    tokenizer1_name: str = "tokenizer1",
    tokenizer2_name: str = "tokenizer2",
    max_examples: int = 5,
) -> VocabSwapReport:
    """
    Compare tokenization impact between two tokenizers.

    Args:
        texts: List of text samples
        tokenizer1: First (baseline) tokenizer
        tokenizer2: Second (new) tokenizer
        tokenizer1_name: Name for tokenizer 1
        tokenizer2_name: Name for tokenizer 2
        max_examples: Maximum examples to include

    Returns:
        VocabSwapReport with detailed comparison
    """
    if not texts:
        return VocabSwapReport(
            tokenizer1_name=tokenizer1_name,
            tokenizer2_name=tokenizer2_name,
            tokenizer1_vocab_size=len(tokenizer1.get_vocab()),
            tokenizer2_vocab_size=len(tokenizer2.get_vocab()),
            total_samples=0,
            total_chars=0,
            tokens1_total=0,
            tokens2_total=0,
            token_count_diff=0,
            token_count_ratio=1,
            chars_per_token1=0,
            chars_per_token2=0,
            compression_improvement=1,
            samples_improved=0,
            samples_same=0,
            samples_worse=0,
            improvement_rate=0,
            max_improvement=0,
            max_regression=0,
            mean_change=0,
            training_speedup=1,
            memory_reduction=0,
        )

    # Tokenize all samples
    results: list[dict] = []
    total_chars = 0

    for text in texts:
        tokens1 = tokenizer1.encode(text, add_special_tokens=True)
        tokens2 = tokenizer2.encode(text, add_special_tokens=True)

        total_chars += len(text)

        results.append(
            {
                "text": text[:200] if len(text) > 200 else text,
                "len1": len(tokens1),
                "len2": len(tokens2),
                "diff": len(tokens2) - len(tokens1),
            }
        )

    # Compute statistics
    tokens1_total = sum(r["len1"] for r in results)
    tokens2_total = sum(r["len2"] for r in results)

    samples_improved = sum(1 for r in results if r["diff"] < 0)
    samples_same = sum(1 for r in results if r["diff"] == 0)
    samples_worse = sum(1 for r in results if r["diff"] > 0)

    diffs = [r["diff"] for r in results]
    max_improvement = -min(diffs) if diffs else 0
    max_regression = max(diffs) if diffs else 0
    mean_change = sum(diffs) / len(diffs) if diffs else 0

    # Find examples
    sorted_by_improvement = sorted(results, key=lambda r: r["diff"])
    improved_examples = sorted_by_improvement[:max_examples]
    regressed_examples = sorted_by_improvement[-max_examples:][::-1]

    # Filter to actual improvements/regressions
    improved_examples = [e for e in improved_examples if e["diff"] < 0]
    regressed_examples = [e for e in regressed_examples if e["diff"] > 0]

    # Training impact estimates
    # Speedup is inverse of token ratio (fewer tokens = faster)
    token_ratio = tokens2_total / tokens1_total if tokens1_total > 0 else 1
    training_speedup = 1 / token_ratio if token_ratio > 0 else 1

    # Memory reduction (approximate, linear with token count)
    memory_reduction = 1 - token_ratio if token_ratio < 1 else 0

    # Recommendations
    recommendations: list[str] = []

    if token_ratio < 0.9:
        pct = (1 - token_ratio) * 100
        recommendations.append(
            f"{tokenizer2_name} produces {pct:.1f}% fewer tokens. "
            f"Estimated {training_speedup:.2f}x speedup."
        )
    elif token_ratio > 1.1:
        pct = (token_ratio - 1) * 100
        recommendations.append(
            f"{tokenizer2_name} produces {pct:.1f}% more tokens. May slow training."
        )
    else:
        recommendations.append(
            f"Token counts similar between tokenizers ({token_ratio:.2f}x ratio)."
        )

    if samples_improved > samples_worse * 2:
        recommendations.append(
            f"{tokenizer2_name} improves {samples_improved}/{len(results)} samples."
        )
    elif samples_worse > samples_improved * 2:
        recommendations.append(f"{tokenizer2_name} worsens {samples_worse}/{len(results)} samples.")

    return VocabSwapReport(
        tokenizer1_name=tokenizer1_name,
        tokenizer2_name=tokenizer2_name,
        tokenizer1_vocab_size=len(tokenizer1.get_vocab()),
        tokenizer2_vocab_size=len(tokenizer2.get_vocab()),
        total_samples=len(texts),
        total_chars=total_chars,
        tokens1_total=tokens1_total,
        tokens2_total=tokens2_total,
        token_count_diff=tokens2_total - tokens1_total,
        token_count_ratio=token_ratio,
        chars_per_token1=total_chars / tokens1_total if tokens1_total > 0 else 0,
        chars_per_token2=total_chars / tokens2_total if tokens2_total > 0 else 0,
        compression_improvement=1 / token_ratio if token_ratio > 0 else 1,
        samples_improved=samples_improved,
        samples_same=samples_same,
        samples_worse=samples_worse,
        improvement_rate=samples_improved / len(results) if results else 0,
        max_improvement=max_improvement,
        max_regression=max_regression,
        mean_change=mean_change,
        improved_examples=improved_examples[:max_examples],
        regressed_examples=regressed_examples[:max_examples],
        training_speedup=training_speedup,
        memory_reduction=memory_reduction,
        recommendations=recommendations,
    )


def estimate_retokenization_cost(
    texts: list[str],
    old_tokenizer: TokenizerProtocol,
    new_tokenizer: TokenizerProtocol,
) -> dict[str, float | int]:
    """
    Estimate the cost of retokenizing a corpus with a new tokenizer.

    Args:
        texts: List of text samples
        old_tokenizer: Current tokenizer
        new_tokenizer: New tokenizer to evaluate

    Returns:
        Dictionary with cost estimates
    """
    if not texts:
        return {
            "total_samples": 0,
            "boundary_changes": 0,
            "boundary_change_rate": 0,
            "embedding_reuse_rate": 0,
        }

    old_vocab = set(old_tokenizer.get_vocab().keys())
    new_vocab = set(new_tokenizer.get_vocab().keys())

    # Vocab overlap
    overlap = old_vocab & new_vocab
    embedding_reuse_rate = len(overlap) / len(old_vocab) if old_vocab else 0

    # Sample boundary analysis
    boundary_changes = 0
    total_positions = 0

    for text in texts[:100]:  # Sample for efficiency
        old_tokens = old_tokenizer.encode(text, add_special_tokens=False)
        new_tokens = new_tokenizer.encode(text, add_special_tokens=False)

        # Count position differences
        total_positions += max(len(old_tokens), len(new_tokens))
        boundary_changes += abs(len(old_tokens) - len(new_tokens))

    return {
        "total_samples": len(texts),
        "vocab_overlap": len(overlap),
        "vocab_overlap_rate": len(overlap) / len(old_vocab) if old_vocab else 0,
        "new_tokens": len(new_vocab - old_vocab),
        "removed_tokens": len(old_vocab - new_vocab),
        "boundary_changes": boundary_changes,
        "boundary_change_rate": boundary_changes / total_positions if total_positions > 0 else 0,
        "embedding_reuse_rate": embedding_reuse_rate,
    }
