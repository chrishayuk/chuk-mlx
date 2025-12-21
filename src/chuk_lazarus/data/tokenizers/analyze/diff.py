"""Retokenization comparison and diff tools."""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class TokenBoundaryShift(BaseModel):
    """A shift in token boundary between tokenizers."""

    position: int = Field(ge=0, description="Character position in text")
    tokenizer1_boundary: bool = Field(description="Is boundary in tokenizer 1")
    tokenizer2_boundary: bool = Field(description="Is boundary in tokenizer 2")
    context: str = Field(description="Text around the boundary")


class RetokenizationDiff(BaseModel):
    """Detailed comparison of tokenization between two tokenizers."""

    text: str = Field(description="Original text")
    tokenizer1_ids: list[int] = Field(description="Token IDs from tokenizer 1")
    tokenizer2_ids: list[int] = Field(description="Token IDs from tokenizer 2")
    tokenizer1_tokens: list[str] = Field(description="Decoded tokens from tokenizer 1")
    tokenizer2_tokens: list[str] = Field(description="Decoded tokens from tokenizer 2")
    length_delta: int = Field(description="Difference in token count (t1 - t2)")
    length_ratio: float = Field(gt=0.0, description="Ratio of lengths (t1 / t2)")
    boundary_shifts: list[TokenBoundaryShift] = Field(
        default_factory=list, description="Token boundary differences"
    )
    common_token_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of tokens in common")


class CorpusDiff(BaseModel):
    """Aggregate diff statistics across a corpus."""

    total_texts: int = Field(ge=0, description="Number of texts compared")
    avg_length_delta: float = Field(description="Average token count difference")
    avg_length_ratio: float = Field(description="Average length ratio")
    tokenizer1_total_tokens: int = Field(ge=0, description="Total tokens from t1")
    tokenizer2_total_tokens: int = Field(ge=0, description="Total tokens from t2")
    compression_improvement: float = Field(
        description="% fewer tokens in t2 vs t1 (positive = t2 better)"
    )
    texts_with_different_lengths: int = Field(ge=0, description="Texts where tokenizers differ")
    worst_cases: list[RetokenizationDiff] = Field(
        default_factory=list, description="Texts with largest differences"
    )


def _get_token_boundaries(text: str, token_ids: list[int], tokenizer) -> list[int]:
    """
    Get character positions where token boundaries occur.

    This is approximate since tokenization isn't always reversible.
    """
    boundaries = [0]
    current_pos = 0

    for tid in token_ids:
        try:
            token_str = tokenizer.decode([tid])
            # Find this token in remaining text
            # This is approximate - subword tokenizers can be tricky
            token_len = len(token_str.strip())
            current_pos += max(1, token_len)
            if current_pos <= len(text):
                boundaries.append(current_pos)
        except Exception:
            current_pos += 1
            boundaries.append(current_pos)

    return boundaries


def compare_tokenizations_detailed(
    text: str,
    tokenizer1: TokenizerProtocol,
    tokenizer2: TokenizerProtocol,
    context_chars: int = 10,
) -> RetokenizationDiff:
    """
    Compare how two tokenizers tokenize the same text.

    Args:
        text: Text to compare
        tokenizer1: First tokenizer
        tokenizer2: Second tokenizer
        context_chars: Characters of context around boundary shifts

    Returns:
        RetokenizationDiff with detailed comparison
    """
    ids1 = tokenizer1.encode(text, add_special_tokens=False)
    ids2 = tokenizer2.encode(text, add_special_tokens=False)

    tokens1 = [tokenizer1.decode([tid]) for tid in ids1]
    tokens2 = [tokenizer2.decode([tid]) for tid in ids2]

    len1 = len(ids1)
    len2 = len(ids2)

    length_delta = len1 - len2
    length_ratio = len1 / len2 if len2 > 0 else float("inf")

    # Find common tokens (by decoded string, position-independent)
    set1 = set(tokens1)
    set2 = set(tokens2)
    common = set1 & set2
    total_unique = len(set1 | set2)
    common_ratio = len(common) / total_unique if total_unique > 0 else 1.0

    # Find boundary shifts (approximate)
    boundaries1 = set(_get_token_boundaries(text, ids1, tokenizer1))
    boundaries2 = set(_get_token_boundaries(text, ids2, tokenizer2))

    shifts = []
    all_boundaries = boundaries1 | boundaries2
    for pos in sorted(all_boundaries):
        in1 = pos in boundaries1
        in2 = pos in boundaries2
        if in1 != in2:
            # Get context
            start = max(0, pos - context_chars)
            end = min(len(text), pos + context_chars)
            context = text[start:end]
            shifts.append(
                TokenBoundaryShift(
                    position=pos,
                    tokenizer1_boundary=in1,
                    tokenizer2_boundary=in2,
                    context=context,
                )
            )

    return RetokenizationDiff(
        text=text,
        tokenizer1_ids=ids1,
        tokenizer2_ids=ids2,
        tokenizer1_tokens=tokens1,
        tokenizer2_tokens=tokens2,
        length_delta=length_delta,
        length_ratio=length_ratio,
        boundary_shifts=shifts[:20],  # Limit to avoid huge outputs
        common_token_ratio=common_ratio,
    )


def diff_corpus(
    texts: list[str],
    tokenizer1: TokenizerProtocol,
    tokenizer2: TokenizerProtocol,
    worst_n: int = 5,
) -> CorpusDiff:
    """
    Compare two tokenizers across an entire corpus.

    Args:
        texts: List of texts to compare
        tokenizer1: First tokenizer (baseline)
        tokenizer2: Second tokenizer (comparison)
        worst_n: Number of worst-case examples to include

    Returns:
        CorpusDiff with aggregate statistics
    """
    diffs: list[RetokenizationDiff] = []
    total_len1 = 0
    total_len2 = 0
    different_count = 0

    for text in texts:
        diff = compare_tokenizations_detailed(text, tokenizer1, tokenizer2)
        diffs.append(diff)
        total_len1 += len(diff.tokenizer1_ids)
        total_len2 += len(diff.tokenizer2_ids)
        if diff.length_delta != 0:
            different_count += 1

    # Calculate aggregates
    if diffs:
        avg_delta = sum(d.length_delta for d in diffs) / len(diffs)
        avg_ratio = sum(d.length_ratio for d in diffs) / len(diffs)
    else:
        avg_delta = 0.0
        avg_ratio = 1.0

    # Compression improvement: positive means t2 uses fewer tokens
    compression = (total_len1 - total_len2) / total_len1 * 100 if total_len1 > 0 else 0.0

    # Get worst cases (largest absolute length difference)
    worst_cases = sorted(diffs, key=lambda d: abs(d.length_delta), reverse=True)[:worst_n]

    return CorpusDiff(
        total_texts=len(texts),
        avg_length_delta=avg_delta,
        avg_length_ratio=avg_ratio,
        tokenizer1_total_tokens=total_len1,
        tokenizer2_total_tokens=total_len2,
        compression_improvement=compression,
        texts_with_different_lengths=different_count,
        worst_cases=worst_cases,
    )
