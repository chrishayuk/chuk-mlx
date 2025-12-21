"""Vocabulary management utilities with Pydantic models."""

from collections import Counter
from enum import Enum

from pydantic import BaseModel, Field


class ConflictResolution(str, Enum):
    """How to handle ID conflicts when merging vocabularies."""

    FIRST = "first"
    SECOND = "second"
    RENUMBER = "renumber"


class SortOrder(str, Enum):
    """Sort order for vocabulary operations."""

    BY_ID = "id"
    ALPHABETICAL = "alpha"


class VocabularyStats(BaseModel):
    """Statistics about a vocabulary."""

    size: int = Field(ge=0, description="Number of tokens in vocabulary")
    min_id: int = Field(description="Minimum token ID")
    max_id: int = Field(description="Maximum token ID")
    id_range: int = Field(ge=0, description="Range of IDs (max - min + 1)")
    avg_token_length: float = Field(ge=0.0, description="Average token string length")
    max_token_length: int = Field(ge=0, description="Maximum token string length")
    min_token_length: int = Field(ge=0, description="Minimum token string length")


class VocabularyIssues(BaseModel):
    """Validation issues found in a vocabulary."""

    duplicate_ids: list[dict] = Field(
        default_factory=list, description="IDs assigned to multiple tokens"
    )
    missing_ids: list[int] = Field(default_factory=list, description="Gaps in ID sequence")
    negative_ids: list[dict] = Field(default_factory=list, description="Negative token IDs")

    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return bool(self.duplicate_ids or self.missing_ids or self.negative_ids)


class VocabularyDiff(BaseModel):
    """Difference between two vocabularies."""

    only_in_first: dict[str, int] = Field(
        default_factory=dict, description="Tokens only in first vocabulary"
    )
    only_in_second: dict[str, int] = Field(
        default_factory=dict, description="Tokens only in second vocabulary"
    )
    in_both_count: int = Field(ge=0, description="Number of tokens in both vocabularies")


def merge_vocabularies(
    vocab1: dict[str, int],
    vocab2: dict[str, int],
    conflict_resolution: ConflictResolution = ConflictResolution.FIRST,
) -> dict[str, int]:
    """
    Merge two vocabularies into one.

    Args:
        vocab1: First vocabulary (token -> id)
        vocab2: Second vocabulary (token -> id)
        conflict_resolution: How to handle ID conflicts

    Returns:
        Merged vocabulary
    """
    if conflict_resolution == ConflictResolution.FIRST:
        merged = vocab1.copy()
        max_id = max(vocab1.values()) if vocab1 else -1
        for token in vocab2:
            if token not in merged:
                max_id += 1
                merged[token] = max_id
        return merged

    elif conflict_resolution == ConflictResolution.SECOND:
        merged = vocab2.copy()
        max_id = max(vocab2.values()) if vocab2 else -1
        for token in vocab1:
            if token not in merged:
                max_id += 1
                merged[token] = max_id
        return merged

    elif conflict_resolution == ConflictResolution.RENUMBER:
        merged = vocab1.copy()
        max_id = max(vocab1.values()) if vocab1 else -1
        for token in vocab2:
            if token not in merged:
                max_id += 1
                merged[token] = max_id
        return merged

    else:
        raise ValueError(f"Unknown conflict_resolution: {conflict_resolution}")


def filter_vocabulary(
    vocab: dict[str, int],
    token_counts: Counter[str] | dict[str, int],
    min_freq: int = 1,
    max_freq: int | None = None,
    keep_special: set[str] | None = None,
) -> dict[str, int]:
    """
    Filter vocabulary based on token frequencies.

    Args:
        vocab: Vocabulary (token -> id)
        token_counts: Token frequency counts
        min_freq: Minimum frequency to keep
        max_freq: Maximum frequency to keep (None = no limit)
        keep_special: Set of tokens to always keep

    Returns:
        Filtered vocabulary with renumbered IDs
    """
    keep_special = keep_special or set()

    filtered_tokens = []
    for token in vocab:
        if token in keep_special:
            filtered_tokens.append(token)
        elif token in token_counts:
            freq = token_counts[token]
            if freq >= min_freq and (max_freq is None or freq <= max_freq):
                filtered_tokens.append(token)

    return {token: i for i, token in enumerate(filtered_tokens)}


def extend_vocabulary(
    vocab: dict[str, int],
    new_tokens: list[str],
    start_id: int | None = None,
) -> dict[str, int]:
    """
    Add new tokens to vocabulary with proper IDs.

    Args:
        vocab: Existing vocabulary
        new_tokens: List of new tokens to add
        start_id: Starting ID for new tokens (uses max+1 if None)

    Returns:
        Extended vocabulary
    """
    extended = vocab.copy()

    if start_id is None:
        start_id = max(vocab.values()) + 1 if vocab else 0

    current_id = start_id
    for token in new_tokens:
        if token not in extended:
            extended[token] = current_id
            current_id += 1

    return extended


def shrink_vocabulary(
    vocab: dict[str, int],
    tokens_to_remove: set[str],
    renumber: bool = True,
) -> dict[str, int]:
    """
    Remove tokens from vocabulary.

    Args:
        vocab: Vocabulary
        tokens_to_remove: Tokens to remove
        renumber: Whether to renumber remaining tokens

    Returns:
        Shrunk vocabulary
    """
    filtered = {t: i for t, i in vocab.items() if t not in tokens_to_remove}

    if renumber:
        return {t: i for i, t in enumerate(sorted(filtered.keys(), key=lambda x: filtered[x]))}

    return filtered


def get_vocabulary_diff(
    vocab1: dict[str, int],
    vocab2: dict[str, int],
) -> VocabularyDiff:
    """
    Compare two vocabularies and find differences.

    Args:
        vocab1: First vocabulary
        vocab2: Second vocabulary

    Returns:
        VocabularyDiff with comparison results
    """
    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())

    only_first = tokens1 - tokens2
    only_second = tokens2 - tokens1
    in_both = tokens1 & tokens2

    return VocabularyDiff(
        only_in_first={t: vocab1[t] for t in only_first},
        only_in_second={t: vocab2[t] for t in only_second},
        in_both_count=len(in_both),
    )


def renumber_vocabulary(
    vocab: dict[str, int],
    start_id: int = 0,
    sort_by: SortOrder = SortOrder.BY_ID,
) -> dict[str, int]:
    """
    Renumber vocabulary IDs to be contiguous.

    Args:
        vocab: Vocabulary with potentially non-contiguous IDs
        start_id: Starting ID
        sort_by: BY_ID to maintain order, ALPHABETICAL for alphabetical

    Returns:
        Vocabulary with contiguous IDs starting at start_id
    """
    if sort_by == SortOrder.BY_ID:
        sorted_tokens = sorted(vocab.keys(), key=lambda x: vocab[x])
    elif sort_by == SortOrder.ALPHABETICAL:
        sorted_tokens = sorted(vocab.keys())
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    return {token: start_id + i for i, token in enumerate(sorted_tokens)}


def create_id_to_token(vocab: dict[str, int]) -> dict[int, str]:
    """
    Create reverse mapping from IDs to tokens.

    Args:
        vocab: Token to ID mapping

    Returns:
        ID to token mapping
    """
    return {i: t for t, i in vocab.items()}


def validate_vocabulary(vocab: dict[str, int]) -> VocabularyIssues:
    """
    Validate vocabulary for common issues.

    Args:
        vocab: Vocabulary to validate

    Returns:
        VocabularyIssues with any problems found
    """
    issues = VocabularyIssues()

    # Check for duplicate IDs
    id_to_tokens: dict[int, list[str]] = {}
    for token, token_id in vocab.items():
        if token_id not in id_to_tokens:
            id_to_tokens[token_id] = []
        id_to_tokens[token_id].append(token)

    for token_id, tokens in id_to_tokens.items():
        if len(tokens) > 1:
            issues.duplicate_ids.append({"id": token_id, "tokens": tokens})

    # Check for negative IDs
    for token, token_id in vocab.items():
        if token_id < 0:
            issues.negative_ids.append({"token": token, "id": token_id})

    # Check for gaps in ID sequence
    if vocab:
        all_ids = set(vocab.values())
        min_id = min(all_ids)
        max_id = max(all_ids)
        expected = set(range(min_id, max_id + 1))
        missing = expected - all_ids
        if missing:
            issues.missing_ids = sorted(missing)

    return issues


def get_vocabulary_stats(vocab: dict[str, int]) -> VocabularyStats:
    """
    Get statistics about a vocabulary.

    Args:
        vocab: Vocabulary

    Returns:
        VocabularyStats with size and length information
    """
    if not vocab:
        return VocabularyStats(
            size=0,
            min_id=0,
            max_id=0,
            id_range=0,
            avg_token_length=0.0,
            max_token_length=0,
            min_token_length=0,
        )

    ids = list(vocab.values())
    token_lengths = [len(t) for t in vocab.keys()]

    return VocabularyStats(
        size=len(vocab),
        min_id=min(ids),
        max_id=max(ids),
        id_range=max(ids) - min(ids) + 1,
        avg_token_length=sum(token_lengths) / len(token_lengths),
        max_token_length=max(token_lengths),
        min_token_length=min(token_lengths),
    )
