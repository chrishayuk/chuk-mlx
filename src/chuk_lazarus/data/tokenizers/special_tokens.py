"""Special token handling utilities with Pydantic models."""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class SpecialTokenType(str, Enum):
    """Types of special tokens."""

    PAD = "pad"
    UNK = "unk"
    BOS = "bos"
    EOS = "eos"
    SEP = "sep"
    CLS = "cls"
    MASK = "mask"


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    pad_token_id: int | None
    unk_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None


class SpecialTokenCount(BaseModel):
    """Count of special tokens in a sequence."""

    pad: int = Field(default=0, ge=0, description="Number of PAD tokens")
    unk: int = Field(default=0, ge=0, description="Number of UNK tokens")
    bos: int = Field(default=0, ge=0, description="Number of BOS tokens")
    eos: int = Field(default=0, ge=0, description="Number of EOS tokens")
    other_special: int = Field(default=0, ge=0, description="Other special tokens")
    total: int = Field(default=0, ge=0, description="Total special tokens")


class SpecialTokenConfig(BaseModel):
    """Configuration for special token handling."""

    pad_token_id: int | None = Field(default=None, description="PAD token ID")
    unk_token_id: int | None = Field(default=None, description="UNK token ID")
    bos_token_id: int | None = Field(default=None, description="BOS token ID")
    eos_token_id: int | None = Field(default=None, description="EOS token ID")
    additional_special_ids: set[int] = Field(
        default_factory=set, description="Additional special token IDs"
    )

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerProtocol) -> "SpecialTokenConfig":
        """Create config from a tokenizer instance."""
        additional = set()
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            ids = getattr(tokenizer, "additional_special_tokens_ids", [])
            if ids:
                additional.update(ids)
        if hasattr(tokenizer, "special_tokens"):
            special = getattr(tokenizer, "special_tokens", {})
            if isinstance(special, dict):
                additional.update(special.values())

        return cls(
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            unk_token_id=getattr(tokenizer, "unk_token_id", None),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            additional_special_ids=additional,
        )

    def all_special_ids(self) -> set[int]:
        """Get all special token IDs."""
        ids = set()
        for attr in ["pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"]:
            val = getattr(self, attr)
            if val is not None:
                ids.add(val)
        ids.update(self.additional_special_ids)
        return ids


def get_special_token_ids(tokenizer: TokenizerProtocol) -> set[int]:
    """
    Get all special token IDs from a tokenizer.

    Args:
        tokenizer: Tokenizer instance with special token attributes

    Returns:
        Set of special token IDs
    """
    config = SpecialTokenConfig.from_tokenizer(tokenizer)
    return config.all_special_ids()


def get_special_token_mask(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> list[bool]:
    """
    Create a boolean mask indicating which tokens are special tokens.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        List of booleans (True for special tokens)
    """
    special_ids = get_special_token_ids(tokenizer)
    return [tid in special_ids for tid in token_ids]


def strip_special_tokens(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> list[int]:
    """
    Remove all special tokens from a sequence.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        Token IDs with special tokens removed
    """
    special_ids = get_special_token_ids(tokenizer)
    return [tid for tid in token_ids if tid not in special_ids]


def strip_padding(
    token_ids: list[int],
    pad_token_id: int,
    from_left: bool = False,
) -> list[int]:
    """
    Remove padding tokens from a sequence.

    Args:
        token_ids: List of token IDs
        pad_token_id: Padding token ID
        from_left: If True, remove from left; else from right

    Returns:
        Token IDs with padding removed
    """
    if not token_ids:
        return []

    if from_left:
        start_idx = 0
        while start_idx < len(token_ids) and token_ids[start_idx] == pad_token_id:
            start_idx += 1
        return token_ids[start_idx:]
    else:
        end_idx = len(token_ids)
        while end_idx > 0 and token_ids[end_idx - 1] == pad_token_id:
            end_idx -= 1
        return token_ids[:end_idx]


def add_bos_token(
    token_ids: list[int],
    bos_token_id: int,
) -> list[int]:
    """
    Add BOS (beginning of sequence) token if not present.

    Args:
        token_ids: List of token IDs
        bos_token_id: BOS token ID

    Returns:
        Token IDs with BOS token at start
    """
    if not token_ids or token_ids[0] != bos_token_id:
        return [bos_token_id] + token_ids
    return token_ids


def add_eos_token(
    token_ids: list[int],
    eos_token_id: int,
) -> list[int]:
    """
    Add EOS (end of sequence) token if not present.

    Args:
        token_ids: List of token IDs
        eos_token_id: EOS token ID

    Returns:
        Token IDs with EOS token at end
    """
    if not token_ids or token_ids[-1] != eos_token_id:
        return token_ids + [eos_token_id]
    return token_ids


def add_special_tokens(
    token_ids: list[int],
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """
    Add BOS and/or EOS tokens to a sequence.

    Args:
        token_ids: List of token IDs
        bos_token_id: BOS token ID (skipped if None)
        eos_token_id: EOS token ID (skipped if None)

    Returns:
        Token IDs with special tokens added
    """
    result = token_ids.copy()

    if bos_token_id is not None:
        result = add_bos_token(result, bos_token_id)

    if eos_token_id is not None:
        result = add_eos_token(result, eos_token_id)

    return result


def ensure_special_tokens(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> list[int]:
    """
    Ensure a sequence has proper BOS and EOS tokens.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        Token IDs with BOS at start and EOS at end
    """
    return add_special_tokens(
        token_ids,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def find_eos_positions(
    token_ids: list[int],
    eos_token_id: int,
) -> list[int]:
    """
    Find all positions of EOS tokens in a sequence.

    Args:
        token_ids: List of token IDs
        eos_token_id: EOS token ID

    Returns:
        List of indices where EOS tokens appear
    """
    return [i for i, tid in enumerate(token_ids) if tid == eos_token_id]


def split_on_eos(
    token_ids: list[int],
    eos_token_id: int,
    keep_eos: bool = True,
) -> list[list[int]]:
    """
    Split a sequence on EOS tokens.

    Args:
        token_ids: List of token IDs
        eos_token_id: EOS token ID
        keep_eos: Whether to keep EOS at end of each split

    Returns:
        List of token sequences
    """
    if not token_ids:
        return []

    sequences = []
    current: list[int] = []

    for tid in token_ids:
        if tid == eos_token_id:
            if keep_eos:
                current.append(tid)
            if current:
                sequences.append(current)
            current = []
        else:
            current.append(tid)

    if current:
        sequences.append(current)

    return sequences


def count_special_tokens(
    token_ids: list[int],
    tokenizer: TokenizerProtocol,
) -> SpecialTokenCount:
    """
    Count occurrences of each special token type.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance

    Returns:
        SpecialTokenCount with counts for each type
    """
    config = SpecialTokenConfig.from_tokenizer(tokenizer)
    special_ids = config.all_special_ids()

    counts = {
        SpecialTokenType.PAD: 0,
        SpecialTokenType.UNK: 0,
        SpecialTokenType.BOS: 0,
        SpecialTokenType.EOS: 0,
    }

    known_special = {
        config.pad_token_id: SpecialTokenType.PAD,
        config.unk_token_id: SpecialTokenType.UNK,
        config.bos_token_id: SpecialTokenType.BOS,
        config.eos_token_id: SpecialTokenType.EOS,
    }

    other_count = 0
    for tid in token_ids:
        if tid in known_special and known_special[tid] is not None:
            token_type = known_special[tid]
            counts[token_type] += 1
        elif tid in special_ids:
            other_count += 1

    total = sum(counts.values()) + other_count

    return SpecialTokenCount(
        pad=counts[SpecialTokenType.PAD],
        unk=counts[SpecialTokenType.UNK],
        bos=counts[SpecialTokenType.BOS],
        eos=counts[SpecialTokenType.EOS],
        other_special=other_count,
        total=total,
    )
