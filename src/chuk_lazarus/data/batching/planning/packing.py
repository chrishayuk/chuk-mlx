"""
Sequence packing for efficient training.

Packing concatenates multiple short sequences into longer ones,
reducing padding waste while maintaining correct semantics through
segment-aware attention masks.

Design principles:
- Packing in collator, not dataset: Pack at batch formation time
- Deterministic: Fixed ordering with deterministic tie-breaks
- Segment-aware: Block-diagonal masks prevent cross-sample attention
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    pass


class PackingMode(str, Enum):
    """
    Strategy for packing sequences into bins.

    FIRST_FIT: Add to first bin with enough space (fast, good enough)
    BEST_FIT: Add to bin with least remaining space (better packing, slower)
    GREEDY: Sort by length descending, pack largest first (balanced)
    """

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    GREEDY = "greedy"


class PackingConfig(BaseModel):
    """
    Configuration for sequence packing.

    Controls how sequences are packed and what metadata is preserved.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: PackingMode = Field(
        default=PackingMode.FIRST_FIT,
        description="Packing strategy",
    )
    max_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum packed sequence length",
    )
    pad_to_max: bool = Field(
        default=True,
        description="Pad all packed sequences to max_length",
    )
    add_separator: bool = Field(
        default=False,
        description="Add separator token between packed sequences",
    )
    separator_token_id: int | None = Field(
        default=None,
        description="Separator token ID (if add_separator=True)",
    )

    @classmethod
    def default(cls, max_length: int = 2048) -> PackingConfig:
        """Create default packing config."""
        return cls(max_length=max_length)


class PackedSequence(BaseModel):
    """
    A packed sequence containing multiple original samples.

    Maintains all information needed for:
    - Correct loss computation (loss_mask)
    - Segment-aware attention (segment_ids)
    - Provenance tracking (sample_ids)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Packed token data
    input_ids: tuple[int, ...] = Field(
        description="Concatenated token IDs",
    )
    loss_mask: tuple[int, ...] = Field(
        description="Concatenated loss masks",
    )
    segment_ids: tuple[int, ...] = Field(
        description="Segment ID for each token (0, 0, 0, 1, 1, 2, 2, ...)",
    )

    # Provenance
    sample_ids: tuple[str, ...] = Field(
        description="Original sample IDs in pack order",
    )
    sample_lengths: tuple[int, ...] = Field(
        description="Original sample lengths",
    )

    # Packing metadata
    num_segments: int = Field(
        ge=1,
        description="Number of segments (samples) in this pack",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total non-padding tokens",
    )
    padding_tokens: int = Field(
        ge=0,
        description="Number of padding tokens added",
    )

    @field_validator("input_ids", "loss_mask", "segment_ids", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert lists to tuples for immutability."""
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @field_validator("sample_ids", "sample_lengths", mode="before")
    @classmethod
    def convert_seq_to_tuple(cls, v):
        """Convert lists to tuples."""
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @property
    def length(self) -> int:
        """Total sequence length including padding."""
        return len(self.input_ids)

    @property
    def efficiency(self) -> float:
        """Fraction of non-padding tokens (1.0 = perfect)."""
        if self.length == 0:
            return 0.0
        return self.total_tokens / self.length

    @property
    def num_loss_tokens(self) -> int:
        """Number of tokens contributing to loss."""
        return sum(self.loss_mask)


class SequenceToPack(BaseModel):
    """
    Input sequence for packing.

    Lightweight wrapper for packing algorithm input.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    input_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]

    @field_validator("input_ids", "loss_mask", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @property
    def length(self) -> int:
        return len(self.input_ids)


# =============================================================================
# Packing Algorithms
# =============================================================================


class PackingBin:
    """
    A bin for accumulating sequences during packing.

    Mutable helper class used during packing process.
    """

    def __init__(self, max_length: int, pad_token_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.sequences: list[SequenceToPack] = []
        self.current_length = 0

    @property
    def remaining_space(self) -> int:
        """Space remaining in this bin."""
        return self.max_length - self.current_length

    def can_fit(self, seq: SequenceToPack, separator_len: int = 0) -> bool:
        """Check if sequence fits in this bin."""
        needed = seq.length + separator_len
        return self.current_length + needed <= self.max_length

    def add(self, seq: SequenceToPack, separator_len: int = 0) -> None:
        """Add sequence to this bin."""
        self.sequences.append(seq)
        self.current_length += seq.length + separator_len

    def to_packed_sequence(
        self,
        pad_to_max: bool = True,
        separator_token_id: int | None = None,
    ) -> PackedSequence:
        """Convert bin contents to a PackedSequence."""
        if not self.sequences:
            raise ValueError("Cannot create PackedSequence from empty bin")

        input_ids: list[int] = []
        loss_mask: list[int] = []
        segment_ids: list[int] = []
        sample_ids: list[str] = []
        sample_lengths: list[int] = []

        for segment_idx, seq in enumerate(self.sequences):
            # Add separator between sequences (not before first)
            if separator_token_id is not None and segment_idx > 0:
                input_ids.append(separator_token_id)
                loss_mask.append(0)  # Don't compute loss on separator
                segment_ids.append(segment_idx)

            # Add sequence
            input_ids.extend(seq.input_ids)
            loss_mask.extend(seq.loss_mask)
            segment_ids.extend([segment_idx] * seq.length)
            sample_ids.append(seq.sample_id)
            sample_lengths.append(seq.length)

        total_tokens = len(input_ids)
        padding_tokens = 0

        # Pad to max length if requested
        if pad_to_max and len(input_ids) < self.max_length:
            padding_tokens = self.max_length - len(input_ids)
            # Use last segment ID for padding (or 0 if empty)
            pad_segment_id = segment_ids[-1] if segment_ids else 0
            input_ids.extend([self.pad_token_id] * padding_tokens)
            loss_mask.extend([0] * padding_tokens)
            segment_ids.extend([pad_segment_id] * padding_tokens)

        return PackedSequence(
            input_ids=tuple(input_ids),
            loss_mask=tuple(loss_mask),
            segment_ids=tuple(segment_ids),
            sample_ids=tuple(sample_ids),
            sample_lengths=tuple(sample_lengths),
            num_segments=len(self.sequences),
            total_tokens=total_tokens,
            padding_tokens=padding_tokens,
        )


def pack_sequences_first_fit(
    sequences: list[SequenceToPack],
    config: PackingConfig,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """
    Pack sequences using first-fit algorithm.

    Adds each sequence to the first bin that has space.
    Fast but may not achieve optimal packing.
    """
    separator_len = 1 if config.add_separator else 0
    bins: list[PackingBin] = []

    for seq in sequences:
        if seq.length > config.max_length:
            # Sequence too long, skip or truncate
            continue

        # Find first bin with space
        placed = False
        for bin in bins:
            if bin.can_fit(seq, separator_len if bin.sequences else 0):
                bin.add(seq, separator_len if bin.sequences else 0)
                placed = True
                break

        if not placed:
            # Create new bin
            new_bin = PackingBin(config.max_length, pad_token_id)
            new_bin.add(seq, 0)  # No separator for first sequence
            bins.append(new_bin)

    # Convert bins to PackedSequences
    return [
        bin.to_packed_sequence(
            pad_to_max=config.pad_to_max,
            separator_token_id=config.separator_token_id if config.add_separator else None,
        )
        for bin in bins
        if bin.sequences
    ]


def pack_sequences_best_fit(
    sequences: list[SequenceToPack],
    config: PackingConfig,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """
    Pack sequences using best-fit algorithm.

    Adds each sequence to the bin with least remaining space.
    Better packing but slower than first-fit.
    """
    separator_len = 1 if config.add_separator else 0
    bins: list[PackingBin] = []

    for seq in sequences:
        if seq.length > config.max_length:
            continue

        # Find best-fit bin (least remaining space that still fits)
        best_bin = None
        best_remaining = float("inf")

        for bin in bins:
            sep = separator_len if bin.sequences else 0
            if bin.can_fit(seq, sep):
                remaining = bin.remaining_space - seq.length - sep
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = bin

        if best_bin is not None:
            sep = separator_len if best_bin.sequences else 0
            best_bin.add(seq, sep)
        else:
            new_bin = PackingBin(config.max_length, pad_token_id)
            new_bin.add(seq, 0)
            bins.append(new_bin)

    return [
        bin.to_packed_sequence(
            pad_to_max=config.pad_to_max,
            separator_token_id=config.separator_token_id if config.add_separator else None,
        )
        for bin in bins
        if bin.sequences
    ]


def pack_sequences_greedy(
    sequences: list[SequenceToPack],
    config: PackingConfig,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """
    Pack sequences using greedy algorithm.

    Sorts sequences by length (descending) then uses first-fit.
    Often achieves better packing than pure first-fit.
    """
    # Sort by length descending
    sorted_seqs = sorted(sequences, key=lambda s: s.length, reverse=True)
    return pack_sequences_first_fit(sorted_seqs, config, pad_token_id)


def pack_sequences(
    sequences: list[SequenceToPack],
    config: PackingConfig,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """
    Pack sequences using the configured algorithm.

    Args:
        sequences: List of sequences to pack
        config: Packing configuration
        pad_token_id: Token ID to use for padding

    Returns:
        List of packed sequences
    """
    if config.mode == PackingMode.FIRST_FIT:
        return pack_sequences_first_fit(sequences, config, pad_token_id)
    elif config.mode == PackingMode.BEST_FIT:
        return pack_sequences_best_fit(sequences, config, pad_token_id)
    elif config.mode == PackingMode.GREEDY:
        return pack_sequences_greedy(sequences, config, pad_token_id)
    else:
        raise ValueError(f"Unknown packing mode: {config.mode}")


# =============================================================================
# Attention Mask Generation
# =============================================================================


def create_segment_attention_mask_numpy(segment_ids: list[int] | tuple[int, ...]):
    """
    Create block-diagonal attention mask from segment IDs.

    Each position can only attend to positions with the same segment ID.
    This prevents information leakage between packed sequences.

    Args:
        segment_ids: Segment ID for each position (e.g., [0, 0, 0, 1, 1, 2, 2])

    Returns:
        NumPy array of shape (seq_len, seq_len) with 1s for allowed attention
    """
    import numpy as np

    segment_ids = np.array(segment_ids)
    seq_len = len(segment_ids)

    # Create mask: position i can attend to position j if segment_ids[i] == segment_ids[j]
    # AND j <= i (causal)
    segment_mask = segment_ids[:, None] == segment_ids[None, :]
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))

    return (segment_mask & causal_mask).astype(np.float32)


def create_segment_attention_mask_mlx(segment_ids: list[int] | tuple[int, ...]):
    """
    Create block-diagonal attention mask from segment IDs using MLX.

    Each position can only attend to positions with the same segment ID.
    This prevents information leakage between packed sequences.

    Args:
        segment_ids: Segment ID for each position

    Returns:
        MLX array of shape (seq_len, seq_len) with 1s for allowed attention
    """
    import mlx.core as mx

    segment_ids = mx.array(segment_ids)
    seq_len = len(segment_ids)

    # Create mask: position i can attend to position j if segment_ids[i] == segment_ids[j]
    segment_mask = segment_ids[:, None] == segment_ids[None, :]

    # Causal mask: position i can only attend to positions j <= i
    causal_mask = mx.tril(mx.ones((seq_len, seq_len)))

    return (segment_mask & causal_mask.astype(mx.bool_)).astype(mx.float32)


def create_segment_attention_mask(
    segment_ids: list[int] | tuple[int, ...],
    use_mlx: bool = True,
):
    """
    Create block-diagonal attention mask from segment IDs.

    Automatically selects MLX or NumPy backend.

    Args:
        segment_ids: Segment ID for each position
        use_mlx: If True, use MLX backend (default); else use NumPy

    Returns:
        Array of shape (seq_len, seq_len) with 1s for allowed attention
    """
    if use_mlx:
        try:
            return create_segment_attention_mask_mlx(segment_ids)
        except ImportError:
            pass
    return create_segment_attention_mask_numpy(segment_ids)


# =============================================================================
# Metrics
# =============================================================================


class PackingMetrics(BaseModel):
    """
    Metrics for evaluating packing efficiency.
    """

    model_config = ConfigDict(frozen=False)

    # Counts
    num_original_samples: int = Field(default=0, ge=0)
    num_packed_sequences: int = Field(default=0, ge=0)
    num_skipped: int = Field(default=0, ge=0, description="Samples too long to pack")

    # Token counts
    total_tokens: int = Field(default=0, ge=0)
    total_padded_length: int = Field(default=0, ge=0)
    total_loss_tokens: int = Field(default=0, ge=0)

    @property
    def packing_ratio(self) -> float:
        """Average samples per packed sequence."""
        if self.num_packed_sequences == 0:
            return 0.0
        return self.num_original_samples / self.num_packed_sequences

    @property
    def efficiency(self) -> float:
        """Fraction of non-padding tokens."""
        if self.total_padded_length == 0:
            return 0.0
        return self.total_tokens / self.total_padded_length

    @property
    def loss_efficiency(self) -> float:
        """Fraction of tokens contributing to loss."""
        if self.total_padded_length == 0:
            return 0.0
        return self.total_loss_tokens / self.total_padded_length

    def record_packed_sequence(self, packed: PackedSequence) -> None:
        """Record a packed sequence."""
        self.num_original_samples += packed.num_segments
        self.num_packed_sequences += 1
        self.total_tokens += packed.total_tokens
        self.total_padded_length += packed.length
        self.total_loss_tokens += packed.num_loss_tokens

    def record_skip(self) -> None:
        """Record a skipped sample."""
        self.num_skipped += 1

    def summary(self) -> dict:
        """Get summary as dict."""
        return {
            "num_original_samples": self.num_original_samples,
            "num_packed_sequences": self.num_packed_sequences,
            "num_skipped": self.num_skipped,
            "packing_ratio": f"{self.packing_ratio:.2f}",
            "efficiency": f"{self.efficiency:.2%}",
            "loss_efficiency": f"{self.loss_efficiency:.2%}",
        }


def compute_packing_metrics(
    packed_sequences: list[PackedSequence],
    num_skipped: int = 0,
) -> PackingMetrics:
    """
    Compute metrics for a list of packed sequences.

    Args:
        packed_sequences: List of packed sequences
        num_skipped: Number of samples that were too long to pack

    Returns:
        PackingMetrics with efficiency statistics
    """
    metrics = PackingMetrics()
    metrics.num_skipped = num_skipped

    for packed in packed_sequences:
        metrics.record_packed_sequence(packed)

    return metrics
