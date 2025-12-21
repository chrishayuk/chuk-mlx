"""Smart sequence packing for efficient batching."""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    pad_token_id: int
    eos_token_id: int | None


class PackingConfig(BaseModel):
    """Configuration for sequence packing."""

    max_seq_length: int = Field(ge=1, description="Maximum sequence length")
    pad_token_id: int = Field(ge=0, description="Padding token ID")
    eos_token_id: int | None = Field(default=None, description="EOS token ID")
    add_eos_between: bool = Field(default=True, description="Add EOS between packed sequences")
    respect_document_boundaries: bool = Field(
        default=True, description="Don't pack across document boundaries"
    )
    min_sequence_length: int = Field(default=1, ge=1, description="Minimum length to include")


class PackedSequence(BaseModel):
    """A single packed sequence."""

    token_ids: list[int] = Field(description="Packed token IDs")
    attention_mask: list[int] = Field(description="Attention mask")
    loss_mask: list[int] = Field(
        description="Which positions to compute loss on (1=compute, 0=ignore)"
    )
    source_indices: list[int] = Field(description="Original sequence indices that were packed")
    num_real_tokens: int = Field(ge=0, description="Non-padding tokens")
    num_padding_tokens: int = Field(ge=0, description="Padding tokens")


class PackedBatch(BaseModel):
    """A batch of packed sequences."""

    sequences: list[PackedSequence] = Field(description="Packed sequences")
    total_sequences: int = Field(ge=0, description="Total packed sequences")
    total_source_sequences: int = Field(ge=0, description="Original sequences packed")
    packing_ratio: float = Field(ge=1.0, description="Sequences per packed sequence")


class PackingStats(BaseModel):
    """Statistics about packing efficiency."""

    total_tokens: int = Field(ge=0, description="Total tokens in packed batch")
    real_tokens: int = Field(ge=0, description="Non-padding tokens")
    padding_tokens: int = Field(ge=0, description="Padding tokens")
    padding_ratio: float = Field(ge=0.0, le=1.0, description="Fraction of padding")
    efficiency: float = Field(ge=0.0, le=1.0, description="Token utilization")
    avg_sequences_per_pack: float = Field(ge=0.0, description="Avg sequences per pack")
    throughput_improvement: float = Field(
        ge=1.0, description="Estimated throughput improvement vs naive padding"
    )


def _pack_greedy(
    sequences: list[list[int]],
    config: PackingConfig,
) -> list[list[tuple[int, list[int]]]]:
    """
    Greedy packing algorithm.

    Packs sequences into bins of max_seq_length using first-fit decreasing.

    Returns list of bins, where each bin is a list of (original_idx, token_ids).
    """
    # Sort by length descending for better packing
    indexed = [(i, seq) for i, seq in enumerate(sequences)]
    indexed.sort(key=lambda x: len(x[1]), reverse=True)

    bins: list[list[tuple[int, list[int]]]] = []
    bin_sizes: list[int] = []

    eos_overhead = 1 if config.add_eos_between and config.eos_token_id is not None else 0

    for idx, seq in indexed:
        if len(seq) < config.min_sequence_length:
            continue
        if len(seq) > config.max_seq_length:
            # Truncate if too long
            seq = seq[: config.max_seq_length]

        # Find first bin that fits
        placed = False
        seq_with_overhead = len(seq) + eos_overhead

        for i, bin_size in enumerate(bin_sizes):
            if bin_size + seq_with_overhead <= config.max_seq_length:
                bins[i].append((idx, seq))
                bin_sizes[i] += seq_with_overhead
                placed = True
                break

        if not placed:
            # Create new bin
            bins.append([(idx, seq)])
            bin_sizes.append(len(seq))

    return bins


def pack_sequences(
    token_sequences: list[list[int]],
    config: PackingConfig,
) -> list[PackedSequence]:
    """
    Pack multiple sequences into efficient batches.

    This packs multiple shorter sequences into single max_seq_length
    sequences to reduce padding waste.

    Args:
        token_sequences: List of tokenized sequences
        config: Packing configuration

    Returns:
        List of PackedSequence
    """
    bins = _pack_greedy(token_sequences, config)
    packed = []

    for bin_contents in bins:
        token_ids = []
        loss_mask = []
        source_indices = []

        for idx, seq in bin_contents:
            source_indices.append(idx)

            # Add sequence tokens
            token_ids.extend(seq)
            loss_mask.extend([1] * len(seq))

            # Add EOS between sequences
            if config.add_eos_between and config.eos_token_id is not None:
                token_ids.append(config.eos_token_id)
                loss_mask.append(0)  # Don't compute loss on separator

        # Remove trailing EOS if present
        if config.add_eos_between and token_ids and token_ids[-1] == config.eos_token_id:
            token_ids = token_ids[:-1]
            loss_mask = loss_mask[:-1]

        # Pad to max length
        num_real = len(token_ids)
        num_pad = config.max_seq_length - num_real

        if num_pad > 0:
            token_ids.extend([config.pad_token_id] * num_pad)
            loss_mask.extend([0] * num_pad)

        attention_mask = [1] * num_real + [0] * num_pad

        packed.append(
            PackedSequence(
                token_ids=token_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                source_indices=source_indices,
                num_real_tokens=num_real,
                num_padding_tokens=num_pad,
            )
        )

    return packed


def create_packed_batch(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    max_seq_length: int,
    add_special_tokens: bool = False,
) -> PackedBatch:
    """
    Create a packed batch from texts.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens during encoding

    Returns:
        PackedBatch with efficiently packed sequences
    """
    # Encode all texts
    sequences = [tokenizer.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    config = PackingConfig(
        max_seq_length=max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    packed_seqs = pack_sequences(sequences, config)

    packing_ratio = len(texts) / len(packed_seqs) if packed_seqs else 1.0

    return PackedBatch(
        sequences=packed_seqs,
        total_sequences=len(packed_seqs),
        total_source_sequences=len(texts),
        packing_ratio=packing_ratio,
    )


def calculate_packing_efficiency(batch: PackedBatch) -> PackingStats:
    """
    Calculate packing efficiency statistics.

    Args:
        batch: PackedBatch to analyze

    Returns:
        PackingStats with efficiency metrics
    """
    total_tokens = 0
    real_tokens = 0
    padding_tokens = 0

    for seq in batch.sequences:
        total_tokens += len(seq.token_ids)
        real_tokens += seq.num_real_tokens
        padding_tokens += seq.num_padding_tokens

    padding_ratio = padding_tokens / total_tokens if total_tokens > 0 else 0.0
    efficiency = real_tokens / total_tokens if total_tokens > 0 else 0.0

    # Estimate throughput improvement
    # Naive padding would pad each sequence to max_length
    if batch.sequences:
        max_len = len(batch.sequences[0].token_ids)
        naive_tokens = batch.total_source_sequences * max_len
        throughput_improvement = naive_tokens / total_tokens if total_tokens > 0 else 1.0
    else:
        throughput_improvement = 1.0

    return PackingStats(
        total_tokens=total_tokens,
        real_tokens=real_tokens,
        padding_tokens=padding_tokens,
        padding_ratio=padding_ratio,
        efficiency=efficiency,
        avg_sequences_per_pack=batch.packing_ratio,
        throughput_improvement=throughput_improvement,
    )
