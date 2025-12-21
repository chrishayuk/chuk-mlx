"""Token-length based curriculum generation."""

from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


class LengthBucketConfig(BaseModel):
    """Configuration for length bucketing."""

    num_buckets: int = Field(default=5, ge=1, le=100, description="Number of buckets")
    min_length: int = Field(default=1, ge=1, description="Minimum token length")
    max_length: int | None = Field(default=None, description="Maximum length (auto if None)")
    log_scale: bool = Field(default=False, description="Use logarithmic bucket boundaries")


class LengthBucket(BaseModel):
    """A single length bucket."""

    bucket_id: int = Field(ge=0, description="Bucket identifier")
    min_tokens: int = Field(ge=0, description="Minimum tokens in bucket")
    max_tokens: int = Field(ge=0, description="Maximum tokens in bucket")
    sample_count: int = Field(ge=0, description="Number of samples in bucket")
    sample_indices: list[int] = Field(
        default_factory=list, description="Indices of samples in this bucket"
    )
    avg_length: float = Field(ge=0.0, description="Average token length in bucket")


class CurriculumSchedule(BaseModel):
    """A curriculum learning schedule based on buckets."""

    buckets: list[LengthBucket] = Field(description="Ordered buckets")
    total_samples: int = Field(ge=0, description="Total samples")
    schedule_order: list[int] = Field(description="Bucket IDs in curriculum order (easy to hard)")
    warmup_samples: int = Field(ge=0, description="Samples in warmup phase")
    ramp_samples: int = Field(ge=0, description="Samples in ramp phase")


def create_length_buckets(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: LengthBucketConfig | None = None,
) -> list[LengthBucket]:
    """
    Create length-based buckets for curriculum learning.

    Args:
        texts: List of texts to bucket
        tokenizer: Tokenizer instance
        config: Bucketing configuration

    Returns:
        List of LengthBucket sorted by length
    """
    if config is None:
        config = LengthBucketConfig()

    # Calculate lengths
    lengths = []
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append((i, len(token_ids)))

    if not lengths:
        return []

    # Determine boundaries
    all_lens = [length for _, length in lengths]
    min_len = config.min_length
    max_len = config.max_length or max(all_lens)

    if config.log_scale:
        import math

        log_min = math.log(max(1, min_len))
        log_max = math.log(max(1, max_len))
        log_step = (log_max - log_min) / config.num_buckets
        boundaries = [int(math.exp(log_min + i * log_step)) for i in range(config.num_buckets + 1)]
    else:
        step = (max_len - min_len) / config.num_buckets
        boundaries = [int(min_len + i * step) for i in range(config.num_buckets + 1)]

    # Create buckets
    buckets: list[LengthBucket] = []
    for i in range(config.num_buckets):
        bucket_min = boundaries[i]
        bucket_max = boundaries[i + 1] if i < config.num_buckets - 1 else max_len + 1

        indices = [idx for idx, length in lengths if bucket_min <= length < bucket_max]
        bucket_lengths = [length for idx, length in lengths if bucket_min <= length < bucket_max]

        avg_len = sum(bucket_lengths) / len(bucket_lengths) if bucket_lengths else 0.0

        buckets.append(
            LengthBucket(
                bucket_id=i,
                min_tokens=bucket_min,
                max_tokens=bucket_max - 1,
                sample_count=len(indices),
                sample_indices=indices,
                avg_length=avg_len,
            )
        )

    return buckets


def sort_by_length(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    reverse: bool = False,
) -> list[tuple[int, str, int]]:
    """
    Sort texts by token length.

    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
        reverse: If True, sort longest first

    Returns:
        List of (original_index, text, token_length) sorted by length
    """
    indexed = []
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        indexed.append((i, text, len(token_ids)))

    return sorted(indexed, key=lambda x: x[2], reverse=reverse)


def get_curriculum_schedule(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    config: LengthBucketConfig | None = None,
    warmup_ratio: float = 0.1,
    ramp_ratio: float = 0.3,
) -> CurriculumSchedule:
    """
    Generate a curriculum learning schedule.

    The schedule progresses from short/easy samples to long/hard samples.

    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
        config: Bucketing configuration
        warmup_ratio: Fraction of training for warmup (shortest samples only)
        ramp_ratio: Fraction of training for ramping up difficulty

    Returns:
        CurriculumSchedule with ordered buckets
    """
    buckets = create_length_buckets(texts, tokenizer, config)

    if not buckets:
        return CurriculumSchedule(
            buckets=[],
            total_samples=0,
            schedule_order=[],
            warmup_samples=0,
            ramp_samples=0,
        )

    total = sum(b.sample_count for b in buckets)
    warmup_samples = int(total * warmup_ratio)
    ramp_samples = int(total * ramp_ratio)

    # Schedule order: shortest first
    schedule_order = [b.bucket_id for b in sorted(buckets, key=lambda x: x.avg_length)]

    return CurriculumSchedule(
        buckets=buckets,
        total_samples=total,
        schedule_order=schedule_order,
        warmup_samples=warmup_samples,
        ramp_samples=ramp_samples,
    )
