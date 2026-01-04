"""Curriculum length buckets command handler."""

import logging

from .._types import CurriculumLengthBucketsConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def curriculum_length_buckets(config: CurriculumLengthBucketsConfig) -> None:
    """Create curriculum buckets based on token length.

    Args:
        config: Length buckets configuration.
    """
    from .....data.tokenizers.curriculum import create_length_buckets, get_curriculum_schedule
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Creating {config.num_buckets} length buckets...")
    buckets = create_length_buckets(texts, tokenizer, num_buckets=config.num_buckets)

    print("\n=== Length Buckets ===")
    for i, bucket in enumerate(buckets):
        print(
            f"Bucket {i + 1}: {bucket.min_tokens}-{bucket.max_tokens} tokens, "
            f"{bucket.sample_count} samples, avg={bucket.avg_length:.1f}"
        )

    if config.schedule:
        schedule = get_curriculum_schedule(texts, tokenizer, num_buckets=config.num_buckets)
        print("\n=== Curriculum Schedule ===")
        print(f"Total phases:    {len(schedule.phases)}")
        print(f"Warmup samples:  {schedule.warmup_samples}")
        print(f"Ramp samples:    {schedule.ramp_samples}")
