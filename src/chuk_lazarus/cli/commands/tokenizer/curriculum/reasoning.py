"""Curriculum reasoning density command handler."""

import logging

from .._types import CurriculumReasoningConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def curriculum_reasoning_density(config: CurriculumReasoningConfig) -> None:
    """Score texts by reasoning density for curriculum ordering.

    Args:
        config: Reasoning density configuration.
    """
    from .....data.tokenizers.curriculum import (
        get_difficulty_percentiles,
        sort_by_reasoning_density,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Scoring reasoning density on {len(texts)} texts...")
    sorted_scores = sort_by_reasoning_density(texts, tokenizer, descending=config.descending)
    percentiles = get_difficulty_percentiles(texts, tokenizer)

    print("\n=== Reasoning Density ===")
    print(f"Mean score:     {percentiles.mean:.4f}")
    print(f"P25:            {percentiles.p25:.4f}")
    print(f"P50 (median):   {percentiles.p50:.4f}")
    print(f"P75:            {percentiles.p75:.4f}")
    print(f"P90:            {percentiles.p90:.4f}")

    print(f"\nTop {min(10, len(sorted_scores))} by reasoning density:")
    for score in sorted_scores[:10]:
        text_preview = texts[score.text_index][:50]
        print(f"  [{score.text_index}] {score.score:.4f}: {text_preview}...")
