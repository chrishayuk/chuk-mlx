"""Token length histogram command handler."""

import logging

from .._types import InstrumentHistogramConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def instrument_histogram(config: InstrumentHistogramConfig) -> None:
    """Display token length histogram.

    Args:
        config: Histogram configuration.
    """
    from .....data.tokenizers.instrumentation import (
        compute_length_histogram,
        format_histogram_ascii,
        get_length_stats,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Computing histogram for {len(texts)} texts...")

    if config.quick:
        stats = get_length_stats(texts, tokenizer)
        print("\n=== Quick Length Stats ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        histogram = compute_length_histogram(texts, tokenizer, num_bins=config.bins)
        print()
        print(format_histogram_ascii(histogram, width=config.width))
