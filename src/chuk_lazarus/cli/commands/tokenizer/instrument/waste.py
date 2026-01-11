"""Padding and truncation waste analysis command handler."""

import logging

from .._types import InstrumentWasteConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def instrument_waste(config: InstrumentWasteConfig) -> None:
    """Analyze padding and truncation waste.

    Args:
        config: Waste analysis configuration.
    """
    from .....data.tokenizers.instrumentation import analyze_waste
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing waste on {len(texts)} texts with max_length={config.max_length}...")

    report = analyze_waste(texts, tokenizer, max_length=config.max_length)

    print("\n=== Token Waste Report ===")
    print(f"  Max length:        {report.max_length}")
    print(f"  Total samples:     {report.total_samples}")
    print(f"  Overall efficiency: {report.overall_efficiency:.1%}")

    print("\n--- Padding Analysis ---")
    print(f"  Total positions:   {report.padding.total_positions:,}")
    print(f"  Content tokens:    {report.padding.total_content_tokens:,}")
    print(f"  Padding tokens:    {report.padding.total_padding_tokens:,}")
    print(f"  Padding rate:      {report.padding.padding_rate:.1%}")
    print(f"  Efficiency:        {report.padding.efficiency:.1%}")
    print(f"  Mean padding:      {report.padding.mean_padding_per_sample:.1f}")
    print(f"  Max padding:       {report.padding.max_padding}")

    print("\n--- Truncation Analysis ---")
    print(
        f"  Truncated samples: {report.truncation.truncated_samples}/{report.truncation.total_samples}"
    )
    print(f"  Truncation rate:   {report.truncation.truncation_rate:.1%}")
    print(f"  Tokens lost:       {report.truncation.total_tokens_lost:,}")
    print(f"  Content loss rate: {report.truncation.content_loss_rate:.1%}")
    print(f"  Minor truncation:  {report.truncation.minor_truncation}")
    print(f"  Major truncation:  {report.truncation.major_truncation}")
    print(f"  Severe truncation: {report.truncation.severe_truncation}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")
