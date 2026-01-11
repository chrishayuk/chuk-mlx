"""OOV and rare token analysis command handler."""

import logging

from .._types import InstrumentOovConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def instrument_oov(config: InstrumentOovConfig) -> None:
    """Analyze OOV and rare tokens.

    Args:
        config: OOV analysis configuration.
    """
    from .....data.tokenizers.instrumentation import (
        analyze_oov,
        find_rare_tokens,
        get_frequency_bands,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing OOV on {len(texts)} texts...")

    # Frequency bands
    bands = get_frequency_bands(texts, tokenizer)
    print("\n=== Token Frequency Bands ===")
    for band, count in sorted(bands.items(), key=lambda x: x[0].value):
        print(f"  {band.value:15s}: {count:,} tokens")

    # OOV report
    report = analyze_oov(texts, tokenizer, vocab_size=config.vocab_size)
    print("\n=== OOV Report ===")
    print(f"  Total tokens:      {report.total_tokens:,}")
    print(f"  Unique tokens:     {report.unique_tokens:,}")
    print(f"  UNK rate:          {report.unk_rate:.2%}")
    print(f"  Singleton rate:    {report.singleton_rate:.2%}")
    print(f"  Vocab utilization: {report.vocab_utilization:.2%}")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

    # Rare tokens
    if config.show_rare:
        rare = find_rare_tokens(texts, tokenizer, max_frequency=config.max_freq, top_k=config.top_k)
        print(f"\n=== Rare Tokens (freq <= {config.max_freq}) ===")
        for token in rare:
            print(f"  {token.token_str!r:20s}: {token.count:4d}x ({token.band.value})")
