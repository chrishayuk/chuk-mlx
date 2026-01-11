"""Analyze token coverage command handler."""

import logging

from .._types import AnalyzeCoverageConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_coverage(config: AnalyzeCoverageConfig) -> None:
    """Analyze token coverage on a corpus.

    Args:
        config: Coverage analysis configuration.
    """
    from .....data.tokenizers.analyze import analyze_coverage as do_analyze
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing coverage on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer, include_fragments=config.fragments)

    print("\n=== Coverage Report ===")
    print(f"Total tokens:      {report.total_tokens:,}")
    print(f"Unique tokens:     {report.unique_tokens:,}")
    print(f"UNK rate:          {report.unk_rate:.2%}")
    print(f"Tokens per word:   {report.tokens_per_word:.2f}")
    print(f"Vocab utilization: {report.vocab_utilization:.2%}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.fragments and config.fragments:
        print("\nTop Fragmented Words:")
        for frag in report.fragments.top_fragmented[:10]:
            print(f"  {frag}")
