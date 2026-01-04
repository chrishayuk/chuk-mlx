"""Analyze token entropy command handler."""

import logging

from .._types import AnalyzeEntropyConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_entropy(config: AnalyzeEntropyConfig) -> None:
    """Analyze token entropy distribution.

    Args:
        config: Entropy analysis configuration.
    """
    from .....data.tokenizers.analyze import analyze_entropy as do_analyze
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing entropy on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer, top_n=config.top_n)

    print("\n=== Entropy Report ===")
    print(f"Entropy:           {report.entropy:.4f} bits")
    print(f"Perplexity:        {report.perplexity:.2f}")
    print(f"Normalized:        {report.normalized_entropy:.4f}")
    print(f"Uniformity:        {report.uniformity_score:.2%}")
    print(f"Concentration:     {report.concentration_ratio:.2%}")

    if report.distribution:
        print(f"\nTop {len(report.distribution.top_tokens)} tokens:")
        for tok, count in list(report.distribution.top_tokens.items())[:10]:
            print(f"  {tok!r:20} {count:,}")
