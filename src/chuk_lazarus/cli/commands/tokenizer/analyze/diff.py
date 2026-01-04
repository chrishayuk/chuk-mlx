"""Compare tokenization between tokenizers command handler."""

import logging

from .._types import AnalyzeDiffConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_diff(config: AnalyzeDiffConfig) -> None:
    """Compare tokenization between two tokenizers on a corpus.

    Args:
        config: Diff configuration.
    """
    from .....data.tokenizers.analyze import diff_corpus
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {config.tokenizer1}")
    tok1 = load_tokenizer(config.tokenizer1)
    logger.info(f"Loading tokenizer 2: {config.tokenizer2}")
    tok2 = load_tokenizer(config.tokenizer2)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Comparing tokenization on {len(texts)} texts...")
    diff = diff_corpus(texts, tok1, tok2)

    print("\n=== Corpus Diff Report ===")
    print(f"Texts compared:        {diff.total_texts}")
    print(f"Avg length delta:      {diff.avg_length_delta:+.2f} tokens")
    print(f"Compression improved:  {diff.compression_improvement:.2%}")
    print(f"Tokenizer 1 total:     {diff.tokenizer1_total:,} tokens")
    print(f"Tokenizer 2 total:     {diff.tokenizer2_total:,} tokens")

    if diff.worst_regressions:
        print("\nWorst Regressions (tokenizer 2 is worse):")
        for reg in diff.worst_regressions[:5]:
            print(f"  Delta: {reg.length_delta:+d}, Text: {reg.text[:50]}...")
