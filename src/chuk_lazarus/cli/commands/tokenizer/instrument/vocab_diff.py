"""Vocabulary diff command handler."""

import logging

from .._types import InstrumentVocabDiffConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def instrument_vocab_diff(config: InstrumentVocabDiffConfig) -> None:
    """Compare two tokenizers on a corpus.

    Args:
        config: Vocab diff configuration.
    """
    from .....data.tokenizers.instrumentation import (
        compare_vocab_impact,
        estimate_retokenization_cost,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {config.tokenizer1}")
    tok1 = load_tokenizer(config.tokenizer1)
    logger.info(f"Loading tokenizer 2: {config.tokenizer2}")
    tok2 = load_tokenizer(config.tokenizer2)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Comparing tokenizers on {len(texts)} texts...")

    report = compare_vocab_impact(
        texts,
        tok1,
        tok2,
        tokenizer1_name=config.tokenizer1,
        tokenizer2_name=config.tokenizer2,
        max_examples=config.examples,
    )

    print("\n=== Vocabulary Comparison ===")
    print(f"  Tokenizer 1:       {report.tokenizer1_name}")
    print(f"  Tokenizer 2:       {report.tokenizer2_name}")
    print(f"  Vocab size 1:      {report.tokenizer1_vocab_size:,}")
    print(f"  Vocab size 2:      {report.tokenizer2_vocab_size:,}")

    print("\n--- Token Counts ---")
    print(f"  Tokens (tok1):     {report.tokens1_total:,}")
    print(f"  Tokens (tok2):     {report.tokens2_total:,}")
    print(f"  Difference:        {report.token_count_diff:+,}")
    print(f"  Token ratio:       {report.token_count_ratio:.2f}x")

    print("\n--- Compression ---")
    print(f"  Chars/token (1):   {report.chars_per_token1:.2f}")
    print(f"  Chars/token (2):   {report.chars_per_token2:.2f}")
    print(f"  Compression impr:  {report.compression_improvement:.2f}x")

    print("\n--- Per-Sample Analysis ---")
    print(f"  Improved:          {report.samples_improved}")
    print(f"  Same:              {report.samples_same}")
    print(f"  Worse:             {report.samples_worse}")
    print(f"  Improvement rate:  {report.improvement_rate:.1%}")

    print("\n--- Training Impact ---")
    print(f"  Training speedup:  {report.training_speedup:.2f}x")
    print(f"  Memory reduction:  {report.memory_reduction:.1%}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Retokenization cost
    if config.cost:
        cost = estimate_retokenization_cost(texts, tok1, tok2)
        print("\n=== Retokenization Cost ===")
        print(
            f"  Vocab overlap:     {cost['vocab_overlap']:,} tokens ({cost['vocab_overlap_rate']:.1%})"
        )
        print(f"  New tokens:        {cost['new_tokens']:,}")
        print(f"  Removed tokens:    {cost['removed_tokens']:,}")
        print(f"  Embedding reuse:   {cost['embedding_reuse_rate']:.1%}")
