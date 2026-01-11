"""Suggest vocabulary additions command handler."""

import logging

from .._types import AnalyzeVocabSuggestConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_vocab_suggest(config: AnalyzeVocabSuggestConfig) -> None:
    """Suggest vocabulary additions based on corpus analysis.

    Args:
        config: Vocab suggestion configuration.
    """
    from .....data.tokenizers.analyze import InductionConfig, analyze_vocab_induction
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    induction_config = InductionConfig(
        min_frequency=config.min_freq,
        min_fragmentation=config.min_frag,
        max_candidates=config.limit,
    )

    logger.info(f"Analyzing vocabulary on {len(texts)} texts...")
    report = analyze_vocab_induction(texts, tokenizer, induction_config)

    print("\n=== Vocabulary Induction Report ===")
    print(f"Candidates found:     {report.total_candidates}")
    print(f"Potential savings:    {report.total_potential_savings:,} tokens")
    print(f"Savings percent:      {report.savings_percent:.1f}%")

    if report.domain_breakdown:
        print("\nBy domain:")
        for domain, count in sorted(report.domain_breakdown.items()):
            print(f"  {domain}: {count}")

    print(f"\nTop {min(config.show, len(report.candidates))} candidates:")
    print("-" * 70)
    print(f"{'Token':<30} {'Freq':>8} {'Tokens':>8} {'Savings':>10}")
    print("-" * 70)

    for c in report.candidates[: config.show]:
        token_display = repr(c.token_str)[:28]
        print(f"{token_display:<30} {c.frequency:>8} {c.current_tokens:>8} {c.total_savings:>10}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")
