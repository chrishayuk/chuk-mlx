"""Analyze token efficiency command handler."""

import logging

from .._types import AnalyzeEfficiencyConfig
from .._utils import load_texts

logger = logging.getLogger(__name__)


def analyze_efficiency(config: AnalyzeEfficiencyConfig) -> None:
    """Analyze token efficiency metrics.

    Args:
        config: Efficiency analysis configuration.
    """
    from .....data.tokenizers.analyze import analyze_efficiency as do_analyze
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    texts = load_texts(config.file)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing efficiency on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer)

    print("\n=== Efficiency Report ===")
    print(f"Efficiency Score:  {report.efficiency_score:.1f}/100")

    print("\n--- Sample Statistics ---")
    print(f"Samples:           {report.sample_stats.count:,}")
    print(f"Total tokens:      {report.sample_stats.total_tokens:,}")
    print(f"Mean tokens:       {report.sample_stats.mean:.1f}")
    print(f"Median tokens:     {report.sample_stats.median:.1f}")
    print(f"Std dev:           {report.sample_stats.std:.1f}")
    print(f"P5/P95:            {report.sample_stats.p5:.0f} / {report.sample_stats.p95:.0f}")
    print(f"Min/Max:           {report.sample_stats.min_tokens} / {report.sample_stats.max_tokens}")

    if report.reasoning_steps:
        print("\n--- Reasoning Steps ---")
        print(f"Count:             {report.reasoning_steps.count}")
        print(f"Mean tokens:       {report.reasoning_steps.mean_tokens:.1f}")

    if report.equations:
        print("\n--- Equations ---")
        print(f"Count:             {report.equations.count}")
        print(f"Mean tokens:       {report.equations.mean_tokens:.1f}")

    if report.tool_calls:
        print("\n--- Tool Calls ---")
        print(f"Count:             {report.tool_calls.count}")
        print(f"Mean tokens:       {report.tool_calls.mean_tokens:.1f}")

    print("\n--- Fragmentation ---")
    print(f"Score:             {report.fragmentation.fragmentation_score:.1%}")
    print(f"Single-char:       {report.fragmentation.single_char_tokens:,}")
    print(f"Subword:           {report.fragmentation.subword_tokens:,}")

    if report.fragmentation.fragmented_words:
        print("\nMost fragmented words:")
        for word in report.fragmentation.fragmented_words[:5]:
            print(f"  {word['word']}: {word['tokens']} tokens")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")
