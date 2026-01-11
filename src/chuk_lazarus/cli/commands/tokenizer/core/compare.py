"""Tokenizer compare command handler."""

import logging

from .._types import CompareConfig, CompareResult

logger = logging.getLogger(__name__)


def tokenizer_compare(config: CompareConfig) -> CompareResult:
    """Compare tokenization between two tokenizers.

    Args:
        config: Compare configuration.

    Returns:
        Compare result with token counts and differences.
    """
    from .....data.tokenizers.token_display import TokenDisplayUtility
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {config.tokenizer1}")
    tok1 = load_tokenizer(config.tokenizer1)
    logger.info(f"Loading tokenizer 2: {config.tokenizer2}")
    tok2 = load_tokenizer(config.tokenizer2)

    text = config.text

    ids1 = tok1.encode(text)
    ids2 = tok2.encode(text)

    print(f"\nText: {text}")
    print(f"\n{'=' * 60}")
    print(f"{config.tokenizer1}:")
    print(f"{'=' * 60}")
    print(f"  Token count: {len(ids1)}")
    print(f"  Token IDs: {ids1[:20]}{'...' if len(ids1) > 20 else ''}")

    if config.verbose:
        display1 = TokenDisplayUtility(tok1)
        display1.display_tokens_from_prompt(text, add_special_tokens=False)

    print(f"\n{'=' * 60}")
    print(f"{config.tokenizer2}:")
    print(f"{'=' * 60}")
    print(f"  Token count: {len(ids2)}")
    print(f"  Token IDs: {ids2[:20]}{'...' if len(ids2) > 20 else ''}")

    if config.verbose:
        display2 = TokenDisplayUtility(tok2)
        display2.display_tokens_from_prompt(text, add_special_tokens=False)

    difference = len(ids1) - len(ids2)
    ratio = len(ids1) / len(ids2) if len(ids2) > 0 else 0

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    print(f"  Difference: {difference:+d} tokens")
    print(f"  Ratio: {ratio:.2f}x" if len(ids2) > 0 else "  Ratio: N/A")

    return CompareResult(
        tokenizer1_count=len(ids1),
        tokenizer2_count=len(ids2),
        difference=difference,
        ratio=ratio,
    )
