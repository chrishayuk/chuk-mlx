"""Tokenizer decode command handler."""

import logging

from .._types import DecodeConfig, DecodeResult

logger = logging.getLogger(__name__)


def tokenizer_decode(config: DecodeConfig) -> DecodeResult:
    """Decode token IDs back to text.

    Args:
        config: Decode configuration.

    Returns:
        Decode result with token IDs and decoded text.
    """
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Parse token IDs from comma-separated or space-separated string
    token_ids = [int(t.strip()) for t in config.ids.replace(",", " ").split()]

    decoded = tokenizer.decode(token_ids)

    return DecodeResult(token_ids=token_ids, decoded=decoded)
