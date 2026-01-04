"""Tokenizer encode command handler."""

import logging

from .._types import EncodeConfig

logger = logging.getLogger(__name__)


def tokenizer_encode(config: EncodeConfig) -> None:
    """Encode text and display tokens.

    Args:
        config: Encode configuration.
    """
    from .....data.tokenizers.token_display import TokenDisplayUtility
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)
    display = TokenDisplayUtility(tokenizer)

    if config.text:
        texts = [config.text]
    elif config.file:
        with open(config.file) as f:
            texts = [f.read()]
    else:
        # Interactive mode
        print("Enter text to tokenize (Ctrl+D to finish):")
        try:
            texts = [input("> ")]
        except EOFError:
            return

    for text in texts:
        print(f"\nText: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Length: {len(text)} chars\n")
        display.display_tokens_from_prompt(text, add_special_tokens=config.special_tokens)
