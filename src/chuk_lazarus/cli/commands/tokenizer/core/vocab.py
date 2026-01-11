"""Tokenizer vocab command handler."""

import logging

from .._types import VocabConfig

logger = logging.getLogger(__name__)


def tokenizer_vocab(config: VocabConfig) -> None:
    """Display vocabulary information.

    Args:
        config: Vocab configuration.
    """
    from .....data.tokenizers.token_display import TokenDisplayUtility
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    vocab = tokenizer.get_vocab()
    print("\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab)}")

    if hasattr(tokenizer, "pad_token_id"):
        print(f"  Pad token ID: {tokenizer.pad_token_id}")
    if hasattr(tokenizer, "eos_token_id"):
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
    if hasattr(tokenizer, "bos_token_id"):
        print(f"  BOS token ID: {tokenizer.bos_token_id}")
    if hasattr(tokenizer, "unk_token_id"):
        print(f"  UNK token ID: {tokenizer.unk_token_id}")

    if config.show_all:
        display = TokenDisplayUtility(tokenizer)
        display.display_full_vocabulary(
            chunk_size=config.chunk_size, pause_between_chunks=config.pause
        )
    elif config.search:
        # Search for tokens containing the search string
        print(f"\nTokens containing '{config.search}':")
        matches = [
            (token, id) for token, id in vocab.items() if config.search.lower() in token.lower()
        ]
        matches.sort(key=lambda x: x[1])
        for token, id in matches[: config.limit]:
            decoded = tokenizer.decode([id])
            print(f"  {id:6d}: {repr(token):30s} -> {repr(decoded)}")
        if len(matches) > config.limit:
            print(f"  ... and {len(matches) - config.limit} more matches")
