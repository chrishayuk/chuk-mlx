"""Special token registry command handler."""

import logging
from typing import Any

from .._types import RuntimeRegistryConfig

logger = logging.getLogger(__name__)


class RuntimeRegistryWithTokenizerConfig(RuntimeRegistryConfig):
    """Extended config with optional tokenizer."""

    tokenizer: str | None = None
    standard: bool = False

    @classmethod
    def from_args(cls, args: Any) -> "RuntimeRegistryWithTokenizerConfig":
        return cls(
            verbose=getattr(args, "verbose", False),
            tokenizer=getattr(args, "tokenizer", None),
            standard=getattr(args, "standard", False),
        )


def runtime_registry(config: RuntimeRegistryWithTokenizerConfig) -> None:
    """Display special token registry.

    Args:
        config: Registry configuration.
    """
    from .....data.tokenizers.runtime import (
        SpecialTokenRegistry,
        TokenCategory,
        create_standard_registry,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    if config.standard:
        registry = create_standard_registry()
    else:
        registry = SpecialTokenRegistry()
        if config.tokenizer:
            tokenizer = load_tokenizer(config.tokenizer)
            # Try to populate from tokenizer's special tokens
            if hasattr(tokenizer, "special_tokens_map"):
                for name, token in tokenizer.special_tokens_map.items():
                    if isinstance(token, str):
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        registry.register(
                            token_str=token,
                            token_id=token_id,
                            category=TokenCategory.CUSTOM,
                            description=name,
                        )

    print("\n=== Special Token Registry ===")
    print(f"Total tokens: {len(registry.tokens)}")

    for entry in registry.tokens:
        print(f"  {entry.token_id:5d}: {entry.token_str:20s} [{entry.category.value}]")
        if entry.description:
            print(f"         {entry.description}")
