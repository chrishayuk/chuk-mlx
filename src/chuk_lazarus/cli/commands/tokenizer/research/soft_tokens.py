"""Soft token research command handler."""

import json
import logging

import numpy as np

from .._types import ResearchSoftTokensConfig

logger = logging.getLogger(__name__)


def research_soft_tokens(config: ResearchSoftTokensConfig) -> None:
    """Create and display soft token bank.

    Args:
        config: Soft tokens configuration.
    """
    from .....data.tokenizers.research import (
        InitializationMethod,
        create_prompt_tuning_bank,
    )

    init_method = InitializationMethod(config.init_method.value)

    bank = create_prompt_tuning_bank(
        num_tokens=config.num_tokens,
        embedding_dim=config.embedding_dim,
        prefix=config.prefix,
        init_method=init_method,
        init_std=config.init_std,
    )

    print("\n=== Soft Token Bank ===")
    print(f"Name:           {bank.name}")
    print(f"Embedding dim:  {bank.embedding_dim}")
    print(f"Num tokens:     {len(bank.tokens)}")
    print(f"Init method:    {init_method.value}")
    print("\nTokens:")

    for token in bank.tokens:
        emb = token.embedding_array
        norm = np.linalg.norm(emb)
        print(f"  {token.token.name} (ID: {token.token.token_id})")
        print(f"    Norm: {norm:.4f}, Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")

    if config.output:
        output_data = {
            "name": bank.name,
            "embedding_dim": bank.embedding_dim,
            "tokens": [
                {
                    "name": t.token.name,
                    "token_id": t.token.token_id,
                    "embedding": t.embedding,
                }
                for t in bank.tokens
            ],
        }
        with open(config.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {config.output}")
