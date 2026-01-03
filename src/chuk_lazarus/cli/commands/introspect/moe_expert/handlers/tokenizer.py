"""Handler for 'tokenizer' action - analyze tokenizer-expert relationships."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_tokenizer(args: Namespace) -> None:
    """Handle the 'tokenizer' action - analyze tokenizer-expert mappings.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert tokenizer -m openai/gpt-oss-20b
    """
    asyncio.run(_async_tokenizer(args))


async def _async_tokenizer(args: Namespace) -> None:
    """Async implementation of tokenizer handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    num_tokens = getattr(args, "num_tokens", 100)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        tokenizer = router.tokenizer

        print(format_header("TOKENIZER-EXPERT ANALYSIS"))
        print(f"Model: {model_id}")
        print(
            f"Vocabulary size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'unknown'}"
        )
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        # Sample tokens and analyze routing
        expert_tokens: dict[int, list[str]] = {i: [] for i in range(info.num_experts)}

        # Analyze individual tokens
        for token_id in range(min(num_tokens, getattr(tokenizer, "vocab_size", 1000))):
            try:
                token = tokenizer.decode([token_id])
                if not token or token.isspace():
                    continue

                weights = await router.capture_router_weights(token, layers=[target_layer])
                if weights and weights[0].positions:
                    top_expert = weights[0].positions[0].expert_indices[0]
                    expert_tokens[top_expert].append(token)
            except Exception:
                continue

        print(f"Token distribution across experts (layer {target_layer}):")
        for expert_idx in sorted(expert_tokens.keys()):
            tokens = expert_tokens[expert_idx]
            if tokens:
                examples = ", ".join(f"'{t[:10]}'" for t in tokens[:5])
                print(f"  E{expert_idx:2d}: {len(tokens):4d} tokens - {examples}")

        print("=" * 70)
