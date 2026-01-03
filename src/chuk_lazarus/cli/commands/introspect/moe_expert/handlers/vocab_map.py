"""Handler for 'vocab-map' action - map vocabulary to experts."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_vocab_map(args: Namespace) -> None:
    """Handle the 'vocab-map' action - map vocabulary tokens to expert preferences.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert vocab-map -m openai/gpt-oss-20b --num-tokens 500
    """
    asyncio.run(_async_vocab_map(args))


async def _async_vocab_map(args: Namespace) -> None:
    """Async implementation of vocab_map handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    num_tokens = getattr(args, "num_tokens", 500)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        tokenizer = router.tokenizer

        print(format_header("VOCABULARY-EXPERT MAPPING"))
        print(f"Model: {model_id}")
        print(f"Analyzing {num_tokens} tokens...")
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        # Map tokens to their preferred experts
        expert_vocab: dict[int, list[str]] = {i: [] for i in range(info.num_experts)}
        vocab_size = getattr(tokenizer, "vocab_size", num_tokens)

        for token_id in range(min(num_tokens, vocab_size)):
            try:
                token = tokenizer.decode([token_id])
                if not token or len(token.strip()) == 0:
                    continue

                weights = await router.capture_router_weights(token, layers=[target_layer])
                if weights and weights[0].positions:
                    top_expert = weights[0].positions[0].expert_indices[0]
                    expert_vocab[top_expert].append(token)
            except Exception:
                continue

        print(f"Expert vocabulary sizes (layer {target_layer}):")
        print("-" * 50)

        for expert_idx in sorted(expert_vocab.keys()):
            tokens = expert_vocab[expert_idx]
            if tokens:
                examples = ", ".join(f"'{t[:8]}'" for t in tokens[:5])
                print(f"  E{expert_idx:2d}: {len(tokens):4d} tokens - {examples}")

        # Find specialists vs generalists
        avg_tokens = num_tokens / info.num_experts
        specialists = [e for e, t in expert_vocab.items() if len(t) > avg_tokens * 1.5]
        rare = [e for e, t in expert_vocab.items() if len(t) < avg_tokens * 0.5]

        print()
        print(f"High-coverage experts: {specialists}")
        print(f"Low-coverage experts: {rare}")

        print("=" * 70)
