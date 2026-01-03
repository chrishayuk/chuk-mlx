"""Handler for 'control-tokens' action - analyze control token expert assignments."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_control_tokens(args: Namespace) -> None:
    """Handle the 'control-tokens' action.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert control-tokens -m openai/gpt-oss-20b
    """
    asyncio.run(_async_control_tokens(args))


async def _async_control_tokens(args: Namespace) -> None:
    """Async implementation of control_tokens handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        tokenizer = router.tokenizer

        print(format_header("CONTROL TOKEN EXPERT ANALYSIS"))
        print(f"Model: {model_id}")
        print()

        # Common control tokens to check
        control_tokens = [
            "<s>",
            "</s>",
            "<pad>",
            "<unk>",
            "<mask>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[UNK]",
            "[MASK]",
        ]

        target_layer = layer if layer is not None else info.moe_layers[0]

        print(f"Control token routing (layer {target_layer}):")
        print("-" * 50)

        for token in control_tokens:
            try:
                # Check if token exists in vocabulary
                encoded = tokenizer.encode(token)
                if not encoded:
                    continue

                weights = await router.capture_router_weights(token, layers=[target_layer])
                if weights and weights[0].positions:
                    pos = weights[0].positions[0]
                    experts = [
                        f"E{e}({w:.2f})" for e, w in zip(pos.expert_indices[:3], pos.weights[:3])
                    ]
                    print(f"  {token:<20} -> {', '.join(experts)}")
            except Exception:
                continue

        print("=" * 70)
