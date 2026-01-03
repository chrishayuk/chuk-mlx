"""Handler for 'trace' action - trace token-level expert assignments."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_trace(args: Namespace) -> None:
    """Handle the 'trace' action - trace per-token expert assignments.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert trace -m openai/gpt-oss-20b -p "Hello world"
    """
    asyncio.run(_async_trace(args))


async def _async_trace(args: Namespace) -> None:
    """Async implementation of trace handler."""
    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for trace action")
        return

    model_id = args.model
    prompt = args.prompt
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("TOKEN-EXPERT TRACE"))
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print()

        layers_to_check = [layer] if layer is not None else list(info.moe_layers[:3])

        weights = await router.capture_router_weights(prompt, layers=layers_to_check)

        for layer_weights in weights:
            print(f"Layer {layer_weights.layer_idx}:")
            for pos in layer_weights.positions:
                experts = ", ".join(
                    f"E{e}({w:.2f})" for e, w in zip(pos.expert_indices, pos.weights)
                )
                print(f"  [{pos.position_idx:2d}] '{pos.token:<10}' -> {experts}")
            print()

        print("=" * 70)
