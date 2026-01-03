"""Handler for 'weights' action - show router weights."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_router_weights


def handle_weights(args: Namespace) -> None:
    """Handle the 'weights' action - display router weights for a prompt.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert weights -m openai/gpt-oss-20b -p "Hello world"
    """
    asyncio.run(_async_weights(args))


async def _async_weights(args: Namespace) -> None:
    """Async implementation of weights handler."""
    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for weights action")
        return

    model_id = args.model
    prompt = args.prompt
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        layers = [layer] if layer is not None else None
        weights = await router.capture_router_weights(prompt, layers=layers)

        output = format_router_weights(weights, model_id, prompt)
        print(output)
