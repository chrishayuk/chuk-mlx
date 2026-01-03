"""Handler for 'collab' action - analyze expert co-activation."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_coactivation


def handle_collaboration(args: Namespace) -> None:
    """Handle the 'collab' action - analyze expert co-activation patterns.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Input prompt (or --prompts for multiple)

    Example:
        lazarus introspect moe-expert collab -m openai/gpt-oss-20b -p "127 * 89 = "
    """
    asyncio.run(_async_collaboration(args))


async def _async_collaboration(args: Namespace) -> None:
    """Async implementation of collaboration handler."""
    # Get prompts - support single or multiple
    prompts: list[str] = []

    if hasattr(args, "prompts") and args.prompts:
        # Multiple prompts from file or pipe-separated
        if args.prompts.startswith("@"):
            with open(args.prompts[1:]) as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = [p.strip() for p in args.prompts.split("|")]
    elif hasattr(args, "prompt") and args.prompt:
        prompts = [args.prompt]
    else:
        print("Error: --prompt/-p or --prompts is required for collab action")
        return

    model_id = args.model
    layer_idx = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")
    print(f"Analyzing co-activation across {len(prompts)} prompt(s)")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        analysis = await router.analyze_coactivation(
            prompts,
            layer_idx=layer_idx,
        )

        target_layer = layer_idx if layer_idx is not None else router.info.moe_layers[0]
        output = format_coactivation(analysis, model_id, target_layer)
        print(output)
