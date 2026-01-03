"""Handler for 'topk' action - vary top-k expert selection."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_topk_result


def handle_topk(args: Namespace) -> None:
    """Handle the 'topk' action - generate with modified top-k.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - k: Top-k value to use
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert topk -m openai/gpt-oss-20b --k 1 -p "Hello world"
    """
    asyncio.run(_async_topk(args))


async def _async_topk(args: Namespace) -> None:
    """Async implementation of topk handler."""
    if not hasattr(args, "k") or args.k is None:
        print("Error: --k is required for topk action")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for topk action")
        return

    model_id = args.model
    k = args.k
    prompt = args.prompt
    max_tokens = getattr(args, "max_tokens", 100)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        result = await router.generate_with_topk(
            prompt,
            k,
            max_tokens=max_tokens,
        )

        output = format_topk_result(result, model_id)
        print(output)
