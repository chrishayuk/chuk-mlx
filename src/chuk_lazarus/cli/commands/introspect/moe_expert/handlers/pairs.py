"""Handler for 'pairs' action - test specific expert pairs."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_pairs(args: Namespace) -> None:
    """Handle the 'pairs' action - test specific expert pairs/groups.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - experts: Comma-separated expert indices
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert pairs -m openai/gpt-oss-20b --experts 6,7 -p "127 * 89 = "
    """
    asyncio.run(_async_pairs(args))


async def _async_pairs(args: Namespace) -> None:
    """Async implementation of pairs handler."""
    experts_str = getattr(args, "experts", None)
    if not experts_str:
        print("Error: --experts is required for pairs action")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for pairs action")
        return

    try:
        expert_indices = [int(e.strip()) for e in experts_str.split(",")]
    except ValueError:
        print(f"Error: Invalid experts format: {experts_str}")
        return

    model_id = args.model
    prompt = args.prompt
    max_tokens = getattr(args, "max_tokens", 100)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        print(format_header(f"EXPERT PAIRS TEST - Experts {expert_indices}"))
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print()

        # Compare each expert individually
        for expert_idx in expert_indices:
            result = await router.chat_with_expert(prompt, expert_idx, max_tokens=max_tokens)
            print(f"Expert {expert_idx}: {result.response}")

        # TODO: Test combinations when group forcing is implemented
        print("=" * 70)
