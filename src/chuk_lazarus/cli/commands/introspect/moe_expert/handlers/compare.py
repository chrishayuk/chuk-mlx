"""Handler for 'compare' action - compare multiple experts."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_comparison_result


def handle_compare(args: Namespace) -> None:
    """Handle the 'compare' action - compare multiple experts on same prompt.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - experts: Comma-separated expert indices
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert compare -m openai/gpt-oss-20b --experts 6,7,20 -p "def fib(n):"
    """
    asyncio.run(_async_compare(args))


async def _async_compare(args: Namespace) -> None:
    """Async implementation of compare handler."""
    # Validate required arguments
    experts_str = getattr(args, "experts", None)
    if not experts_str:
        print("Error: --experts is required for compare action (e.g., --experts 6,7,20)")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for compare action")
        return

    # Parse expert indices
    try:
        expert_indices = [int(e.strip()) for e in experts_str.split(",")]
    except ValueError:
        print(f"Error: Invalid experts format: {experts_str}")
        print("Expected comma-separated integers (e.g., 6,7,20)")
        return

    if len(expert_indices) < 2:
        print("Error: At least 2 experts required for comparison")
        return

    model_id = args.model
    prompt = args.prompt
    max_tokens = getattr(args, "max_tokens", 100)
    verbose = getattr(args, "verbose", False)

    print(f"Loading model: {model_id}")
    print(f"Comparing experts: {expert_indices}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        result = await router.compare_experts(
            prompt,
            expert_indices,
            max_tokens=max_tokens,
        )

        output = format_comparison_result(result, model_id, verbose=verbose)
        print(output)
