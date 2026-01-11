"""Handler for 'chat' action - chat with a specific expert."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_chat_result


def handle_chat(args: Namespace) -> None:
    """Handle the 'chat' action - generate with forced expert routing.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - expert: Expert index to force
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert chat -m openai/gpt-oss-20b -e 6 -p "127 * 89 = "
    """
    asyncio.run(_async_chat(args))


async def _async_chat(args: Namespace) -> None:
    """Async implementation of chat handler."""
    # Validate required arguments
    if not hasattr(args, "expert") or args.expert is None:
        print("Error: --expert/-e is required for chat action")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for chat action")
        return

    model_id = args.model
    expert_idx = args.expert
    prompt = args.prompt
    max_tokens = getattr(args, "max_tokens", 100)
    temperature = getattr(args, "temperature", 0.0)
    apply_template = not getattr(args, "raw", False)
    verbose = getattr(args, "verbose", False)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        result = await router.chat_with_expert(
            prompt,
            expert_idx,
            max_tokens=max_tokens,
            temperature=temperature,
            apply_chat_template=apply_template,
        )

        output = format_chat_result(
            result,
            model_id,
            router._moe_type,
            verbose=verbose,
        )
        print(output)
