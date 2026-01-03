"""Handler for 'interactive' action - interactive expert explorer."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_interactive(args: Namespace) -> None:
    """Handle the 'interactive' action - interactive expert exploration.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert interactive -m openai/gpt-oss-20b
    """
    asyncio.run(_async_interactive(args))


async def _async_interactive(args: Namespace) -> None:
    """Async implementation of interactive handler."""
    model_id = args.model

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("INTERACTIVE EXPERT EXPLORER"))
        print(f"Model: {model_id}")
        print(f"Experts: {info.num_experts} per layer, {info.num_experts_per_tok} active")
        print(f"MoE Layers: {list(info.moe_layers)}")
        print()
        print("Commands:")
        print("  chat <expert> <prompt>  - Chat with specific expert")
        print("  compare <e1,e2> <prompt> - Compare experts")
        print("  weights <prompt>        - Show router weights")
        print("  quit                    - Exit")
        print()

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()

            try:
                if cmd == "chat" and len(parts) >= 3:
                    expert_idx = int(parts[1])
                    prompt = parts[2]
                    result = await router.chat_with_expert(prompt, expert_idx)
                    print(f"\nExpert {expert_idx}: {result.response}\n")

                elif cmd == "compare" and len(parts) >= 3:
                    experts = [int(e) for e in parts[1].split(",")]
                    prompt = parts[2]
                    result = await router.compare_experts(prompt, experts)
                    print()
                    for r in result.expert_results:
                        print(f"Expert {r.expert_idx}: {r.response}")
                    print()

                elif cmd == "weights" and len(parts) >= 2:
                    prompt = parts[1]
                    weights = await router.capture_router_weights(prompt)
                    print()
                    for layer_w in weights[:3]:  # Show first 3 layers
                        print(f"Layer {layer_w.layer_idx}:")
                        for pos in layer_w.positions[:5]:  # Show first 5 positions
                            experts = [f"E{e}" for e in pos.expert_indices[:3]]
                            print(f"  '{pos.token}': {', '.join(experts)}")
                    print()

                else:
                    print("Unknown command or missing arguments")
                    print("Try: chat 6 Hello world")

            except Exception as e:
                print(f"Error: {e}")

        print("=" * 70)
