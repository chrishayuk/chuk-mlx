"""Handler for 'analyze' action - analyze expert routing patterns."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, get_prompts_flat
from ..formatters import format_header


def handle_analyze(args: Namespace) -> None:
    """Handle the 'analyze' action - analyze expert routing patterns.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert analyze -m openai/gpt-oss-20b
    """
    asyncio.run(_async_analyze(args))


async def _async_analyze(args: Namespace) -> None:
    """Async implementation of analyze handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    num_prompts = getattr(args, "num_prompts", 50)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("EXPERT ROUTING ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Architecture: {info.architecture.value}")
        print(f"Experts: {info.num_experts} per layer")
        print(f"Active per token: {info.num_experts_per_tok}")
        print(f"MoE layers: {list(info.moe_layers)}")
        print()

        # Get sample prompts from dataset
        prompts = [p for _, p in get_prompts_flat()[:num_prompts]]

        target_layer = layer if layer is not None else info.moe_layers[0]
        analysis = await router.analyze_coactivation(prompts, layer_idx=target_layer)

        print(f"Layer {target_layer} Analysis:")
        print(f"  Total activations: {analysis.total_activations}")
        print(f"  Generalist experts: {list(analysis.generalist_experts)}")
        print()

        print("Top co-activated pairs:")
        for pair in analysis.top_pairs[:10]:
            print(
                f"  E{pair.expert_a} + E{pair.expert_b}: "
                f"{pair.coactivation_count} ({pair.coactivation_rate:.1%})"
            )

        print("=" * 70)
