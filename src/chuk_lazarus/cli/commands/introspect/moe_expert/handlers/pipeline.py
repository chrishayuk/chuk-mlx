"""Handler for 'pipeline' action - track expert pipelines across layers."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, get_prompts_flat
from ......introspection.moe.tracking import (
    analyze_cross_layer_routing,
    print_alignment_matrix,
    print_pipeline_summary,
)
from ..formatters import format_header


def handle_pipeline(args: Namespace) -> None:
    """Handle the 'pipeline' action - track expert pipelines across layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert pipeline -m openai/gpt-oss-20b
        lazarus introspect moe-expert pipeline -m openai/gpt-oss-20b --prompts "2+2=" "What is Python?"
    """
    asyncio.run(_async_pipeline(args))


async def _async_pipeline(args: Namespace) -> None:
    """Async implementation of pipeline handler."""
    model_id = args.model
    prompts = getattr(args, "prompts", None)
    num_prompts = getattr(args, "num_prompts", 20)
    min_coverage = getattr(args, "min_coverage", 0.3)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("CROSS-LAYER EXPERT PIPELINE ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Architecture: {info.architecture.value}")
        print(f"Experts: {info.num_experts}")
        print(f"MoE layers: {len(info.moe_layers)} layers")
        print()

        # Get prompts
        if prompts is None:
            prompts = [p for _, p in get_prompts_flat()[:num_prompts]]

        print(f"Analyzing {len(prompts)} prompts across {len(info.moe_layers)} layers...")
        print()

        # Collect routing data across all layers
        all_layer_weights = []
        for prompt in prompts[:5]:  # Use subset for efficiency
            weights = await router.capture_router_weights(prompt)
            all_layer_weights.extend(weights)

        if not all_layer_weights:
            print("No routing data captured")
            return

        # Run cross-layer analysis
        analysis = analyze_cross_layer_routing(
            all_layer_weights,
            info.num_experts,
        )

        print(f"Global consistency: {analysis.global_consistency:.2%}")
        print(f"Pipelines identified: {len(analysis.pipelines)}")
        print()

        # Print pipeline summary
        if analysis.pipelines:
            print_pipeline_summary(list(analysis.pipelines))
        else:
            print("No clear pipelines identified (experts may be too generalist)")

        print()

        # Print layer alignment matrix
        if analysis.layer_alignments:
            print_alignment_matrix(list(analysis.layer_alignments))

        print("=" * 70)
