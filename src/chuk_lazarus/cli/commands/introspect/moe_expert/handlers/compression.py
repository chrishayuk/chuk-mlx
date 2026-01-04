"""Handler for 'compression' action - analyze compression opportunities."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import (
    ExpertRouter,
    analyze_compression_opportunities,
    get_prompts_flat,
    print_compression_summary,
)
from ......introspection.moe.compression import (
    find_merge_candidates_with_activations,
    print_activation_overlap_matrix,
)
from ..formatters import format_header


def handle_compression(args: Namespace) -> None:
    """Handle the 'compression' action - analyze compression opportunities.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert compression -m openai/gpt-oss-20b
        lazarus introspect moe-expert compression -m openai/gpt-oss-20b --layer 10 --threshold 0.8
    """
    asyncio.run(_async_compression(args))


async def _async_compression(args: Namespace) -> None:
    """Async implementation of compression handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    threshold = getattr(args, "threshold", 0.8)
    num_prompts = getattr(args, "num_prompts", 30)
    show_overlap = getattr(args, "show_overlap", False)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("EXPERT COMPRESSION ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Architecture: {info.architecture.value}")
        print(f"Experts: {info.num_experts}")
        print(f"Merge threshold: {threshold}")
        print()

        # Get sample prompts
        prompts = [p for _, p in get_prompts_flat()[:num_prompts]]

        # Determine target layer
        target_layer = layer if layer is not None else info.moe_layers[0]

        print(f"Analyzing layer {target_layer} with {len(prompts)} prompts...")
        print()

        # Run compression analysis
        analysis = await router.analyze_compression(
            prompts,
            layer_idx=target_layer,
            similarity_threshold=threshold,
        )

        if analysis is None:
            print("Could not analyze compression opportunities")
            return

        # Print summary
        print_compression_summary(analysis)

        # Show merge candidates with activation overlap
        if analysis.similarities:
            # Find candidates that pass both weight and activation thresholds
            candidates = find_merge_candidates_with_activations(
                list(analysis.similarities),
                weight_threshold=threshold,
                activation_threshold=0.3,
                require_both=False,
            )

            if candidates:
                print()
                print("Merge Candidates (Weight + Activation Analysis):")
                print("-" * 60)
                for expert_a, expert_b, weight_sim, act_overlap in candidates[:10]:
                    print(f"  E{expert_a} + E{expert_b}:")
                    print(f"    Weight similarity: {weight_sim:.2%}")
                    print(f"    Activation overlap: {act_overlap:.2%}")

        # Show activation overlap matrix if requested
        if show_overlap and analysis.similarities:
            print()
            print_activation_overlap_matrix(
                list(analysis.similarities),
                info.num_experts,
            )

        print()

        # Show potential savings
        if analysis.merge_candidates:
            num_mergeable = len(analysis.merge_candidates)
            potential_reduction = num_mergeable / info.num_experts * 100
            print(f"Potential expert reduction: {num_mergeable}/{info.num_experts} ({potential_reduction:.0f}%)")

        print("=" * 70)
