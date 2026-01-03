"""Handler for 'layer-sweep' action - sweep analysis across all layers."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, get_prompts_flat
from ..formatters import format_header


def handle_layer_sweep(args: Namespace) -> None:
    """Handle the 'layer-sweep' action - sweep analysis across all MoE layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert layer-sweep -m openai/gpt-oss-20b
    """
    asyncio.run(_async_layer_sweep(args))


async def _async_layer_sweep(args: Namespace) -> None:
    """Async implementation of layer_sweep handler."""
    model_id = args.model
    num_prompts = getattr(args, "num_prompts", 50)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("LAYER SWEEP ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"MoE layers: {len(info.moe_layers)}")
        print(f"Analyzing {num_prompts} prompts...")
        print()

        # Get sample prompts
        prompts = [p for _, p in get_prompts_flat()[:num_prompts]]

        # Analyze each layer
        print(f"{'Layer':<8} {'Entropy':<10} {'Top Expert':<12} {'Usage %':<10} {'Generalists'}")
        print("-" * 60)

        for layer_idx in info.moe_layers:
            analysis = await router.analyze_coactivation(prompts, layer_idx=layer_idx)

            # Calculate metrics
            expert_counts = {}
            total = analysis.total_activations

            for pair in analysis.top_pairs:
                expert_counts[pair.expert_a] = (
                    expert_counts.get(pair.expert_a, 0) + pair.coactivation_count
                )
                expert_counts[pair.expert_b] = (
                    expert_counts.get(pair.expert_b, 0) + pair.coactivation_count
                )

            top_expert = (
                max(expert_counts.keys(), key=lambda e: expert_counts.get(e, 0))
                if expert_counts
                else 0
            )
            top_usage = expert_counts.get(top_expert, 0) / total * 100 if total > 0 else 0

            # Estimate entropy from usage distribution
            import math

            entropy = 0
            for exp, count in expert_counts.items():
                p = count / total if total > 0 else 0
                if p > 0:
                    entropy -= p * math.log(p)

            num_generalists = len(analysis.generalist_experts)

            print(
                f"L{layer_idx:<6} {entropy:<10.3f} E{top_expert:<11} {top_usage:<10.1f} {num_generalists}"
            )

        print("=" * 70)
