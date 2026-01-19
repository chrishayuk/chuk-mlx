"""Handler for cross-layer expert circuit analysis.

Identifies expert circuits that form functional units across layers:
- Track co-activation patterns across layers
- Identify stable circuits (L4-E17 → L8-E35 → L12-E3)
- Correlate circuits with computation types
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    # Arithmetic
    "127 * 89 = ",
    "456 + 789 = ",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    # Language
    "The capital of France is",
    "A synonym for happy is",
]


def handle_expert_circuits(args: Namespace) -> None:
    """Handle expert circuit analysis.

    Args:
        args: Command arguments with model, prompts, etc.
    """
    asyncio.run(_async_handle_expert_circuits(args))


async def _async_handle_expert_circuits(args: Namespace) -> dict:
    """Async implementation of expert circuit analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.tracking import (
        analyze_cross_layer_routing,
        print_alignment_matrix,
        print_pipeline_summary,
    )

    model_id = args.model
    num_prompts = getattr(args, "num_prompts", 20)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    prompts = DEFAULT_PROMPTS[:num_prompts]

    print(f"Analyzing {len(prompts)} prompts for cross-layer circuits...")

    # Collect router weights for all prompts
    all_weights = []
    for prompt in prompts:
        weights_list = await router.capture_router_weights(prompt)
        all_weights.extend(weights_list)

    # Analyze cross-layer patterns
    analysis = analyze_cross_layer_routing(
        all_weights,
        num_experts=router.info.num_experts,
    )

    # Print results
    print("\n" + "=" * 70)
    print("CROSS-LAYER EXPERT CIRCUIT ANALYSIS")
    print("=" * 70)

    print(f"\nLayers analyzed: {analysis.num_layers}")
    print(f"Experts per layer: {analysis.num_experts}")
    print(f"Global consistency: {analysis.global_consistency:.2%}")
    print(f"Pipelines found: {len(analysis.pipelines)}")

    if analysis.pipelines:
        print_pipeline_summary(list(analysis.pipelines))

    if analysis.layer_alignments:
        print_alignment_matrix(list(analysis.layer_alignments))

    # Build circuit summary
    circuits = []
    for pipeline in analysis.pipelines[:10]:
        circuit = {
            "name": pipeline.name,
            "category": pipeline.category.value,
            "path": [
                {"layer": n.layer_idx, "expert": n.expert_idx}
                for n in pipeline.nodes
            ],
            "consistency": pipeline.consistency_score,
            "coverage": pipeline.coverage,
        }
        circuits.append(circuit)

    return {
        "num_layers": analysis.num_layers,
        "num_experts": analysis.num_experts,
        "global_consistency": analysis.global_consistency,
        "num_pipelines": len(analysis.pipelines),
        "circuits": circuits,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze expert circuits")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=20, help="Number of prompts"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_expert_circuits(args))
    return result


if __name__ == "__main__":
    main()
