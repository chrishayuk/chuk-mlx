"""Handler for routing manipulation analysis.

Analyzes and crafts inputs for specific routing:
- Discover expert triggers
- Test routing stability under perturbations
- Craft inputs that activate specific experts
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    # Arithmetic
    "127 * 89 = ",
    "456 + 789 = ",
    "Calculate the sum of 45 and 67",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    # Language
    "The capital of France is",
    "A synonym for happy is",
    "Shakespeare wrote many famous",
    # Mixed
    "To solve this problem, first",
    "The quick brown fox jumps",
    "Once upon a time in a",
    "Hello, my name is",
    "In the beginning there was",
    "Please explain how to",
]


def handle_routing_manipulation(args: Namespace) -> None:
    """Handle routing manipulation analysis.

    Args:
        args: Command arguments with model, etc.
    """
    asyncio.run(_async_handle_routing_manipulation(args))


async def _async_handle_routing_manipulation(args: Namespace) -> dict:
    """Async implementation of routing manipulation analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.routing_manipulation_service import (
        RoutingManipulationService,
        print_manipulation_analysis,
    )

    model_id = args.model
    num_prompts = getattr(args, "num_prompts", 30)
    layers_arg = getattr(args, "layers", None)
    experts_per_layer = getattr(args, "experts_per_layer", 5)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = RoutingManipulationService(router)

    # Get prompts
    prompts = DEFAULT_PROMPTS[:num_prompts]
    if hasattr(args, "prompts") and args.prompts:
        prompts = args.prompts

    # Parse layers
    layers = None
    if layers_arg:
        if isinstance(layers_arg, str):
            layers = [int(x.strip()) for x in layers_arg.split(",")]
        elif isinstance(layers_arg, list):
            layers = layers_arg

    print(f"Analyzing routing manipulation with {len(prompts)} prompts...")

    analysis = await service.analyze(
        prompts=prompts,
        layers_to_analyze=layers,
        experts_per_layer=experts_per_layer,
        model_id=model_id,
    )

    # Print report
    print_manipulation_analysis(analysis)

    # Return summary
    return {
        "model": model_id,
        "num_experts": analysis.num_experts,
        "num_layers": analysis.num_layers,
        "controllability_score": analysis.controllability_score,
        "num_triggers_found": len(analysis.triggers),
        "stable_experts": analysis.stable_experts,
        "volatile_experts": analysis.volatile_experts,
        "top_triggers": [
            {
                "layer": t.layer_idx,
                "expert": t.expert_idx,
                "tokens": list(t.trigger_tokens[:3]),
                "activation_rate": t.activation_rate,
            }
            for t in sorted(
                analysis.triggers,
                key=lambda x: x.activation_rate * x.specificity,
                reverse=True,
            )[:5]
        ],
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Routing manipulation analysis")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=30, help="Number of prompts"
    )
    parser.add_argument(
        "-l", "--layers", help="Comma-separated layer indices to analyze"
    )
    parser.add_argument(
        "-e", "--experts-per-layer", type=int, default=5,
        help="Number of experts per layer to analyze"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_routing_manipulation(args))
    return result


if __name__ == "__main__":
    main()
