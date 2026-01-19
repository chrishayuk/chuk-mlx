"""Handler for cold expert analysis.

Analyzes rarely-activated experts to understand:
- Which experts are "cold" (< threshold activation rate)
- What tokens/contexts trigger cold experts
- Whether cold experts can be safely pruned
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

# Default prompts for analysis
DEFAULT_PROMPTS = [
    # Arithmetic
    "127 * 89 = ",
    "456 + 789 = ",
    "1000 - 357 = ",
    "What is 45 times 37?",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class Database:",
    "for i in range(10):",
    # Language
    "The capital of France is",
    "A synonym for happy is",
    "The opposite of cold is",
    "Shakespeare wrote",
    # Mixed
    "Calculate 45 * 37 and explain",
    "Paris is to France as Tokyo is to",
    "Translate 'hello' to Spanish",
    # More diverse
    "The quick brown fox jumps",
    "In the beginning, there was",
    "Once upon a time",
    "To be or not to be",
    "Hello, my name is",
]


def handle_cold_experts(args: Namespace) -> None:
    """Handle cold expert analysis.

    Args:
        args: Command arguments with model, threshold, etc.
    """
    asyncio.run(_async_handle_cold_experts(args))


async def _async_handle_cold_experts(args: Namespace) -> dict:
    """Async implementation of cold expert analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.cold_expert_service import (
        ColdExpertService,
        print_cold_expert_report,
    )

    model_id = args.model
    threshold = getattr(args, "threshold", 0.01)
    num_prompts = getattr(args, "num_prompts", 50)
    analyze_triggers = getattr(args, "triggers", True)
    analyze_ablation = getattr(args, "ablation", False)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = ColdExpertService(router)

    # Get prompts
    prompts = DEFAULT_PROMPTS[:num_prompts]
    if hasattr(args, "prompts") and args.prompts:
        prompts = args.prompts

    print(f"Analyzing {len(prompts)} prompts with threshold {threshold:.2%}...")

    analysis = await service.analyze(
        prompts=prompts,
        cold_threshold=threshold,
        analyze_triggers=analyze_triggers,
        analyze_ablation=analyze_ablation,
    )

    # Print report
    print_cold_expert_report(analysis)

    # Return summary for potential JSON output
    return {
        "model": model_id,
        "threshold": threshold,
        "total_experts": analysis.total_experts,
        "cold_count": analysis.cold_expert_count,
        "cold_percentage": analysis.cold_expert_percentage,
        "cold_experts_by_layer": {
            str(layer): [e.expert_idx for e in analysis.get_cold_experts_for_layer(layer)]
            for layer in set(e.layer_idx for e in analysis.cold_experts)
        },
        "pruning_recommendations": [
            {"layer": layer, "expert": exp}
            for layer, exp in analysis.pruning_recommendations
        ],
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze cold experts")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.01, help="Cold threshold (default: 0.01)"
    )
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=50, help="Number of prompts"
    )
    parser.add_argument(
        "--triggers", action="store_true", default=True, help="Analyze triggers"
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Analyze ablation impact (slow)"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_cold_experts(args))
    return result


if __name__ == "__main__":
    main()
