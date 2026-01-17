"""Handler for generation dynamics analysis.

Analyzes expert routing behavior during autoregressive generation:
- Token-by-token routing traces
- Expert handoff patterns
- Phase transitions
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "127 * 89 = ",
    "def fibonacci(n):",
    "The capital of France is",
    "Once upon a time",
    "To solve this problem, we need to",
]


def handle_generation_dynamics(args: Namespace) -> None:
    """Handle generation dynamics analysis.

    Args:
        args: Command arguments with model, prompt, etc.
    """
    asyncio.run(_async_handle_generation_dynamics(args))


async def _async_handle_generation_dynamics(args: Namespace) -> dict:
    """Async implementation of generation dynamics analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.generation_dynamics_service import (
        GenerationDynamicsService,
        print_dynamics_analysis,
        print_generation_trace,
    )

    model_id = args.model
    max_tokens = getattr(args, "max_tokens", 30)
    single_prompt = getattr(args, "prompt", None)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = GenerationDynamicsService(router)

    if single_prompt:
        # Single prompt analysis
        print(f"Analyzing generation for: {single_prompt[:50]}...")
        trace = await service.analyze_generation(single_prompt, max_tokens=max_tokens)
        print_generation_trace(trace)

        return {
            "prompt": single_prompt,
            "generated": trace.generated_text,
            "num_tokens": len(trace.snapshots),
            "consistency": trace.consistency_score,
            "handoffs": len(trace.handoffs),
            "phase_boundaries": list(trace.phase_boundaries),
        }
    else:
        # Batch analysis
        prompts = DEFAULT_PROMPTS
        if hasattr(args, "prompts") and args.prompts:
            prompts = args.prompts

        print(f"Analyzing {len(prompts)} prompts...")
        analysis = await service.analyze_batch(prompts, max_tokens=max_tokens)
        print_dynamics_analysis(analysis)

        return {
            "num_traces": analysis.num_traces,
            "avg_consistency": analysis.avg_consistency,
            "avg_handoffs_per_token": analysis.avg_handoffs_per_token,
            "phase_patterns_detected": analysis.phase_pattern_detected,
            "layer_stability": {str(k): v for k, v in analysis.layer_stability.items()},
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze generation dynamics")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument("-p", "--prompt", help="Single prompt to analyze")
    parser.add_argument(
        "-t", "--max-tokens", type=int, default=30, help="Max tokens to generate"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_generation_dynamics(args))
    return result


if __name__ == "__main__":
    main()
