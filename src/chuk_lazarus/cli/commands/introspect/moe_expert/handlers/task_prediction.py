"""Handler for task-aware expert prediction.

Uses early-layer probes to predict which experts will be needed:
- Train probe at L4 (or specified layer)
- Predict expert routing for later layers
- Evaluate prefetch efficiency
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
    "1000 / 25 = ",
    "Calculate the square root of 144",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    "for item in items:",
    # Language
    "The capital of France is",
    "A synonym for happy is",
    "Shakespeare wrote many",
    "The opposite of hot is",
    # Reasoning
    "If all cats are mammals, then",
    "To solve this equation, first",
    "The pattern continues as",
    # Diverse
    "Once upon a time",
    "The quick brown fox",
    "Hello, my name is",
    "In the beginning",
    "To summarize the main points",
]


def handle_task_prediction(args: Namespace) -> None:
    """Handle task-aware expert prediction analysis.

    Args:
        args: Command arguments with model, probe layer, etc.
    """
    asyncio.run(_async_handle_task_prediction(args))


async def _async_handle_task_prediction(args: Namespace) -> dict:
    """Async implementation of task prediction analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.task_prediction_service import (
        TaskPredictionService,
        print_task_prediction_analysis,
    )

    model_id = args.model
    num_prompts = getattr(args, "num_prompts", 40)
    probe_layer = getattr(args, "probe_layer", 4)
    target_layers_arg = getattr(args, "target_layers", None)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = TaskPredictionService(router)

    # Get prompts
    prompts = DEFAULT_PROMPTS[:num_prompts]
    if hasattr(args, "prompts") and args.prompts:
        prompts = args.prompts

    # Parse target layers
    target_layers = None
    if target_layers_arg:
        if isinstance(target_layers_arg, str):
            target_layers = [int(x.strip()) for x in target_layers_arg.split(",")]
        elif isinstance(target_layers_arg, list):
            target_layers = target_layers_arg

    print(f"Analyzing task prediction from L{probe_layer}...")

    analysis = await service.analyze(
        prompts=prompts,
        probe_layer=probe_layer,
        target_layers=target_layers,
        model_id=model_id,
    )

    # Print report
    print_task_prediction_analysis(analysis)

    # Return summary
    return {
        "model": model_id,
        "probe_layer": analysis.probe_layer,
        "num_prompts": len(prompts),
        "overall_accuracy": analysis.overall_accuracy,
        "overall_prefetch_efficiency": analysis.overall_prefetch_efficiency,
        "best_predicted_layers": analysis.best_predicted_layers,
        "layer_metrics": {
            str(k): {
                "accuracy": v.accuracy,
                "precision": v.precision,
                "recall": v.recall,
            }
            for k, v in analysis.layer_metrics.items()
        },
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Task-aware expert prediction")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=40, help="Number of prompts"
    )
    parser.add_argument(
        "-p", "--probe-layer", type=int, default=4, help="Layer to probe from"
    )
    parser.add_argument(
        "-t", "--target-layers", help="Comma-separated target layer indices"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_task_prediction(args))
    return result


if __name__ == "__main__":
    main()
