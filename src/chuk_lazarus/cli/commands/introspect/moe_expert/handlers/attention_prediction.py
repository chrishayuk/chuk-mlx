"""Handler for attention-based routing prediction.

Predicts expert routing from attention patterns alone:
- Extract attention features (entropy, self-attention, etc.)
- Learn mapping from attention to routing
- Evaluate prediction accuracy
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
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    # Language
    "The capital of France is",
    "A synonym for happy is",
    "Shakespeare wrote",
    # Reasoning
    "If all cats are mammals, and all mammals",
    "The sum of angles in a triangle is",
    # Diverse
    "Once upon a time in a",
    "The quick brown fox jumps",
    "Hello, my name is",
    "To solve this problem, first",
]


def handle_attention_prediction(args: Namespace) -> None:
    """Handle attention-based routing prediction analysis.

    Args:
        args: Command arguments with model, prompts, etc.
    """
    asyncio.run(_async_handle_attention_prediction(args))


async def _async_handle_attention_prediction(args: Namespace) -> dict:
    """Async implementation of attention prediction analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.attention_prediction_service import (
        AttentionPredictionService,
        print_prediction_analysis,
    )

    model_id = args.model
    num_prompts = getattr(args, "num_prompts", 30)
    layers_arg = getattr(args, "layers", None)
    learn = getattr(args, "learn", True)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = AttentionPredictionService(router)

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

    print(f"Analyzing {len(prompts)} prompts...")

    analysis = await service.analyze(
        prompts=prompts,
        layers=layers,
        learn_mappings=learn,
        model_id=model_id,
    )

    # Print report
    print_prediction_analysis(analysis)

    # Return summary
    return {
        "model": model_id,
        "num_prompts": len(prompts),
        "layers_analyzed": analysis.num_layers_analyzed,
        "top1_accuracy": analysis.evaluation.top1_accuracy,
        "topk_overlap": analysis.evaluation.topk_overlap,
        "weight_correlation": analysis.evaluation.weight_correlation,
        "predictability_score": analysis.predictability_score,
        "layer_accuracies": analysis.evaluation.layer_accuracies,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Attention-based routing prediction")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=30, help="Number of prompts"
    )
    parser.add_argument(
        "-l", "--layers", help="Comma-separated layer indices to analyze"
    )
    parser.add_argument(
        "--no-learn", action="store_true", help="Skip learning phase"
    )

    args = parser.parse_args()
    args.learn = not args.no_learn
    result = asyncio.run(_async_handle_attention_prediction(args))
    return result


if __name__ == "__main__":
    main()
