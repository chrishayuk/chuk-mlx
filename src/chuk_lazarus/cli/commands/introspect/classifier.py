"""Classifier probing command handlers for introspection CLI.

This module provides thin CLI wrappers for classifier probing commands.
All business logic is delegated to the framework layer (introspection module).
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import AnalysisDefaults, DisplayDefaults, LayerDepthRatio, ProbeDefaults

logger = logging.getLogger(__name__)


async def introspect_classifier(args: Namespace) -> None:
    """Train multi-class linear probe to detect operation classifiers.

    This is a thin wrapper that:
    1. Converts CLI args to ClassifierConfig
    2. Calls ClassifierService.train_and_evaluate() which handles all logic
    3. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.classifier import ClassifierConfig, ClassifierService
    from .._constants import Delimiters

    # Load categories from file or parse from CLI
    categories_file = getattr(args, "categories_file", None)
    if categories_file:
        import json

        with open(categories_file) as f:
            categories = json.load(f)
    else:
        # Parse from CLI args: --category "label|p1|p2|p3"
        categories = {}
        for cat_arg in getattr(args, "category", []) or []:
            parts = cat_arg.split("|")
            if len(parts) < 2:
                raise ValueError(f"Invalid category format: {cat_arg}. Use 'label|prompt1|prompt2'")
            label = parts[0]
            prompts = parts[1:]
            categories[label] = prompts

    if len(categories) < 2:
        raise ValueError("Need at least 2 categories for classifier training")

    # Parse layers
    layers = None
    if getattr(args, "layers", None):
        layers = [int(x.strip()) for x in args.layers.split(Delimiters.LAYER_SEPARATOR)]

    config = ClassifierConfig(
        model=args.model,
        categories=categories,
        layers=layers,
        all_layers=getattr(args, "all_layers", False),
        layer_depth_ratio=(
            LayerDepthRatio.DECISION.value
            if layers is None and not getattr(args, "all_layers", False)
            else None
        ),
        max_iter=ProbeDefaults.LOGISTIC_MAX_ITER,
        random_seed=AnalysisDefaults.RANDOM_SEED,
        bar_width=DisplayDefaults.PROBABILITY_BAR_WIDTH,
    )

    # Run classifier training - all logic is in the service
    result = await ClassifierService.train_and_evaluate(config)

    # Print formatted result
    print(result.to_display())

    # Save if requested
    output_path = getattr(args, "output", None)
    if output_path:
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")


async def introspect_logit_lens(args: Namespace) -> None:
    """Run logit lens analysis on a prompt.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.logit_lens import LogitLensConfig, LogitLensService
    from .._constants import Delimiters

    # Parse layers
    layers = None
    if getattr(args, "layers", None):
        layers = [int(x.strip()) for x in args.layers.split(Delimiters.LAYER_SEPARATOR)]

    # Parse tracked tokens
    track_tokens = []
    if getattr(args, "track", None):
        track_tokens = args.track.split(Delimiters.LAYER_SEPARATOR)

    config = LogitLensConfig(
        model=args.model,
        prompt=args.prompt,
        layers=layers,
        layer_step=getattr(args, "layer_step", 4),
        top_k=getattr(args, "top_k", AnalysisDefaults.TOP_K),
        track_tokens=track_tokens,
    )

    # Run logit lens - all logic is in the service
    result = await LogitLensService.analyze(config)

    # Print formatted result
    print(result.to_display())


__all__ = [
    "introspect_classifier",
    "introspect_logit_lens",
]
