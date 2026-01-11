"""Activation clustering command handlers for introspection CLI.

This module provides thin CLI wrappers for clustering analysis commands.
All business logic is delegated to the framework layer (introspection module).
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import DisplayDefaults, LayerDepthRatio
from ._utils import extract_arg, get_layer_depth_ratio, parse_layers, parse_prompts

logger = logging.getLogger(__name__)


async def introspect_activation_cluster(args: Namespace) -> None:
    """Visualize activation clusters using PCA.

    This is a thin wrapper that:
    1. Converts CLI args to ClusteringConfig
    2. Calls ClusteringService.analyze() which handles all logic
    3. Formats and prints results

    Supports two syntaxes:
    1. Legacy two-class: --class-a "prompts" --class-b "prompts" --label-a X --label-b Y
    2. Multi-class: --prompts "p1|p2|p3" --label L1 --prompts "p4|p5" --label L2 ...

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.clustering import ClusteringConfig, ClusteringService

    # Parse prompts with labels - support both legacy and new syntax
    prompts: list[str] = []
    labels: list[str] = []

    # Check for new multi-class syntax
    prompt_groups = extract_arg(args, "prompt_groups")
    labels_arg = extract_arg(args, "labels")
    if prompt_groups and labels_arg:
        if len(prompt_groups) != len(labels_arg):
            raise ValueError(
                f"Number of --prompts ({len(prompt_groups)}) must match "
                f"number of --label ({len(labels_arg)})"
            )

        for prompt_group, label in zip(prompt_groups, labels_arg):
            group_prompts = parse_prompts(prompt_group)
            prompts.extend(group_prompts)
            labels.extend([label] * len(group_prompts))

    # Fall back to legacy two-class syntax
    else:
        class_a = extract_arg(args, "class_a")
        class_b = extract_arg(args, "class_b")

        if not class_a and not class_b:
            raise ValueError("Must provide either --prompts/--label pairs or --class-a/--class-b")

        if class_a:
            class_a_prompts = parse_prompts(class_a)
            prompts.extend(class_a_prompts)
            labels.extend([extract_arg(args, "label_a", "A")] * len(class_a_prompts))

        if class_b:
            class_b_prompts = parse_prompts(class_b)
            prompts.extend(class_b_prompts)
            labels.extend([extract_arg(args, "label_b", "B")] * len(class_b_prompts))

    if len(prompts) < 2:
        raise ValueError("Need at least 2 prompts for clustering")

    # Parse layers (using shared utility)
    layer_arg = extract_arg(args, "layer")
    target_layers = parse_layers(str(layer_arg)) if layer_arg is not None else None

    config = ClusteringConfig(
        model=args.model,
        prompts=prompts,
        labels=labels,
        target_layers=target_layers,
        layer_depth_ratio=get_layer_depth_ratio(
            target_layers[0] if target_layers else None,
            LayerDepthRatio.MIDDLE,
        ),
        grid_width=DisplayDefaults.ASCII_GRID_WIDTH,
        grid_height=DisplayDefaults.ASCII_GRID_HEIGHT,
        save_plot=extract_arg(args, "save_plot"),
    )

    # Run clustering - all logic is in the service
    result = await ClusteringService.analyze(config)

    # Print formatted result
    print(result.to_display())


__all__ = [
    "introspect_activation_cluster",
]
