"""Probing analysis command handlers for introspection CLI.

This module provides thin CLI wrappers for probing analysis commands.
All business logic is delegated to the framework layer (introspection module).

IMPORTANT: CLI commands should NOT contain hardcoded sample data.
Use --calibration-file or framework-level dataset loaders instead.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import AnalysisDefaults, Delimiters, LayerDepthRatio, ProbeDefaults
from ._utils import (
    extract_arg,
    get_layer_depth_ratio,
    load_json_file,
    parse_layers,
    parse_prompts,
)

logger = logging.getLogger(__name__)


async def introspect_metacognitive(args: Namespace) -> None:
    """Detect metacognitive strategy switch at a specific layer.

    This is a thin wrapper that:
    1. Calls MetacognitiveService.analyze() which handles all logic
    2. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.probing import MetacognitiveConfig, MetacognitiveService

    # Parse prompts from CLI
    prompts = parse_prompts(args.prompts)
    decision_layer = extract_arg(args, "decision_layer")

    config = MetacognitiveConfig(
        model=args.model,
        prompts=prompts,
        decision_layer=decision_layer,
        layer_depth_ratio=get_layer_depth_ratio(decision_layer, LayerDepthRatio.LATE),
        top_k=extract_arg(args, "top_k", AnalysisDefaults.TOP_K_LAYER),
        use_raw=extract_arg(args, "raw", False),
    )

    # Run analysis - all logic is in the service
    result = await MetacognitiveService.analyze(config)

    # Print formatted result
    print(result.to_display())


async def introspect_uncertainty(args: Namespace) -> None:
    """Analyze model's uncertainty and calibration.

    IMPORTANT: Calibration prompts should come from framework datasets
    or user-provided files, not hardcoded in the CLI.

    Args:
        args: Parsed command-line arguments
    """
    from ....datasets import load_calibration_prompts
    from ....introspection.probing import UncertaintyConfig, UncertaintyService

    # Load calibration prompts from file or use framework defaults
    calibration_file = extract_arg(args, "calibration_file")
    if calibration_file:
        calibration_data = load_json_file(calibration_file)
        working_prompts = calibration_data.get("working", [])
        broken_prompts = calibration_data.get("broken", [])
    else:
        # Use framework-provided calibration datasets (no hardcoded data in CLI)
        calibration = load_calibration_prompts()
        working_prompts = calibration.working
        broken_prompts = calibration.broken

    layer = extract_arg(args, "layer")

    config = UncertaintyConfig(
        model=args.model,
        prompt=extract_arg(args, "prompt"),
        working_prompts=working_prompts,
        broken_prompts=broken_prompts,
        layer=layer,
        layer_depth_ratio=get_layer_depth_ratio(layer, LayerDepthRatio.DEEP),
    )

    # Run analysis - all logic is in the service
    result = await UncertaintyService.analyze(config)

    # Print formatted result
    print(result.to_display())


async def introspect_probe(args: Namespace) -> None:
    """Train and evaluate linear probes on model activations.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.probing import ProbeConfig, ProbeService

    # Load probe data from file or CLI args
    probe_file = extract_arg(args, "probe_file")
    if probe_file:
        probe_data = load_json_file(probe_file)
        positive_prompts = probe_data.get("positive", [])
        negative_prompts = probe_data.get("negative", [])
    else:
        # Require explicit data for probing (no defaults)
        positive_arg = extract_arg(args, "positive")
        negative_arg = extract_arg(args, "negative")

        if not positive_arg or not negative_arg:
            raise ValueError(
                "Probing requires either --probe-file or both --positive and --negative"
            )

        positive_prompts = positive_arg.split(Delimiters.PROMPT_SEPARATOR)
        negative_prompts = negative_arg.split(Delimiters.PROMPT_SEPARATOR)

    # Parse layers (using shared utility)
    layers = parse_layers(extract_arg(args, "layers"))

    config = ProbeConfig(
        model=args.model,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layers=layers,
        all_layers=extract_arg(args, "all_layers", False),
        ridge_alpha=ProbeDefaults.RIDGE_ALPHA,
        logistic_max_iter=ProbeDefaults.LOGISTIC_MAX_ITER,
        random_seed=AnalysisDefaults.RANDOM_SEED,
        cross_val_folds=AnalysisDefaults.CROSS_VAL_FOLDS,
    )

    # Run probing - all logic is in the service
    result = await ProbeService.train_and_evaluate(config)

    # Print formatted result
    print(result.to_display())

    # Save if requested
    output_path = extract_arg(args, "output")
    if output_path:
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")


__all__ = [
    "introspect_metacognitive",
    "introspect_probe",
    "introspect_uncertainty",
]
