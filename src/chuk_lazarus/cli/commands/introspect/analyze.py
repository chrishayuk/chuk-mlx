"""Analysis command handlers for chuk-lazarus introspection CLI.

This module provides thin CLI wrappers for introspection analysis commands.
All business logic is delegated to the framework layer (introspection module).
"""

from __future__ import annotations

import logging
from argparse import Namespace

logger = logging.getLogger(__name__)


async def introspect_analyze(args: Namespace) -> None:
    """Run logit lens analysis on a prompt.

    This is a thin wrapper that:
    1. Validates arguments
    2. Converts CLI args to AnalysisConfig
    3. Calls ModelAnalyzer.analyze() which handles all the logic
    4. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection import (
        AnalysisConfig,
        LayerStrategy,
    )
    from ....introspection.analyzer.service import AnalyzerService
    from .._constants import Delimiters

    # Validate input - either --prompt or --prefix required
    prompt = getattr(args, "prompt", None)
    prefix = getattr(args, "prefix", None)
    if not prompt and not prefix:
        raise ValueError("Either --prompt/-p or --prefix is required")

    # Build analysis config from CLI args
    custom_layers = None
    if getattr(args, "layers", None):
        custom_layers = [int(x.strip()) for x in args.layers.split(Delimiters.LAYER_SEPARATOR)]
        layer_strategy = LayerStrategy.CUSTOM
    elif getattr(args, "all_layers", False):
        layer_strategy = LayerStrategy.ALL
    else:
        layer_strategy = LayerStrategy(getattr(args, "layer_strategy", "evenly_spaced"))

    analysis_config = AnalysisConfig(
        layer_strategy=layer_strategy,
        layer_step=getattr(args, "layer_step", 4),
        top_k=getattr(args, "top_k", 10),
        track_tokens=getattr(args, "track", None) or [],
        custom_layers=custom_layers,
    )

    # Build service config from CLI args
    service_config = AnalyzerService.Config(
        model=args.model,
        adapter_path=getattr(args, "adapter", None),
        embedding_scale=getattr(args, "embedding_scale", None),
        use_raw=getattr(args, "raw", False),
        use_prefix_mode=prefix is not None,
        # Steering config
        steer_file=getattr(args, "steer", None),
        steer_neuron=getattr(args, "steer_neuron", None),
        steer_layer=getattr(args, "steer_layer", None),
        steer_strength=getattr(args, "strength", None),
        # Injection config
        inject_layer=getattr(args, "inject_layer", None),
        inject_token=getattr(args, "inject_token", None),
        inject_blend=getattr(args, "inject_blend", 1.0),
        # Compute override
        compute_override=getattr(args, "compute_override", "none"),
        compute_layer=getattr(args, "compute_layer", None),
        # Answer finding
        find_answer=getattr(args, "find_answer", None),
        no_find_answer=getattr(args, "no_find_answer", False),
        gen_tokens=getattr(args, "gen_tokens", 30),
        expected=getattr(args, "expected", None),
    )

    # Run analysis - all logic is in the service
    result = await AnalyzerService.analyze(
        prompt=prompt or prefix,
        analysis_config=analysis_config,
        service_config=service_config,
    )

    # Print formatted result
    print(result.to_display(top_k=args.top_k))

    # Export if requested
    output_path = getattr(args, "output", None)
    if output_path:
        result.save(output_path)
        print(f"\nResults saved to {output_path}")


async def introspect_compare(args: Namespace) -> None:
    """Compare two models' predictions using logit lens.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.analyzer.service import AnalyzerService

    result = await AnalyzerService.compare_models(
        model1=args.model1,
        model2=args.model2,
        prompt=args.prompt,
        top_k=getattr(args, "top_k", 10),
        track_tokens=getattr(args, "track", "").split(",") if getattr(args, "track", None) else [],
    )

    print(result.to_display())


async def introspect_hooks(args: Namespace) -> None:
    """Low-level hook demonstration.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.analyzer.service import AnalyzerService
    from .._constants import Delimiters

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(Delimiters.LAYER_SEPARATOR)]
    else:
        layers = list(range(0, 32, 4))

    result = await AnalyzerService.demonstrate_hooks(
        model=args.model,
        prompt=args.prompt,
        layers=layers,
        capture_attention=getattr(args, "capture_attention", False),
        last_only=getattr(args, "last_only", False),
        no_logit_lens=getattr(args, "no_logit_lens", False),
    )

    print(result.to_display())


__all__ = [
    "introspect_analyze",
    "introspect_compare",
    "introspect_hooks",
]
