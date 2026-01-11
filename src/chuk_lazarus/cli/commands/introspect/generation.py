"""Generation analysis command handlers for introspection CLI.

This module provides thin CLI wrappers for generation analysis commands.
All business logic is delegated to the framework layer (introspection module).
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import AnalysisDefaults

logger = logging.getLogger(__name__)


async def introspect_generate(args: Namespace) -> None:
    """Generate with logit lens analysis.

    This is a thin wrapper that:
    1. Converts CLI args to GenerationConfig
    2. Calls GenerationService.generate() which handles all logic
    3. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.generation import GenerationConfig, GenerationService

    # Get prompt - CLI may use --prompts or --prompt
    prompt = getattr(args, "prompts", None) or getattr(args, "prompt", "")

    config = GenerationConfig(
        model=args.model,
        prompt=prompt,
        max_tokens=getattr(args, "max_tokens", AnalysisDefaults.GEN_TOKENS),
        temperature=getattr(args, "temperature", 0.0),
        top_k=getattr(args, "top_k", AnalysisDefaults.TOP_K),
        layer_step=getattr(args, "layer_step", 4),
        track_tokens=getattr(args, "track", None) or [],
        chat_template_file=getattr(args, "chat_template", None),
        use_raw=getattr(args, "raw", False),
        expected_answer=getattr(args, "expected", None),
        find_answer=getattr(args, "find_answer", None),
        no_find_answer=getattr(args, "no_find_answer", False),
    )

    # Run generation - all logic is in the service
    result = await GenerationService.generate(config)

    # Print formatted result
    print(result.to_display())

    # Save if requested
    output_path = getattr(args, "output", None)
    if output_path:
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")


async def introspect_logit_evolution(args: Namespace) -> None:
    """Show how logits evolve across layers for specific tokens.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.generation import LogitEvolutionConfig, LogitEvolutionService
    from .._constants import Delimiters

    # Parse tracked tokens
    track_tokens = []
    if getattr(args, "track", None):
        track_tokens = args.track.split(Delimiters.LAYER_SEPARATOR)

    config = LogitEvolutionConfig(
        model=args.model,
        prompt=args.prompt,
        track_tokens=track_tokens,
        layer_step=getattr(args, "layer_step", 4),
        top_k=getattr(args, "top_k", AnalysisDefaults.TOP_K),
    )

    # Run evolution analysis - all logic is in the service
    result = await LogitEvolutionService.analyze(config)

    # Print formatted result
    print(result.to_display())


__all__ = [
    "introspect_generate",
    "introspect_logit_evolution",
]
