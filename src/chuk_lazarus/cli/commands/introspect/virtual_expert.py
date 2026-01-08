"""Virtual expert command handlers for introspection CLI.

This module provides thin CLI wrappers for virtual expert commands.
All business logic is delegated to the framework layer (introspection module).

IMPORTANT: CLI commands should NOT contain hardcoded test data.
Use --test-file or framework-level dataset loaders instead.
"""

from __future__ import annotations

import logging
from argparse import Namespace

logger = logging.getLogger(__name__)


async def introspect_virtual_expert(args: Namespace) -> None:
    """Virtual expert command dispatcher.

    This is a thin wrapper that:
    1. Converts CLI args to VirtualExpertConfig
    2. Calls VirtualExpertService methods which handle all logic
    3. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.virtual_expert import (
        VirtualExpertService,
        VirtualExpertConfig,
    )

    # Determine action
    action = getattr(args, "action", "solve")

    config = VirtualExpertConfig(
        model=args.model,
        layer=getattr(args, "layer", None),
        expert=getattr(args, "expert", None),
        prompt=getattr(args, "prompt", None),
    )

    if action == "analyze":
        # Load test data from file (no hardcoded data in CLI)
        test_file = getattr(args, "test_file", None)
        if test_file:
            import json
            with open(test_file) as f:
                test_data = json.load(f)
            config.test_categories = test_data
        else:
            # Use framework-provided test categories
            from ....datasets import load_expert_test_categories
            config.test_categories = load_expert_test_categories()

        result = await VirtualExpertService.analyze(config)

    elif action == "solve":
        if not config.prompt:
            raise ValueError("--prompt required for solve action")
        result = await VirtualExpertService.solve(config)

    elif action == "benchmark":
        # Load benchmark data from file (no hardcoded data in CLI)
        benchmark_file = getattr(args, "benchmark_file", None)
        if benchmark_file:
            import json
            with open(benchmark_file) as f:
                benchmark_data = json.load(f)
            problems = benchmark_data.get("problems", [])
        else:
            # Use framework-provided benchmark problems
            from ....datasets import load_expert_benchmark
            problems = load_expert_benchmark()

        config.benchmark_problems = problems
        result = await VirtualExpertService.benchmark(config)

    elif action == "compare":
        if not config.prompt:
            raise ValueError("--prompt required for compare action")
        result = await VirtualExpertService.compare(config)

    elif action == "interactive":
        result = await VirtualExpertService.interactive(config)

    else:
        raise ValueError(f"Unknown action: {action}")

    # Print formatted result
    print(result.to_display())


__all__ = [
    "introspect_virtual_expert",
]
