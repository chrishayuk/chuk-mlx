"""Virtual expert command handlers for introspection CLI.

This module provides thin CLI wrappers for virtual expert commands.
All business logic is delegated to the framework layer (introspection module).

IMPORTANT: CLI commands should NOT contain hardcoded test data.
Use --test-file or framework-level dataset loaders instead.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ....introspection.virtual_expert import (
        VirtualExpertConfig,
        VirtualExpertServiceResult,
    )

    HandlerFunc = Callable[[VirtualExpertConfig], Awaitable[VirtualExpertServiceResult]]

logger = logging.getLogger(__name__)


async def introspect_virtual_expert(args: Namespace) -> None:
    """Virtual expert command dispatcher.

    This is a thin wrapper that:
    1. Converts CLI args to VirtualExpertConfig
    2. Uses dispatch table to route to VirtualExpertService methods
    3. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.virtual_expert import (
        VirtualExpertAction,
        VirtualExpertConfig,
        VirtualExpertService,
    )

    # Build dispatch table
    handlers: dict[VirtualExpertAction, HandlerFunc] = {
        VirtualExpertAction.ANALYZE: VirtualExpertService.analyze,
        VirtualExpertAction.SOLVE: VirtualExpertService.solve,
        VirtualExpertAction.BENCHMARK: VirtualExpertService.benchmark,
        VirtualExpertAction.COMPARE: VirtualExpertService.compare,
        VirtualExpertAction.INTERACTIVE: VirtualExpertService.interactive,
    }

    # Determine action
    action_str = getattr(args, "action", "solve")

    # Convert string to enum
    try:
        action = VirtualExpertAction(action_str)
    except ValueError:
        available = ", ".join(a.value for a in VirtualExpertAction)
        print(f"Unknown action: {action_str}")
        print(f"Available actions: {available}")
        return

    # Build config
    config = VirtualExpertConfig(
        model=args.model,
        layer=getattr(args, "layer", None),
        expert=getattr(args, "expert", None),
        prompt=getattr(args, "prompt", None),
        verbose=getattr(args, "verbose", False),
    )

    # Handle special config requirements per action
    if action == VirtualExpertAction.ANALYZE:
        test_file = getattr(args, "test_file", None)
        if test_file:
            import json

            with open(test_file) as f:
                test_data = json.load(f)
            config.test_categories = test_data
        else:
            from ....datasets import load_expert_test_categories

            config.test_categories = load_expert_test_categories()

    elif action == VirtualExpertAction.SOLVE:
        if not config.prompt:
            raise ValueError("--prompt required for solve action")

    elif action == VirtualExpertAction.BENCHMARK:
        benchmark_file = getattr(args, "benchmark_file", None)
        if benchmark_file:
            import json

            with open(benchmark_file) as f:
                benchmark_data = json.load(f)
            config.benchmark_problems = benchmark_data.get("problems", [])
        else:
            from ....datasets import load_expert_benchmark

            config.benchmark_problems = load_expert_benchmark()

    elif action == VirtualExpertAction.COMPARE:
        if not config.prompt:
            raise ValueError("--prompt required for compare action")

    # Get handler from dispatch table
    handler = handlers.get(action)
    if handler is None:
        print(f"Handler not implemented for action: {action.value}")
        return

    # Execute handler
    logger.debug(f"Dispatching to handler for action: {action.value}")
    result = await handler(config)

    # Print formatted result
    print(result.to_display())


__all__ = [
    "introspect_virtual_expert",
]
