"""MoE Expert CLI subpackage.

Modular CLI commands for MoE expert routing manipulation and analysis.

This package provides:
- Dispatcher for routing to action handlers
- Individual handlers for each action
- Output formatters for structured display

Example:
    >>> from chuk_lazarus.cli.commands.introspect.moe_expert import dispatch
    >>> dispatch(args)  # Routes to appropriate handler based on args.action
"""

from .dispatcher import dispatch


def introspect_moe_expert(args):
    """MoE expert command entry point.

    Routes to the new modular dispatcher while maintaining backwards
    compatibility with the old flat module interface.

    Args:
        args: Parsed command-line arguments with 'action' attribute.
    """
    dispatch(args)


__all__ = ["dispatch", "introspect_moe_expert"]
