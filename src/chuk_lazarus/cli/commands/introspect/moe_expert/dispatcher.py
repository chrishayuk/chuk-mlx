"""Dispatcher for MoE expert CLI commands.

Routes action strings to their corresponding handlers using a dispatch table
instead of if/elif chains.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Callable

from .....introspection.moe import MoEAction

logger = logging.getLogger(__name__)

# Type alias for handler functions
HandlerFunc = Callable[[Namespace], None]


def _get_handlers() -> dict[MoEAction, HandlerFunc]:
    """Get the dispatch table mapping actions to handlers.

    We use lazy imports to avoid circular dependencies and speed up CLI startup.
    """
    from .handlers import (
        handle_ablate,
        handle_analyze,
        handle_attention_pattern,
        handle_attention_routing,
        handle_chat,
        handle_compare,
        handle_context_test,
        handle_context_window,
        handle_domain_test,
        handle_explore,
        handle_full_taxonomy,
        handle_heatmap,
        handle_moe_overlay_compress,
        handle_moe_overlay_compute,
        handle_moe_overlay_estimate,
        handle_moe_overlay_verify,
        handle_moe_type_analyze,
        handle_moe_type_compare,
        handle_token_routing,
        handle_trace,
        handle_weights,
    )

    return {
        MoEAction.ANALYZE: handle_analyze,
        MoEAction.ATTENTION_PATTERN: handle_attention_pattern,
        MoEAction.ATTENTION_ROUTING: handle_attention_routing,
        MoEAction.CHAT: handle_chat,
        MoEAction.COMPARE: handle_compare,
        MoEAction.ABLATE: handle_ablate,
        MoEAction.WEIGHTS: handle_weights,
        MoEAction.TRACE: handle_trace,
        MoEAction.CONTEXT_TEST: handle_context_test,
        MoEAction.CONTEXT_WINDOW: handle_context_window,
        MoEAction.FULL_TAXONOMY: handle_full_taxonomy,
        MoEAction.HEATMAP: handle_heatmap,
        MoEAction.DOMAIN_TEST: handle_domain_test,
        MoEAction.TOKEN_ROUTING: handle_token_routing,
        MoEAction.EXPLORE: handle_explore,
        MoEAction.MOE_TYPE_ANALYZE: handle_moe_type_analyze,
        MoEAction.MOE_TYPE_COMPARE: handle_moe_type_compare,
        MoEAction.MOE_OVERLAY_COMPUTE: handle_moe_overlay_compute,
        MoEAction.MOE_OVERLAY_VERIFY: handle_moe_overlay_verify,
        MoEAction.MOE_OVERLAY_ESTIMATE: handle_moe_overlay_estimate,
        MoEAction.MOE_OVERLAY_COMPRESS: handle_moe_overlay_compress,
    }


def dispatch(args: Namespace) -> None:
    """Dispatch to appropriate handler based on action.

    Args:
        args: Parsed command-line arguments. Must have 'action' attribute.

    Example:
        >>> args = Namespace(action="chat", model="openai/gpt-oss-20b", ...)
        >>> dispatch(args)  # Calls handle_chat(args)
    """
    action_str = getattr(args, "action", "chat")

    # Convert string to enum
    try:
        action = MoEAction(action_str)
    except ValueError:
        available = ", ".join(a.value for a in MoEAction)
        print(f"Unknown action: {action_str}")
        print(f"Available actions: {available}")
        return

    # Get handler from dispatch table
    handlers = _get_handlers()
    handler = handlers.get(action)

    if handler is None:
        print(f"Handler not implemented for action: {action.value}")
        return

    # Execute handler
    logger.debug(f"Dispatching to handler for action: {action.value}")
    handler(args)
