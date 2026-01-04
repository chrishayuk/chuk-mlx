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
        handle_chat,
        handle_collaboration,
        handle_compare,
        handle_compression,
        handle_context_test,
        handle_control_tokens,
        handle_divergence,
        handle_entropy,
        handle_full_taxonomy,
        handle_heatmap,
        handle_interactive,
        handle_layer_sweep,
        handle_pairs,
        handle_pattern_discovery,
        handle_pipeline,
        handle_role,
        handle_router_probe,
        handle_tokenizer,
        handle_topk,
        handle_trace,
        handle_vocab_contrib,
        handle_vocab_map,
        handle_weights,
    )

    return {
        MoEAction.ANALYZE: handle_analyze,
        MoEAction.CHAT: handle_chat,
        MoEAction.COMPARE: handle_compare,
        MoEAction.ABLATE: handle_ablate,
        MoEAction.TOPK: handle_topk,
        MoEAction.COLLABORATION: handle_collaboration,
        MoEAction.PAIRS: handle_pairs,
        MoEAction.INTERACTIVE: handle_interactive,
        MoEAction.WEIGHTS: handle_weights,
        MoEAction.TOKENIZER: handle_tokenizer,
        MoEAction.CONTROL_TOKENS: handle_control_tokens,
        MoEAction.TRACE: handle_trace,
        MoEAction.ENTROPY: handle_entropy,
        MoEAction.DIVERGENCE: handle_divergence,
        MoEAction.ROLE: handle_role,
        MoEAction.CONTEXT_TEST: handle_context_test,
        MoEAction.VOCAB_MAP: handle_vocab_map,
        MoEAction.ROUTER_PROBE: handle_router_probe,
        MoEAction.PATTERN_DISCOVERY: handle_pattern_discovery,
        MoEAction.FULL_TAXONOMY: handle_full_taxonomy,
        MoEAction.LAYER_SWEEP: handle_layer_sweep,
        MoEAction.HEATMAP: handle_heatmap,
        MoEAction.PIPELINE: handle_pipeline,
        MoEAction.VOCAB_CONTRIB: handle_vocab_contrib,
        MoEAction.COMPRESSION: handle_compression,
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
