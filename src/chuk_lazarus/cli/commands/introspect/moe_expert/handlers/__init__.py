"""Handler functions for MoE expert CLI actions.

Each handler is a thin wrapper that:
1. Validates arguments
2. Calls the framework layer (ExpertRouter)
3. Formats and prints output

No business logic should be in handlers - only argument validation,
framework calls, and output formatting.
"""

from .ablate import handle_ablate
from .analyze import handle_analyze
from .chat import handle_chat
from .collaboration import handle_collaboration
from .compare import handle_compare
from .context_test import handle_context_test
from .control_tokens import handle_control_tokens
from .divergence import handle_divergence
from .entropy import handle_entropy
from .full_taxonomy import handle_full_taxonomy
from .interactive import handle_interactive
from .layer_sweep import handle_layer_sweep
from .pairs import handle_pairs
from .pattern_discovery import handle_pattern_discovery
from .role import handle_role
from .router_probe import handle_router_probe
from .tokenizer import handle_tokenizer
from .topk import handle_topk
from .trace import handle_trace
from .vocab_map import handle_vocab_map
from .weights import handle_weights

__all__ = [
    "handle_ablate",
    "handle_analyze",
    "handle_chat",
    "handle_collaboration",
    "handle_compare",
    "handle_context_test",
    "handle_control_tokens",
    "handle_divergence",
    "handle_entropy",
    "handle_full_taxonomy",
    "handle_interactive",
    "handle_layer_sweep",
    "handle_pairs",
    "handle_pattern_discovery",
    "handle_role",
    "handle_router_probe",
    "handle_tokenizer",
    "handle_topk",
    "handle_trace",
    "handle_vocab_map",
    "handle_weights",
]
