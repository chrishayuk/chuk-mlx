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
from .attention_pattern import handle_attention_pattern
from .attention_prediction import handle_attention_prediction
from .attention_routing import handle_attention_routing
from .chat import handle_chat
from .cold_experts import handle_cold_experts
from .compare import handle_compare
from .context_attention_routing import handle_context_attention_routing
from .context_test import handle_context_test
from .context_window import handle_context_window
from .domain_test import handle_domain_test
from .expert_circuits import handle_expert_circuits
from .expert_interference import handle_expert_interference
from .expert_merging import handle_expert_merging
from .explore import handle_explore
from .full_taxonomy import handle_full_taxonomy
from .generation_dynamics import handle_generation_dynamics
from .heatmap import handle_heatmap
from .moe_overlay_compress import handle_moe_overlay_compress
from .moe_overlay_compute import handle_moe_overlay_compute
from .moe_overlay_estimate import handle_moe_overlay_estimate
from .moe_overlay_verify import handle_moe_overlay_verify
from .moe_type_analyze import handle_moe_type_analyze
from .moe_type_compare import handle_moe_type_compare
from .routing_manipulation import handle_routing_manipulation
from .task_prediction import handle_task_prediction
from .token_routing import handle_token_routing
from .trace import handle_trace
from .weights import handle_weights

__all__ = [
    "handle_ablate",
    "handle_analyze",
    "handle_attention_pattern",
    "handle_attention_prediction",
    "handle_attention_routing",
    "handle_chat",
    "handle_cold_experts",
    "handle_compare",
    "handle_context_attention_routing",
    "handle_context_test",
    "handle_context_window",
    "handle_domain_test",
    "handle_expert_circuits",
    "handle_expert_interference",
    "handle_expert_merging",
    "handle_explore",
    "handle_full_taxonomy",
    "handle_generation_dynamics",
    "handle_heatmap",
    "handle_moe_overlay_compress",
    "handle_moe_overlay_compute",
    "handle_moe_overlay_estimate",
    "handle_moe_overlay_verify",
    "handle_moe_type_analyze",
    "handle_moe_type_compare",
    "handle_routing_manipulation",
    "handle_task_prediction",
    "handle_token_routing",
    "handle_trace",
    "handle_weights",
]
