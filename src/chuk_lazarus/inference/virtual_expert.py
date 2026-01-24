"""
Virtual Expert System - Compatibility Module.

This module re-exports all virtual expert classes from the virtual_experts
subpackage for backwards compatibility.

For new code, prefer importing from the subpackage directly:

    from chuk_lazarus.inference.virtual_experts import (
        VirtualMoEWrapper,
        VirtualExpertPlugin,
        MathExpertPlugin,
    )

Or from inference:

    from chuk_lazarus.inference import VirtualMoEWrapper
"""

# Re-export everything from the subpackage
from .virtual_experts import (
    InferenceResult,
    MathExpertPlugin,
    RoutingDecision,
    RoutingTrace,
    SafeMathEvaluator,
    VirtualDenseRouter,
    VirtualDenseWrapper,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertRegistry,
    VirtualExpertResult,
    VirtualMoEWrapper,
    VirtualRouter,
    create_virtual_dense_wrapper,
    create_virtual_expert_wrapper,
    get_default_registry,
)

__all__ = [
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "InferenceResult",
    # Routing trace (verbose output)
    "RoutingDecision",
    "RoutingTrace",
    # MoE
    "VirtualMoEWrapper",
    "VirtualRouter",
    "create_virtual_expert_wrapper",
    # Dense
    "VirtualDenseWrapper",
    "VirtualDenseRouter",
    "create_virtual_dense_wrapper",
    # Plugins
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "get_default_registry",
]
