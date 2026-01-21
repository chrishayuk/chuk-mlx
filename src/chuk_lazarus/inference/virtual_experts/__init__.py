"""
Virtual Expert System for MoE Models.

This subpackage provides a plugin-based framework for adding virtual experts
to MoE models. Virtual experts are external tools (Python functions, APIs,
databases, etc.) that can be routed to by the MoE router.

Virtual expert base class is provided by chuk-virtual-expert package.

Structure:
    - base.py: Re-exports VirtualExpert from chuk-virtual-expert + inference types
    - registry.py: VirtualExpertRegistry for managing plugins
    - router.py: VirtualRouter that wraps MoE routers
    - wrapper.py: VirtualMoEWrapper main interface
    - plugins/: Built-in plugin implementations
        - math.py: MathExpert for arithmetic

Example - Using built-in math expert:
    >>> from chuk_lazarus.inference import VirtualMoEWrapper
    >>>
    >>> wrapper = VirtualMoEWrapper(model, tokenizer)
    >>> wrapper.calibrate()
    >>>
    >>> result = wrapper.solve("127 * 89 = ")
    >>> print(result.answer)  # "11303"

Example - Using TimeExpert from chuk-virtual-expert-time:
    >>> from chuk_virtual_expert_time import TimeExpert
    >>> from chuk_virtual_expert import adapt_expert
    >>>
    >>> expert = TimeExpert()
    >>> adapter = adapt_expert(expert)
    >>> wrapper.register_plugin(adapter)
"""

# Re-export VirtualExpert from chuk-virtual-expert
from chuk_virtual_expert import VirtualExpert

from .base import (
    RoutingDecision,
    RoutingTrace,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,  # Alias for VirtualExpert (backwards compat)
    VirtualExpertResult,
)
from .cot_rewriter import (
    CoTRewriter,
    DirectCoTRewriter,
    FewShotCoTRewriter,
    VirtualExpertAction,
)
from .dense_wrapper import (
    VirtualDenseRouter,
    VirtualDenseWrapper,
    create_virtual_dense_wrapper,
)
from .plugins.math import MathExpert, MathExpertPlugin, SafeMathEvaluator
from .registry import VirtualExpertRegistry, get_default_registry
from .router import VirtualRouter
from .wrapper import VirtualMoEWrapper, create_virtual_expert_wrapper

__all__ = [
    # Base classes (from chuk-virtual-expert)
    "VirtualExpert",
    "VirtualExpertPlugin",  # Alias for backwards compat
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    # CoT rewriting
    "VirtualExpertAction",
    "CoTRewriter",
    "FewShotCoTRewriter",
    "DirectCoTRewriter",
    # Routing trace (verbose output)
    "RoutingDecision",
    "RoutingTrace",
    # Registry
    "VirtualExpertRegistry",
    "get_default_registry",
    # Router (MoE)
    "VirtualRouter",
    # Wrapper (MoE)
    "VirtualMoEWrapper",
    "create_virtual_expert_wrapper",
    # Router (Dense)
    "VirtualDenseRouter",
    # Wrapper (Dense)
    "VirtualDenseWrapper",
    "create_virtual_dense_wrapper",
    # Built-in plugins
    "MathExpert",
    "MathExpertPlugin",  # Alias for backwards compat
    "SafeMathEvaluator",
]
