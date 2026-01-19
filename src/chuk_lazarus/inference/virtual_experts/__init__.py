"""
Virtual Expert System for MoE Models.

This subpackage provides a plugin-based framework for adding virtual experts
to MoE models. Virtual experts are external tools (Python functions, APIs,
databases, etc.) that can be routed to by the MoE router.

Structure:
    - base.py: VirtualExpertPlugin base class and result types
    - registry.py: VirtualExpertRegistry for managing plugins
    - router.py: VirtualRouter that wraps MoE routers
    - wrapper.py: VirtualMoEWrapper main interface
    - plugins/: Built-in plugin implementations
        - math.py: MathExpertPlugin for arithmetic

Example - Using built-in math expert:
    >>> from chuk_lazarus.inference import VirtualMoEWrapper
    >>>
    >>> wrapper = VirtualMoEWrapper(model, tokenizer)
    >>> wrapper.calibrate()
    >>>
    >>> result = wrapper.solve("127 * 89 = ")
    >>> print(result.answer)  # "11303"

Example - Creating a custom expert:
    >>> from chuk_lazarus.inference import VirtualExpertPlugin
    >>>
    >>> class WikipediaExpert(VirtualExpertPlugin):
    ...     name = "wikipedia"
    ...     description = "Looks up facts on Wikipedia"
    ...
    ...     def can_handle(self, prompt: str) -> bool:
    ...         return "who is" in prompt.lower()
    ...
    ...     def execute(self, prompt: str) -> str | None:
    ...         return fetch_wikipedia(prompt)
    ...
    ...     def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
    ...         return ["Who is Einstein?"], ["Hello world"]
    >>>
    >>> wrapper.register_plugin(WikipediaExpert())
"""

from .base import (
    RoutingDecision,
    RoutingTrace,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertResult,
)
from .dense_wrapper import (
    VirtualDenseRouter,
    VirtualDenseWrapper,
    create_virtual_dense_wrapper,
)
from .plugins.math import MathExpertPlugin, SafeMathEvaluator
from .registry import VirtualExpertRegistry, get_default_registry
from .router import VirtualRouter
from .wrapper import VirtualMoEWrapper, create_virtual_expert_wrapper

__all__ = [
    # Base classes
    "VirtualExpertPlugin",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
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
    "MathExpertPlugin",
    "SafeMathEvaluator",
]
