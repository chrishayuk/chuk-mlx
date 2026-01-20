"""
CSP Virtual Expert Experiment.

Tests whether Chain-of-Thought normalizes natural language constraint problems
into structured invocation formats that trigger a detectable "CSP Gate" at
early layers, enabling routing to an exact constraint solver.
"""

from .expert.csp_plugin import CSPVirtualExpertPlugin
from .extraction.csp_extractor import CSPSpec, extract_csp_spec

__all__ = [
    "CSPVirtualExpertPlugin",
    "CSPSpec",
    "extract_csp_spec",
]
