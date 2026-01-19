"""
IR-Attention Routing: CoT as Circuit Invocation

Unified experiment testing whether CoT serves as a learned compiler
frontend that normalizes arbitrary input into circuit invocation formats.
"""

from .experiment import IRAttentionRoutingExperiment

__all__ = ["IRAttentionRoutingExperiment"]
