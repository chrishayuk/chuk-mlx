"""
CSP-CoT Schema.

Defines the trace format for verifiable reasoning.
"""

from .trace import Step, Trace, State, Action
from .problem import ProblemSpec, Entity, Operation, Query, ProblemType
from .verifier import TraceVerifier, VerificationResult

__all__ = [
    "Step",
    "Trace",
    "State",
    "Action",
    "ProblemSpec",
    "Entity",
    "Operation",
    "Query",
    "ProblemType",
    "TraceVerifier",
    "VerificationResult",
]
