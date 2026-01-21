"""
Trace Generators.

Convert ProblemSpec into verifiable Trace.
"""

from .base import TraceGenerator
from .entity import EntityTraceGenerator
from .arithmetic import ArithmeticTraceGenerator
from .comparison import ComparisonTraceGenerator
from .allocation import AllocationTraceGenerator
from .router import route_to_generator, generate_trace

__all__ = [
    "TraceGenerator",
    "EntityTraceGenerator",
    "ArithmeticTraceGenerator",
    "ComparisonTraceGenerator",
    "AllocationTraceGenerator",
    "route_to_generator",
    "generate_trace",
]
