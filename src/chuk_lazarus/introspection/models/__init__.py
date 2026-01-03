"""Pydantic models for introspection results.

This package contains structured, validated data models for all
introspection operations, replacing ad-hoc dictionaries throughout
the codebase.
"""

from .arithmetic import (
    ArithmeticStats,
    ArithmeticTestCase,
    ArithmeticTestResult,
    ArithmeticTestSuite,
    ParsedArithmeticPrompt,
)
from .circuit import (
    CapturedCircuit,
    CircuitComparisonResult,
    CircuitDirection,
    CircuitEntry,
    CircuitInvocationResult,
    CircuitTestResult,
)
from .facts import (
    CapitalFact,
    ElementFact,
    Fact,
    FactNeighborhood,
    FactSet,
    MathFact,
)
from .memory import (
    AttractorNode,
    MemoryAnalysisResult,
    MemoryStats,
    RetrievalResult,
)
from .patching import (
    CommutativityPair,
    CommutativityResult,
    PatchingLayerResult,
    PatchingResult,
)
from .probing import (
    ProbeLayerResult,
    ProbeResult,
    ProbeTopNeuron,
)
from .uncertainty import (
    CalibrationResult,
    MetacognitiveResult,
    UncertaintyResult,
)

__all__ = [
    # Arithmetic
    "ParsedArithmeticPrompt",
    "ArithmeticTestCase",
    "ArithmeticTestResult",
    "ArithmeticStats",
    "ArithmeticTestSuite",
    # Circuit
    "CircuitEntry",
    "CircuitDirection",
    "CapturedCircuit",
    "CircuitInvocationResult",
    "CircuitTestResult",
    "CircuitComparisonResult",
    # Facts
    "Fact",
    "MathFact",
    "CapitalFact",
    "ElementFact",
    "FactSet",
    "FactNeighborhood",
    # Memory
    "RetrievalResult",
    "AttractorNode",
    "MemoryStats",
    "MemoryAnalysisResult",
    # Patching
    "CommutativityPair",
    "CommutativityResult",
    "PatchingLayerResult",
    "PatchingResult",
    # Probing
    "ProbeLayerResult",
    "ProbeTopNeuron",
    "ProbeResult",
    # Uncertainty
    "MetacognitiveResult",
    "UncertaintyResult",
    "CalibrationResult",
]
