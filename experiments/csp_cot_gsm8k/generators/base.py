"""
Base Trace Generator.

Abstract interface for all trace generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema.trace import Trace
from ..schema.problem import ProblemSpec, ProblemType


class TraceGenerator(ABC):
    """
    Abstract base class for trace generators.

    Each generator handles a specific problem type and converts
    a ProblemSpec into a verifiable Trace.
    """

    @property
    @abstractmethod
    def supported_types(self) -> list[ProblemType]:
        """Return list of problem types this generator handles."""
        pass

    @abstractmethod
    def generate(self, spec: ProblemSpec) -> Trace:
        """
        Generate a verifiable trace from a problem specification.

        Args:
            spec: The structured problem specification

        Returns:
            A Trace that can be verified by replay
        """
        pass

    def can_handle(self, spec: ProblemSpec) -> bool:
        """Check if this generator can handle the given spec."""
        return spec.problem_type in self.supported_types
