"""
Base classes for virtual expert plugins.

This module defines the abstract base class for virtual experts and
the result types used throughout the virtual expert system.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class VirtualExpertPlugin(ABC):
    """
    Base class for virtual expert plugins.

    Subclass this to create custom virtual experts that can be routed to
    by the MoE router. Each plugin defines:

    - name: Unique identifier
    - description: Human-readable description
    - priority: Higher priority = checked first (default: 0)
    - can_handle(): Quick check if plugin might handle this prompt
    - execute(): Actually compute the result
    - get_calibration_prompts(): Examples for learning routing direction

    Example:
        >>> class TranslationExpert(VirtualExpertPlugin):
        ...     name = "translate"
        ...     description = "Translates text between languages"
        ...     priority = 5
        ...
        ...     def can_handle(self, prompt: str) -> bool:
        ...         return "translate" in prompt.lower()
        ...
        ...     def execute(self, prompt: str) -> str | None:
        ...         # Call translation API
        ...         return translate(prompt)
        ...
        ...     def get_calibration_prompts(self):
        ...         pos = ["Translate 'hello' to French", "What is 'cat' in Spanish?"]
        ...         neg = ["Hello!", "What is 2+2?", "Write a poem"]
        ...         return pos, neg
    """

    name: str = "base"
    description: str = "Base virtual expert"
    priority: int = 0

    @abstractmethod
    def can_handle(self, prompt: str) -> bool:
        """
        Check if this expert can handle the given prompt.

        This is used as a fast pre-filter before the router makes its decision.
        Return True if the prompt might be handled by this expert.

        This should be fast - it's called for every prompt to find potential
        handlers. The actual routing decision is made by the learned router.
        """
        pass

    @abstractmethod
    def execute(self, prompt: str) -> str | None:
        """
        Execute the expert's computation.

        Args:
            prompt: The input prompt

        Returns:
            The computed result as a string, or None if execution failed
        """
        pass

    @abstractmethod
    def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
        """
        Get prompts for calibrating this expert's routing.

        Returns:
            (positive_prompts, negative_prompts):
            - positive_prompts should route TO this expert
            - negative_prompts should NOT route to this expert

        These are used to learn a direction in activation space that
        separates prompts this expert should handle from those it shouldn't.
        """
        pass

    def validate_result(self, prompt: str, result: str) -> bool:
        """
        Validate the execution result.

        Override this to add custom validation logic. Default returns True
        if result is not None.

        Returns:
            True if the result is valid and should be used
        """
        return result is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"


class VirtualExpertApproach(str, Enum):
    """Which approach was used to generate the answer."""
    VIRTUAL_EXPERT = "virtual_expert"
    MODEL_DIRECT = "model_direct"


@dataclass
class VirtualExpertResult:
    """Result from virtual expert computation."""

    prompt: str
    answer: str
    correct_answer: float | int | None
    approach: VirtualExpertApproach
    used_virtual_expert: bool
    plugin_name: str | None = None
    routing_score: float | None = None
    virtual_expert_selected_count: int = 0
    total_tokens: int = 0
    is_correct: bool = False

    def __post_init__(self):
        """Check if answer matches expected value."""
        if self.correct_answer is not None:
            try:
                match = re.search(r'-?\d+(?:\.\d+)?', self.answer)
                if match:
                    answer_num = float(match.group())
                    self.is_correct = abs(answer_num - self.correct_answer) < 0.01
            except (ValueError, TypeError):
                pass


@dataclass
class VirtualExpertAnalysis:
    """Analysis of virtual expert behavior across multiple problems."""

    model_name: str
    total_problems: int
    correct_with_virtual: int
    correct_without_virtual: int
    times_virtual_used: int
    avg_routing_score: float
    plugins_used: dict[str, int] = field(default_factory=dict)
    results: list[VirtualExpertResult] = field(default_factory=list)

    @property
    def virtual_accuracy(self) -> float:
        """Accuracy when using virtual experts."""
        return self.correct_with_virtual / self.total_problems if self.total_problems > 0 else 0

    @property
    def model_accuracy(self) -> float:
        """Accuracy with model only (no virtual experts)."""
        return self.correct_without_virtual / self.total_problems if self.total_problems > 0 else 0

    @property
    def improvement(self) -> float:
        """Improvement from using virtual experts."""
        return self.virtual_accuracy - self.model_accuracy

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Virtual Expert Analysis",
            f"{'=' * 50}",
            f"Model: {self.model_name}",
            f"Problems: {self.total_problems}",
            f"",
            f"Model-only accuracy:   {self.model_accuracy:.1%}",
            f"With virtual expert:   {self.virtual_accuracy:.1%}",
            f"Improvement:           {self.improvement:+.1%}",
            f"",
            f"Virtual expert used:   {self.times_virtual_used}/{self.total_problems}",
        ]
        if self.plugins_used:
            lines.append(f"Plugins used:")
            for name, count in sorted(self.plugins_used.items(), key=lambda x: -x[1]):
                lines.append(f"  - {name}: {count}")
        return "\n".join(lines)
