"""
Base classes for virtual expert plugins.

Re-exports from chuk-virtual-expert and provides Lazarus-specific
inference tracking types for benchmarking and analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# Import core classes from chuk-virtual-expert
from chuk_virtual_expert import (
    VirtualExpert,
    VirtualExpertAction,
    VirtualExpertResult,
)

# Re-export for use across Lazarus
__all__ = [
    "VirtualExpert",
    "VirtualExpertAction",
    "VirtualExpertResult",
    "VirtualExpertApproach",
    "InferenceResult",
    "RoutingDecision",
    "RoutingTrace",
    "VirtualExpertAnalysis",
    # Backwards compatibility
    "VirtualExpertPlugin",
]

# Backwards compatibility alias
VirtualExpertPlugin = VirtualExpert


class VirtualExpertApproach(str, Enum):
    """Which approach was used to generate the answer."""

    VIRTUAL_EXPERT = "virtual_expert"
    MODEL_DIRECT = "model_direct"


@dataclass
class RoutingDecision:
    """A single routing decision at one layer."""

    layer: int
    confidence: float  # 0-1 routing score
    selected: bool  # Was virtual expert selected in top-k?
    task: str | None = None  # Detected task type (e.g., "multiply")
    attention_contribution: float | None = None  # % signal from attention


@dataclass
class RoutingTrace:
    """Complete routing trace across layers."""

    decisions: list[RoutingDecision] = field(default_factory=list)
    hijack_layer: int | None = None  # Layer where hijack decision was made
    hijack_confidence: float | None = None

    def add_decision(
        self,
        layer: int,
        confidence: float,
        selected: bool,
        task: str | None = None,
        attention_contribution: float | None = None,
    ) -> None:
        """Add a routing decision."""
        decision = RoutingDecision(
            layer=layer,
            confidence=confidence,
            selected=selected,
            task=task,
            attention_contribution=attention_contribution,
        )
        self.decisions.append(decision)
        # Track hijack point (first layer where selected with high confidence)
        if selected and confidence > 0.5 and self.hijack_layer is None:
            self.hijack_layer = layer
            self.hijack_confidence = confidence

    def format_verbose(self) -> str:
        """Format trace for verbose CLI output."""
        lines = []
        for d in self.decisions:
            marker = "→" if d.selected else " "
            task_str = f" Task: {d.task}" if d.task else ""
            conf_str = f"(confidence: {d.confidence:.2f})"
            attn_str = ""
            if d.attention_contribution is not None:
                attn_str = f" [Attention: {d.attention_contribution:.0%}]"
            lines.append(f"  {marker} [L{d.layer:02d}]{task_str} {conf_str}{attn_str}")

        if self.hijack_layer is not None:
            lines.append(f"\n  [Hijack] Layer {self.hijack_layer} "
                        f"(confidence: {self.hijack_confidence:.2f}) → virtual calculator")
        return "\n".join(lines)


@dataclass
class InferenceResult:
    """Result from Lazarus inference with virtual expert routing."""

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
    routing_trace: RoutingTrace | None = None

    def __post_init__(self):
        """Check if answer matches expected value."""
        if self.correct_answer is not None:
            try:
                match = re.search(r"-?\d+(?:\.\d+)?", self.answer)
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
    results: list[InferenceResult] = field(default_factory=list)

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
            "Virtual Expert Analysis",
            f"{'=' * 50}",
            f"Model: {self.model_name}",
            f"Problems: {self.total_problems}",
            "",
            f"Model-only accuracy:   {self.model_accuracy:.1%}",
            f"With virtual expert:   {self.virtual_accuracy:.1%}",
            f"Improvement:           {self.improvement:+.1%}",
            "",
            f"Virtual expert used:   {self.times_virtual_used}/{self.total_problems}",
        ]
        if self.plugins_used:
            lines.append("Plugins used:")
            for name, count in sorted(self.plugins_used.items(), key=lambda x: -x[1]):
                lines.append(f"  - {name}: {count}")
        return "\n".join(lines)
