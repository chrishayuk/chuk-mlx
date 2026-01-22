"""Trace schema definitions for extended CoT format."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """Base class for trace steps."""
    pass


@dataclass
class InitStep(TraceStep):
    """Initialize a variable/entity."""
    var: str
    value: float


@dataclass
class TransferStep(TraceStep):
    """Transfer quantity between entities."""
    from_entity: str
    to_entity: str
    amount: float


@dataclass
class ConsumeStep(TraceStep):
    """Consume/remove quantity from entity."""
    entity: str
    amount: float


@dataclass
class ComputeStep(TraceStep):
    """Compute operation."""
    op: str  # add, sub, mul, div
    args: list
    var: str | None = None
    result: float = 0.0


@dataclass
class StateStep(TraceStep):
    """Assert current state."""
    state: dict[str, float]


@dataclass
class QueryStep(TraceStep):
    """Query for final answer."""
    var: str


@dataclass
class Trace:
    """Complete trace with expert and answer."""
    expert: str
    steps: list[dict] = field(default_factory=list)
    answer: Any = None

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        import yaml
        return yaml.dump({
            "expert": self.expert,
            "trace": self.steps,
            "answer": self.answer
        }, default_flow_style=None, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Trace":
        """Parse from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls(
            expert=data.get("expert", ""),
            steps=data.get("trace", []),
            answer=data.get("answer")
        )


# Expert types
EXPERT_TYPES = [
    "entity_track",
    "arithmetic",
    "rate_equation",
    "comparison",
    "allocation",
    "percentage",
]
