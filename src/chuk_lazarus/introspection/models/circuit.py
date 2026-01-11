"""Pydantic models for circuit analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..enums import InvocationMethod, TestStatus


class CircuitEntry(BaseModel):
    """A single entry in a captured circuit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str = Field(description="The prompt used")
    operand_a: int | None = Field(default=None, description="First operand if arithmetic")
    operand_b: int | None = Field(default=None, description="Second operand if arithmetic")
    operator: str | None = Field(default=None, description="Operator if arithmetic")
    result: int | None = Field(default=None, description="Expected result")
    activation: np.ndarray | None = Field(default=None, description="Captured activation vector")


class CircuitDirection(BaseModel):
    """A direction vector extracted from a circuit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: np.ndarray = Field(description="The direction vector")
    norm: float = Field(description="L2 norm of direction")
    r2_score: float = Field(default=0.0, description="R2 score if from regression")
    mae: float = Field(default=0.0, description="Mean absolute error if from regression")
    scale: float = Field(default=1.0, description="Scale factor for result prediction")
    intercept: float = Field(default=0.0, description="Intercept for result prediction")


class CapturedCircuit(BaseModel):
    """A complete captured circuit with activations and optional direction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = Field(description="Model identifier")
    layer: int = Field(description="Layer where activations were captured")
    entries: list[CircuitEntry] = Field(default_factory=list, description="Circuit entries")
    direction: CircuitDirection | None = Field(
        default=None, description="Extracted direction if available"
    )
    activations: np.ndarray | None = Field(default=None, description="Stacked activation matrix")

    @property
    def num_entries(self) -> int:
        """Number of entries in the circuit."""
        return len(self.entries)

    @property
    def has_direction(self) -> bool:
        """Check if circuit has an extracted direction."""
        return self.direction is not None

    def save(self, path: str | Path) -> None:
        """Save circuit to npz file."""
        path = Path(path)
        save_data: dict[str, Any] = {
            "model_id": self.model_id,
            "layer": self.layer,
            "prompts": np.array([e.prompt for e in self.entries]),
            "operands_a": np.array([e.operand_a for e in self.entries]),
            "operands_b": np.array([e.operand_b for e in self.entries]),
            "operators": np.array([e.operator for e in self.entries]),
            "results": np.array([e.result for e in self.entries]),
        }

        if self.activations is not None:
            save_data["activations"] = self.activations
        elif self.entries and self.entries[0].activation is not None:
            save_data["activations"] = np.array([e.activation for e in self.entries])

        if self.direction is not None:
            save_data["direction"] = self.direction.direction
            save_data["direction_stats"] = {
                "norm": self.direction.norm,
                "r2": self.direction.r2_score,
                "mae": self.direction.mae,
                "scale": self.direction.scale,
                "intercept": self.direction.intercept,
            }

        np.savez(path, **save_data)

    @classmethod
    def load(cls, path: str | Path) -> CapturedCircuit:
        """Load circuit from npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        entries = []
        prompts = list(data["prompts"])
        operands_a = list(data["operands_a"])
        operands_b = list(data["operands_b"])
        operators = list(data["operators"])
        results = list(data["results"])
        activations = data["activations"] if "activations" in data else None

        for i, prompt in enumerate(prompts):
            entry = CircuitEntry(
                prompt=str(prompt),
                operand_a=int(operands_a[i]) if operands_a[i] is not None else None,
                operand_b=int(operands_b[i]) if operands_b[i] is not None else None,
                operator=str(operators[i]) if operators[i] is not None else None,
                result=int(results[i]) if results[i] is not None else None,
                activation=activations[i] if activations is not None else None,
            )
            entries.append(entry)

        direction = None
        if "direction" in data:
            stats = data["direction_stats"].item() if "direction_stats" in data else {}
            direction = CircuitDirection(
                direction=data["direction"],
                norm=float(stats.get("norm", np.linalg.norm(data["direction"]))),
                r2_score=float(stats.get("r2", 0.0)),
                mae=float(stats.get("mae", 0.0)),
                scale=float(stats.get("scale", 1.0)),
                intercept=float(stats.get("intercept", 0.0)),
            )

        return cls(
            model_id=str(data["model_id"]),
            layer=int(data["layer"]),
            entries=entries,
            direction=direction,
            activations=activations,
        )


class CircuitInvocationResult(BaseModel):
    """Result of invoking a circuit with new operands."""

    operand_a: int = Field(description="First operand")
    operand_b: int = Field(description="Second operand")
    predicted: float = Field(description="Predicted result")
    true_result: int | None = Field(default=None, description="True result if known")
    error: float | None = Field(default=None, description="Prediction error")
    method: InvocationMethod = Field(description="Method used for invocation")


class CircuitTestResult(BaseModel):
    """Result of testing circuit generalization."""

    prompt: str = Field(description="Test prompt")
    true_result: float = Field(description="True result")
    predicted: float = Field(description="Predicted result")
    error: float = Field(description="Prediction error")
    in_training: bool = Field(default=False, description="Whether this prompt was in training data")
    status: TestStatus = Field(default=TestStatus.NOVEL, description="Test status")


class CircuitComparisonResult(BaseModel):
    """Result of comparing multiple circuits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    circuit_names: list[str] = Field(description="Names of compared circuits")
    similarity_matrix: np.ndarray = Field(description="Pairwise cosine similarities")
    angles: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Pairwise angles in degrees"
    )
    shared_neurons: list[tuple[int, list[tuple[str, float]]]] = Field(
        default_factory=list,
        description="Neurons that appear in multiple circuits with their weights",
    )
