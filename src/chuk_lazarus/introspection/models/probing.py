"""Pydantic models for linear probing analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..enums import DirectionMethod


class ProbeLayerResult(BaseModel):
    """Result of probing at a single layer."""

    layer: int = Field(description="Layer index")
    accuracy: float = Field(description="Cross-validation accuracy")
    std: float = Field(default=0.0, description="Standard deviation of accuracy")


class ProbeTopNeuron(BaseModel):
    """A top neuron from probe weights."""

    index: int = Field(description="Neuron index")
    weight: float = Field(description="Weight in probe direction")


class ProbeResult(BaseModel):
    """Complete result of linear probing experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = Field(description="Model identifier")
    class_a_label: str = Field(description="Label for class A (positive)")
    class_b_label: str = Field(description="Label for class B (negative)")
    num_class_a: int = Field(description="Number of class A samples")
    num_class_b: int = Field(description="Number of class B samples")
    best_layer: int = Field(description="Layer with best accuracy")
    best_accuracy: float = Field(description="Best accuracy achieved")
    method: DirectionMethod = Field(description="Direction extraction method")
    layer_results: list[ProbeLayerResult] = Field(default_factory=list)
    direction: np.ndarray | None = Field(default=None, description="Extracted direction vector")
    direction_norm: float = Field(default=0.0, description="L2 norm of direction")
    top_neurons: list[ProbeTopNeuron] = Field(default_factory=list)
    separation: float = Field(default=0.0, description="Class separation in projection space")
    class_a_mean_projection: float = Field(default=0.0)
    class_b_mean_projection: float = Field(default=0.0)

    def save_direction(self, path: str | Path) -> None:
        """Save direction vector to npz file."""
        if self.direction is None:
            raise ValueError("No direction to save")

        np.savez(
            path,
            direction=self.direction,
            layer=self.best_layer,
            label_positive=self.class_a_label,
            label_negative=self.class_b_label,
            model_id=self.model_id,
            method=self.method.value,
            accuracy=self.best_accuracy,
            separation=self.separation,
            class_a_mean_projection=self.class_a_mean_projection,
            class_b_mean_projection=self.class_b_mean_projection,
        )

    @classmethod
    def load_direction(cls, path: str | Path) -> ProbeResult:
        """Load probe result from npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            model_id=str(data["model_id"]),
            class_a_label=str(data["label_positive"]),
            class_b_label=str(data["label_negative"]),
            num_class_a=0,
            num_class_b=0,
            best_layer=int(data["layer"]),
            best_accuracy=float(data["accuracy"]),
            method=DirectionMethod(str(data["method"])),
            direction=data["direction"],
            separation=float(data["separation"]),
            class_a_mean_projection=float(data["class_a_mean_projection"]),
            class_b_mean_projection=float(data["class_b_mean_projection"]),
        )
