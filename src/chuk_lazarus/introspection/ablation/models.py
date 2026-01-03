"""
Data models for ablation study results.

This module contains the Pydantic models for representing ablation
experiment results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class AblationResult(BaseModel):
    """Result of a single ablation experiment."""

    model_config = ConfigDict(frozen=True)

    layer: int = Field(description="Layer index that was ablated")
    component: str = Field(description="Component that was ablated")
    original_output: str = Field(description="Output before ablation")
    ablated_output: str = Field(description="Output after ablation")
    original_criterion: bool = Field(description="Criterion result before ablation")
    ablated_criterion: bool = Field(description="Criterion result after ablation")
    criterion_changed: bool = Field(description="Whether ablation changed criterion")
    output_coherent: bool = Field(default=True, description="Whether output is coherent")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LayerSweepResult(BaseModel):
    """Results from sweeping across layers."""

    model_config = ConfigDict(frozen=True)

    task_name: str = Field(description="Name of the task")
    criterion_name: str = Field(description="Name of the criterion")
    results: list[AblationResult] = Field(default_factory=list, description="Ablation results")

    @computed_field
    @property
    def causal_layers(self) -> list[int]:
        """Layers where ablation changed the criterion."""
        return [r.layer for r in self.results if r.criterion_changed]
