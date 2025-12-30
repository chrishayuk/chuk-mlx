"""
Data models for ablation study results.

This module contains the dataclasses for representing ablation
experiment results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    layer: int
    component: str
    original_output: str
    ablated_output: str
    original_criterion: bool
    ablated_criterion: bool
    criterion_changed: bool
    output_coherent: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class LayerSweepResult:
    """Results from sweeping across layers."""

    task_name: str
    criterion_name: str
    results: list[AblationResult]
    causal_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.causal_layers = [r.layer for r in self.results if r.criterion_changed]
