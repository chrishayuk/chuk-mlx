"""Pydantic models for uncertainty and metacognitive analysis."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..enums import ComputeStrategy, ConfidenceLevel


class MetacognitiveResult(BaseModel):
    """Result of metacognitive strategy detection for a single problem."""

    problem: str = Field(description="The problem prompt")
    expected: str | None = Field(default=None, description="Expected answer")
    generated: str = Field(default="", description="Generated output (first 50 chars)")
    decision_layer: int = Field(description="Layer where strategy was detected")
    decision_token: str = Field(description="Top token at decision layer")
    decision_prob: float = Field(description="Probability of decision token")
    strategy: ComputeStrategy = Field(description="Detected strategy")
    is_digit: bool = Field(default=False, description="Whether decision token is a digit")
    correct_start: bool = Field(default=False, description="Whether digit matches expected answer start")
    final_token: str = Field(default="", description="Final layer top token")
    final_prob: float = Field(default=0.0, description="Final layer top probability")


class MetacognitiveAnalysis(BaseModel):
    """Complete metacognitive analysis results."""

    model_id: str = Field(description="Model identifier")
    decision_layer: int = Field(description="Layer used for detection")
    total_problems: int = Field(description="Total problems analyzed")
    direct_count: int = Field(default=0, description="Problems using direct computation")
    cot_count: int = Field(default=0, description="Problems using chain-of-thought")
    results: list[MetacognitiveResult] = Field(default_factory=list)

    @property
    def direct_ratio(self) -> float:
        """Ratio of problems using direct computation."""
        return self.direct_count / self.total_problems if self.total_problems > 0 else 0.0

    @property
    def direct_accuracy(self) -> float:
        """Accuracy among direct computation answers."""
        direct = [r for r in self.results if r.strategy == ComputeStrategy.DIRECT]
        if not direct:
            return 0.0
        correct = sum(1 for r in direct if r.correct_start)
        return correct / len(direct)


class UncertaintyResult(BaseModel):
    """Result of uncertainty detection for a single prompt."""

    prompt: str = Field(description="The prompt")
    score: float = Field(description="Uncertainty score (positive = confident)")
    prediction: ConfidenceLevel = Field(description="Predicted confidence level")
    dist_to_compute: float = Field(description="Distance to compute center")
    dist_to_refusal: float = Field(description="Distance to refusal center")


class CalibrationResult(BaseModel):
    """Calibration data for uncertainty detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = Field(description="Model identifier")
    detection_layer: int = Field(description="Layer used for detection")
    compute_center: np.ndarray = Field(description="Center of working prompts")
    refusal_center: np.ndarray = Field(description="Center of broken prompts")
    separation: float = Field(description="Distance between centers")
    working_prompts: list[str] = Field(default_factory=list)
    broken_prompts: list[str] = Field(default_factory=list)


class UncertaintyAnalysis(BaseModel):
    """Complete uncertainty analysis results."""

    model_id: str = Field(description="Model identifier")
    detection_layer: int = Field(description="Layer used for detection")
    separation: float = Field(description="Compute-refusal separation")
    results: list[UncertaintyResult] = Field(default_factory=list)

    @property
    def confident_count(self) -> int:
        """Number of confident predictions."""
        return sum(1 for r in self.results if r.prediction == ConfidenceLevel.CONFIDENT)

    @property
    def uncertain_count(self) -> int:
        """Number of uncertain predictions."""
        return sum(1 for r in self.results if r.prediction == ConfidenceLevel.UNCERTAIN)
