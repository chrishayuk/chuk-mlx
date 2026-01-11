"""Pydantic models for memory analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import MemorizationLevel
from .facts import FactNeighborhood


class RetrievalResult(BaseModel):
    """Result of retrieving a single fact."""

    query: str = Field(description="The query prompt")
    answer: str = Field(description="Expected answer")
    category: str = Field(default="", description="Fact category")
    predictions: list[dict] = Field(default_factory=list, description="Top-k predictions")
    neighborhood: FactNeighborhood = Field(default_factory=FactNeighborhood)
    memorization_level: MemorizationLevel = Field(
        default=MemorizationLevel.NOT_MEMORIZED,
        description="Classification of memorization strength",
    )

    @classmethod
    def classify_memorization(cls, rank: int | None, prob: float | None) -> MemorizationLevel:
        """Classify memorization level based on rank and probability."""
        if rank == 1 and prob is not None and prob > 0.1:
            return MemorizationLevel.MEMORIZED
        elif rank is not None and rank <= 5 and prob is not None and prob > 0.01:
            return MemorizationLevel.PARTIAL
        elif rank is not None and rank <= 15 and prob is not None and prob > 0.001:
            return MemorizationLevel.WEAK
        else:
            return MemorizationLevel.NOT_MEMORIZED


class AttractorNode(BaseModel):
    """A frequently co-activated answer."""

    answer: str = Field(description="The answer token")
    count: int = Field(description="Number of times it appears as a neighbor")
    avg_probability: float = Field(description="Average probability when appearing")


class MemoryStats(BaseModel):
    """Aggregated memory statistics."""

    top1_correct: int = Field(default=0, description="Number with correct answer at rank 1")
    top5_correct: int = Field(default=0, description="Number with correct answer in top 5")
    not_found: int = Field(default=0, description="Number where answer not in top-k")
    total: int = Field(default=0, description="Total number of facts")
    same_category_total: int = Field(default=0, description="Total same-category neighbors")
    same_category_alt_total: int = Field(default=0, description="Total alt-category neighbors")
    other_answers_total: int = Field(default=0, description="Total other-answer neighbors")
    non_answers_total: int = Field(default=0, description="Total non-answer neighbors")

    @property
    def top1_accuracy(self) -> float:
        """Compute top-1 accuracy."""
        return self.top1_correct / self.total if self.total > 0 else 0.0

    @property
    def top5_accuracy(self) -> float:
        """Compute top-5 accuracy."""
        return self.top5_correct / self.total if self.total > 0 else 0.0


class MemoryAnalysisResult(BaseModel):
    """Complete result of memory structure analysis."""

    model_id: str = Field(description="Model identifier")
    fact_type: str = Field(description="Type of facts analyzed")
    layer: int = Field(description="Layer analyzed")
    num_facts: int = Field(description="Number of facts analyzed")
    stats: MemoryStats = Field(default_factory=MemoryStats)
    attractors: list[AttractorNode] = Field(default_factory=list, description="Top attractor nodes")
    results: list[RetrievalResult] = Field(default_factory=list, description="Per-fact results")
    category_stats: dict[str, MemoryStats] = Field(
        default_factory=dict,
        description="Stats broken down by category",
    )
    asymmetries: list[dict] = Field(
        default_factory=list,
        description="Asymmetric pairs (A*B != B*A difficulty)",
    )
    row_bias_count: int = Field(default=0, description="Count favoring primary category")
    col_bias_count: int = Field(default=0, description="Count favoring alt category")
    neutral_count: int = Field(default=0, description="Count with no bias")
