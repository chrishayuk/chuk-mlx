"""MoE Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import ExpertCategory, ExpertRole, MoEArchitecture


class MoELayerInfo(BaseModel):
    """Information about an MoE layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    num_experts: int = Field(ge=1)
    num_experts_per_tok: int = Field(ge=1)
    has_shared_expert: bool = False
    architecture: MoEArchitecture = MoEArchitecture.GENERIC
    router_type: str = "linear"
    uses_softmax: bool = True
    uses_sigmoid: bool = False


class RouterEntropy(BaseModel):
    """Router entropy analysis result."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    mean_entropy: float = Field(ge=0)
    max_entropy: float = Field(ge=0)
    normalized_entropy: float = Field(ge=0, le=1)
    per_position_entropy: tuple[float, ...] = Field(default_factory=tuple)


class ExpertUtilization(BaseModel):
    """Expert utilization statistics."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    num_experts: int = Field(ge=1)
    total_activations: int = Field(ge=0)
    expert_counts: tuple[int, ...] = Field(default_factory=tuple)
    expert_frequencies: tuple[float, ...] = Field(default_factory=tuple)
    load_balance_score: float = Field(ge=0, le=1)
    most_used_expert: int = Field(ge=0)
    least_used_expert: int = Field(ge=0)


class ExpertIdentity(BaseModel):
    """Identity/specialization of a single expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    primary_category: ExpertCategory
    secondary_categories: tuple[ExpertCategory, ...] = Field(default_factory=tuple)
    role: ExpertRole = ExpertRole.GENERALIST
    confidence: float = Field(ge=0, le=1)
    activation_rate: float = Field(ge=0, le=1)
    top_tokens: tuple[str, ...] = Field(default_factory=tuple)
    description: str = ""


class ExpertPair(BaseModel):
    """A pair of experts that frequently co-activate."""

    model_config = ConfigDict(frozen=True)

    expert_a: int = Field(ge=0)
    expert_b: int = Field(ge=0)
    coactivation_count: int = Field(ge=0)
    coactivation_rate: float = Field(ge=0, le=1)


class CoactivationAnalysis(BaseModel):
    """Analysis of expert co-activation patterns."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    total_activations: int = Field(ge=0)
    top_pairs: tuple[ExpertPair, ...] = Field(default_factory=tuple)
    specialist_pairs: tuple[ExpertPair, ...] = Field(default_factory=tuple)
    generalist_experts: tuple[int, ...] = Field(default_factory=tuple)


class ExpertAblationResult(BaseModel):
    """Result of ablating a single expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0)
    layer_idx: int = Field(ge=0)
    baseline_output: str
    ablated_output: str
    output_changed: bool
    would_have_activated: bool
    activation_count: int = Field(ge=0)


class CompressionPlan(BaseModel):
    """Plan for compressing experts."""

    model_config = ConfigDict(frozen=True)

    source_num_experts: int = Field(ge=1)
    target_num_experts: int = Field(ge=1)
    merge_groups: tuple[tuple[int, ...], ...] = Field(default_factory=tuple)
    estimated_quality_loss: float = Field(ge=0)
    estimated_size_reduction: float = Field(ge=0, le=1)
