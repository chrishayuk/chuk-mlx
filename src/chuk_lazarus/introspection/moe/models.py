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


# =============================================================================
# MoE Model Information
# =============================================================================


class MoEModelInfo(BaseModel):
    """Complete MoE model information."""

    model_config = ConfigDict(frozen=True)

    moe_layers: tuple[int, ...] = Field(default_factory=tuple, description="Indices of MoE layers")
    num_experts: int = Field(ge=0, description="Number of experts per layer")
    num_experts_per_tok: int = Field(ge=0, description="Number of experts selected per token")
    total_layers: int = Field(ge=1, description="Total number of layers in model")
    architecture: MoEArchitecture = MoEArchitecture.GENERIC
    has_shared_expert: bool = Field(default=False, description="Whether model has a shared expert")

    @property
    def is_moe(self) -> bool:
        """Check if this is an MoE model."""
        return len(self.moe_layers) > 0


# =============================================================================
# Generation Results
# =============================================================================


class GenerationStats(BaseModel):
    """Statistics from expert-controlled generation."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=-1, description="Forced expert index (-1 for normal)")
    tokens_generated: int = Field(ge=0, description="Number of tokens generated")
    layers_modified: int = Field(ge=0, description="Number of layers modified")
    moe_type: str = Field(description="Type of MoE architecture")
    prompt_tokens: int = Field(ge=0, default=0, description="Number of prompt tokens")


class ExpertChatResult(BaseModel):
    """Result from chatting with a specific expert."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="The input prompt")
    response: str = Field(description="The generated response")
    expert_idx: int = Field(ge=0, description="The expert that was forced")
    layer_idx: int | None = Field(default=None, description="Specific layer if targeted")
    stats: GenerationStats = Field(description="Generation statistics")


class ExpertComparisonResult(BaseModel):
    """Result from comparing multiple experts on the same prompt."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="The input prompt")
    expert_results: tuple[ExpertChatResult, ...] = Field(
        default_factory=tuple, description="Results from each expert"
    )

    def get_result_for_expert(self, expert_idx: int) -> ExpertChatResult | None:
        """Get result for a specific expert."""
        for result in self.expert_results:
            if result.expert_idx == expert_idx:
                return result
        return None


class TopKVariationResult(BaseModel):
    """Result from varying top-k expert selection."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="The input prompt")
    k_value: int = Field(ge=1, description="The top-k value used")
    default_k: int = Field(ge=1, description="The model's default top-k")
    response: str = Field(description="The generated response")
    normal_response: str = Field(description="Response with default k")


# =============================================================================
# Router Weight Capture
# =============================================================================


class RouterWeightCapture(BaseModel):
    """Captured router weights for a single token position."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    position_idx: int = Field(ge=0, description="Token position index")
    token: str = Field(default="", description="The token at this position")
    expert_indices: tuple[int, ...] = Field(
        default_factory=tuple, description="Selected expert indices"
    )
    weights: tuple[float, ...] = Field(default_factory=tuple, description="Corresponding weights")

    @property
    def top_expert(self) -> int | None:
        """Get the top-weighted expert."""
        if not self.expert_indices:
            return None
        return self.expert_indices[0]


class LayerRouterWeights(BaseModel):
    """Router weights for all positions in a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    positions: tuple[RouterWeightCapture, ...] = Field(
        default_factory=tuple, description="Weights at each position"
    )


# =============================================================================
# Layer Analysis
# =============================================================================


class LayerRoutingAnalysis(BaseModel):
    """Comprehensive routing analysis for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    entropy: RouterEntropy = Field(description="Entropy analysis")
    utilization: ExpertUtilization = Field(description="Expert utilization stats")
    coactivation: CoactivationAnalysis | None = Field(
        default=None, description="Co-activation analysis"
    )


class LayerDivergenceResult(BaseModel):
    """Result of analyzing divergence between layers."""

    model_config = ConfigDict(frozen=True)

    layer_a: int = Field(ge=0, description="First layer index")
    layer_b: int = Field(ge=0, description="Second layer index")
    divergence_score: float = Field(ge=0, description="Divergence score between layers")
    shared_experts: tuple[int, ...] = Field(
        default_factory=tuple, description="Experts frequently shared"
    )
    unique_to_a: tuple[int, ...] = Field(
        default_factory=tuple, description="Experts unique to layer A"
    )
    unique_to_b: tuple[int, ...] = Field(
        default_factory=tuple, description="Experts unique to layer B"
    )


# =============================================================================
# Pattern Discovery
# =============================================================================


class ExpertPattern(BaseModel):
    """A discovered activation pattern for an expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0, description="Expert index")
    layer_idx: int = Field(ge=0, description="Layer index")
    pattern_type: str = Field(description="Type of pattern (e.g., 'numeric', 'punctuation')")
    trigger_tokens: tuple[str, ...] = Field(
        default_factory=tuple, description="Tokens that trigger this expert"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in this pattern")
    sample_activations: int = Field(ge=0, description="Number of sample activations observed")
    description: str = Field(default="", description="Human-readable pattern description")


class ExpertTaxonomy(BaseModel):
    """Complete taxonomy of all experts in a model."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(ge=1, description="Number of layers")
    num_experts: int = Field(ge=1, description="Number of experts per layer")
    expert_identities: tuple[ExpertIdentity, ...] = Field(
        default_factory=tuple, description="Identity of each expert"
    )
    patterns: tuple[ExpertPattern, ...] = Field(
        default_factory=tuple, description="Discovered patterns"
    )
    layer_analyses: tuple[LayerRoutingAnalysis, ...] = Field(
        default_factory=tuple, description="Per-layer analysis"
    )

    def get_experts_by_role(self, role: ExpertRole) -> tuple[ExpertIdentity, ...]:
        """Get all experts with a specific role."""
        return tuple(e for e in self.expert_identities if e.role == role)

    def get_experts_by_category(self, category: ExpertCategory) -> tuple[ExpertIdentity, ...]:
        """Get all experts with a specific primary category."""
        return tuple(e for e in self.expert_identities if e.primary_category == category)

    def get_layer_analysis(self, layer_idx: int) -> LayerRoutingAnalysis | None:
        """Get analysis for a specific layer."""
        for analysis in self.layer_analyses:
            if analysis.layer_idx == layer_idx:
                return analysis
        return None


# =============================================================================
# Tokenizer Analysis
# =============================================================================


class TokenExpertMapping(BaseModel):
    """Mapping of a token to its preferred experts."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(description="The token")
    token_id: int = Field(ge=0, description="Token ID")
    preferred_experts: tuple[int, ...] = Field(
        default_factory=tuple, description="Experts that frequently handle this token"
    )
    activation_counts: tuple[int, ...] = Field(
        default_factory=tuple, description="Activation count for each expert"
    )


class VocabExpertAnalysis(BaseModel):
    """Analysis of vocabulary-to-expert mappings."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    total_tokens_analyzed: int = Field(ge=0, description="Number of tokens analyzed")
    mappings: tuple[TokenExpertMapping, ...] = Field(
        default_factory=tuple, description="Token-to-expert mappings"
    )
    expert_vocab_sizes: tuple[int, ...] = Field(
        default_factory=tuple, description="Number of tokens each expert handles"
    )
