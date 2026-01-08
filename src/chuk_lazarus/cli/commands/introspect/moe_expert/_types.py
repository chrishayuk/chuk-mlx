"""Pydantic models for MoE expert CLI commands.

This module provides typed configuration and result models for MoE analysis
to ensure type safety and eliminate dictionary "goop".
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..._base import CommandConfig, CommandResult
from ..._constants import (
    ContextType,
    ContextVerdict,
    Domain,
    LayerPhase,
    LayerPhaseDefaults,
    MoEDefaults,
    PatternCategory,
    TokenType,
)


# =============================================================================
# Token Classification Models
# =============================================================================


class TokenClassification(BaseModel):
    """Classification of a single token."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(..., description="The token string")
    token_type: TokenType = Field(..., description="Semantic type of the token")
    position: int = Field(..., description="Position in sequence")


class Trigram(BaseModel):
    """A trigram pattern (prev -> current -> next)."""

    model_config = ConfigDict(frozen=True)

    prev_type: str = Field(..., description="Previous token type (or ^ for start)")
    curr_type: TokenType = Field(..., description="Current token type")
    next_type: str = Field(..., description="Next token type (or $ for end)")

    @property
    def pattern(self) -> str:
        """Return the trigram pattern string."""
        return f"{self.prev_type}→{self.curr_type.value}→{self.next_type}"


# =============================================================================
# Expert Routing Models
# =============================================================================


class ExpertWeight(BaseModel):
    """Weight for a single expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(..., description="Expert index")
    weight: float = Field(..., description="Routing weight")


class PositionRouting(BaseModel):
    """Routing information for a single token position."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    token_type: TokenType = Field(..., description="Semantic token type")
    trigram: str = Field(..., description="Trigram pattern")
    experts: list[ExpertWeight] = Field(default_factory=list, description="Expert routing")

    @property
    def top_expert(self) -> int | None:
        """Get the top expert index."""
        return self.experts[0].expert_idx if self.experts else None


class LayerRouting(BaseModel):
    """Routing information for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(..., description="Layer index")
    positions: list[PositionRouting] = Field(default_factory=list, description="Position routing")


# =============================================================================
# Pattern Analysis Models
# =============================================================================


class PatternExpertInfo(BaseModel):
    """Information about an expert's pattern specialization."""

    model_config = ConfigDict(frozen=True)

    layer: int = Field(..., description="Layer index")
    expert: int = Field(..., description="Expert index")
    trigram: str = Field(..., description="Trigram pattern")
    count: int = Field(..., description="Activation count")
    examples: list[str] = Field(default_factory=list, description="Example contexts")


class CategoryLayerStats(BaseModel):
    """Statistics for a category at a specific layer."""

    model_config = ConfigDict(frozen=True)

    category: PatternCategory = Field(..., description="Pattern category")
    layer: int = Field(..., description="Layer index")
    expert_count: int = Field(..., description="Number of experts handling this category")
    experts: list[int] = Field(default_factory=list, description="Expert indices")


class TaxonomyResult(CommandResult):
    """Result of full taxonomy analysis."""

    model_id: str = Field(..., description="Model identifier")
    num_experts: int = Field(..., description="Number of experts in model")
    num_moe_layers: int = Field(..., description="Number of MoE layers")
    prompts_analyzed: int = Field(..., description="Number of prompts analyzed")
    pattern_experts: list[PatternExpertInfo] = Field(
        default_factory=list, description="Pattern-expert mappings"
    )
    category_stats: list[CategoryLayerStats] = Field(
        default_factory=list, description="Category-layer statistics"
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "\n=== TAXONOMY ANALYSIS ===",
            f"Model: {self.model_id}",
            f"Experts: {self.num_experts}",
            f"MoE Layers: {self.num_moe_layers}",
            f"Prompts analyzed: {self.prompts_analyzed}",
        ]
        return "\n".join(lines)


# =============================================================================
# Attention Analysis Models
# =============================================================================


class AttentionRoutingResult(BaseModel):
    """Result of attention-routing correlation analysis."""

    model_config = ConfigDict(frozen=True)

    context_name: str = Field(..., description="Context test name")
    context: str = Field(..., description="Full context string")
    tokens: list[str] = Field(..., description="Tokenized context")
    target_pos: int = Field(..., description="Target position analyzed")
    target_token: str = Field(..., description="Target token")
    primary_expert: int = Field(..., description="Primary expert selected")
    all_experts: list[int] = Field(..., description="All selected experts")
    weights: list[float] = Field(..., description="Expert weights")
    attention_summary: dict[str, float] = Field(
        default_factory=dict, description="Attention weight summary"
    )


class AttentionPatternResult(CommandResult):
    """Result of attention pattern analysis."""

    model_id: str = Field(..., description="Model identifier")
    prompt: str = Field(..., description="Analyzed prompt")
    layer: int = Field(..., description="Layer analyzed")
    query_position: int = Field(..., description="Query position")
    query_token: str = Field(..., description="Query token")
    attention_weights: list[tuple[int, float]] = Field(
        default_factory=list, description="Position-weight pairs"
    )
    expert_routing: list[ExpertWeight] = Field(
        default_factory=list, description="Expert routing for query position"
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n=== ATTENTION PATTERN (Layer {self.layer}) ===",
            f'Position {self.query_position}: "{self.query_token}"',
            "\nTop attended positions:",
        ]
        for pos, weight in self.attention_weights[:5]:
            lines.append(f"  [{pos}] {weight:.3f}")
        lines.append("\nExpert routing:")
        for ew in self.expert_routing[:4]:
            lines.append(f"  E{ew.expert_idx}: {ew.weight:.3f}")
        return "\n".join(lines)


# =============================================================================
# Domain Analysis Models
# =============================================================================


class ExpertDomainStats(BaseModel):
    """Domain statistics for an expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(..., description="Expert index")
    layer: int = Field(..., description="Layer index")
    domain_counts: dict[str, int] = Field(
        default_factory=dict, description="Count by domain"
    )
    primary_domain: Domain | None = Field(default=None, description="Primary domain")
    is_generalist: bool = Field(default=False, description="Handles multiple domains")


class DomainTestResult(CommandResult):
    """Result of domain testing."""

    model_id: str = Field(..., description="Model identifier")
    domains_tested: list[str] = Field(..., description="Domains tested")
    expert_stats: list[ExpertDomainStats] = Field(
        default_factory=list, description="Expert domain statistics"
    )
    generalist_count: int = Field(default=0, description="Number of generalist experts")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n=== DOMAIN TEST RESULTS ===",
            f"Model: {self.model_id}",
            f"Domains: {', '.join(self.domains_tested)}",
            f"Generalist experts: {self.generalist_count}",
        ]
        return "\n".join(lines)


# =============================================================================
# Context Window Models
# =============================================================================


class ContextWindowResult(BaseModel):
    """Result of context window analysis for a single position."""

    model_config = ConfigDict(frozen=True)

    context_name: str = Field(..., description="Context test name")
    layer: int = Field(..., description="Layer analyzed")
    trigram_experts: tuple[int, ...] = Field(..., description="Experts with trigram context")
    extended_experts: tuple[int, ...] = Field(..., description="Experts with extended context")
    context_affects_routing: bool = Field(
        ..., description="Whether extended context affects routing"
    )


class ContextWindowAnalysisResult(CommandResult):
    """Aggregated context window analysis result."""

    model_id: str = Field(..., description="Model identifier")
    num_layers: int = Field(..., description="Number of layers analyzed")
    results: list[ContextWindowResult] = Field(
        default_factory=list, description="Individual results"
    )
    verdict: ContextVerdict = Field(..., description="Overall verdict")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n=== CONTEXT WINDOW ANALYSIS ===",
            f"Model: {self.model_id}",
            f"Verdict: {self.verdict.value}",
        ]
        return "\n".join(lines)


# =============================================================================
# Exploration Models
# =============================================================================


class LayerExpertTransition(BaseModel):
    """Expert transition across layer phases."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    early_expert: int | None = Field(default=None, description="Dominant expert in early layers")
    middle_expert: int | None = Field(default=None, description="Dominant expert in middle layers")
    late_expert: int | None = Field(default=None, description="Dominant expert in late layers")
    has_transition: bool = Field(default=False, description="Whether expert changes between phases")

    @property
    def transition_str(self) -> str:
        """Get string representation of transitions."""
        if not self.has_transition:
            dom = self.early_expert or self.middle_expert or self.late_expert
            return f"E{dom} (stable)" if dom is not None else "unknown"
        parts = []
        if self.early_expert != self.middle_expert and self.early_expert is not None and self.middle_expert is not None:
            parts.append(f"E{self.early_expert}→E{self.middle_expert}")
        if self.middle_expert != self.late_expert and self.middle_expert is not None and self.late_expert is not None:
            parts.append(f"E{self.middle_expert}→E{self.late_expert}")
        return " then ".join(parts) if parts else "stable"


class ExploreAnalysisResult(CommandResult):
    """Result of exploration analysis."""

    prompt: str = Field(..., description="Analyzed prompt")
    layer: int = Field(..., description="Current layer")
    positions: list[PositionRouting] = Field(
        default_factory=list, description="Position routing info"
    )
    transitions: list[LayerExpertTransition] = Field(
        default_factory=list, description="Expert transitions"
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n=== EXPLORE ANALYSIS ===",
            f'Prompt: "{self.prompt}"',
            f"Layer: {self.layer}",
            f"Positions: {len(self.positions)}",
        ]
        return "\n".join(lines)


# =============================================================================
# Configuration Models
# =============================================================================


class MoEExpertConfig(CommandConfig):
    """Configuration for MoE expert CLI commands."""

    model: str = Field(..., description="Model path or name")
    prompt: str | None = Field(default=None, description="Prompt to analyze")
    layer: int | None = Field(default=None, description="Layer to analyze")
    position: int | None = Field(default=None, description="Position to analyze")
    action: str = Field(default="trace", description="Action to perform")
    verbose: bool = Field(default=False, description="Verbose output")
    output: str | None = Field(default=None, description="Output file path")

    @classmethod
    def from_args(cls, args: Namespace) -> MoEExpertConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            prompt=getattr(args, "prompt", None),
            layer=getattr(args, "layer", None),
            position=getattr(args, "position", None),
            action=getattr(args, "action", "trace"),
            verbose=getattr(args, "verbose", False),
            output=getattr(args, "output", None),
        )


class FullTaxonomyConfig(CommandConfig):
    """Configuration for full taxonomy analysis."""

    model: str = Field(..., description="Model path or name")
    categories: str | None = Field(default=None, description="Categories to analyze")
    verbose: bool = Field(default=False, description="Verbose output")

    @classmethod
    def from_args(cls, args: Namespace) -> FullTaxonomyConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            categories=getattr(args, "categories", None),
            verbose=getattr(args, "verbose", False),
        )


class ExploreConfig(CommandConfig):
    """Configuration for interactive exploration."""

    model: str = Field(..., description="Model path or name")
    layer: int = Field(default=MoEDefaults.DEFAULT_LAYER, description="Initial layer")

    @classmethod
    def from_args(cls, args: Namespace) -> ExploreConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            layer=getattr(args, "layer", MoEDefaults.DEFAULT_LAYER),
        )


class AttentionPatternConfig(CommandConfig):
    """Configuration for attention pattern analysis."""

    model: str = Field(..., description="Model path or name")
    prompt: str = Field(
        default="King is to queen as man is to woman",
        description="Prompt to analyze",
    )
    position: int | None = Field(default=None, description="Position to analyze")
    layer: int | None = Field(default=None, description="Layer to analyze")
    head: int | None = Field(default=None, description="Attention head to analyze")
    top_k: int = Field(default=5, description="Top k attention weights to show")

    @classmethod
    def from_args(cls, args: Namespace) -> AttentionPatternConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            prompt=getattr(args, "prompt", "King is to queen as man is to woman"),
            position=getattr(args, "position", None),
            layer=getattr(args, "layer", None),
            head=getattr(args, "head", None),
            top_k=getattr(args, "top_k", 5),
        )


# =============================================================================
# Utility Functions
# =============================================================================


def get_layer_phase(layer: int) -> LayerPhase:
    """Determine the phase of a layer based on its index."""
    if layer < LayerPhaseDefaults.EARLY_END:
        return LayerPhase.EARLY
    elif layer < LayerPhaseDefaults.MIDDLE_END:
        return LayerPhase.MIDDLE
    else:
        return LayerPhase.LATE


__all__ = [
    # Token Classification
    "TokenClassification",
    "Trigram",
    # Expert Routing
    "ExpertWeight",
    "PositionRouting",
    "LayerRouting",
    # Pattern Analysis
    "PatternExpertInfo",
    "CategoryLayerStats",
    "TaxonomyResult",
    # Attention
    "AttentionRoutingResult",
    "AttentionPatternResult",
    # Domain
    "ExpertDomainStats",
    "DomainTestResult",
    # Context Window
    "ContextWindowResult",
    "ContextWindowAnalysisResult",
    # Exploration
    "LayerExpertTransition",
    "ExploreAnalysisResult",
    # Config
    "MoEExpertConfig",
    "FullTaxonomyConfig",
    "ExploreConfig",
    "AttentionPatternConfig",
    # Utilities
    "get_layer_phase",
]
