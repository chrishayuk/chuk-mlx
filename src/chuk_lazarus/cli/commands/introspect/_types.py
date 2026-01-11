"""Pydantic models for introspect CLI commands.

This module provides typed configuration and result models for CLI commands
to ensure type safety and eliminate dictionary "goop".
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .._base import CommandConfig, CommandResult
from .._constants import (
    AnalysisDefaults,
    LayerPhase,
    LayerPhaseDefaults,
    SteeringDefaults,
)

# =============================================================================
# Steering Models
# =============================================================================


class SteeringDirectionConfig(BaseModel):
    """Configuration for a steering direction vector."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    direction: Any = Field(..., description="Direction vector (numpy array)")
    layer: int = Field(..., description="Layer index for steering")
    coefficient: float = Field(
        default=SteeringDefaults.DEFAULT_COEFFICIENT,
        description="Steering coefficient",
    )
    positive_label: str | None = Field(default=None, description="Label for positive direction")
    negative_label: str | None = Field(default=None, description="Label for negative direction")
    norm: float | None = Field(default=None, description="Direction vector norm")
    cosine_similarity: float | None = Field(
        default=None, description="Cosine similarity between positive and negative"
    )
    source_file: str | None = Field(default=None, description="Source file path")

    @classmethod
    def from_npz(cls, path: str | Path) -> SteeringDirectionConfig:
        """Load direction from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            direction=data["direction"],
            layer=int(data["layer"]) if "layer" in data else 0,
            positive_label=(str(data["positive_prompt"]) if "positive_prompt" in data else None),
            negative_label=(str(data["negative_prompt"]) if "negative_prompt" in data else None),
            norm=float(data["norm"]) if "norm" in data else None,
            cosine_similarity=(
                float(data["cosine_similarity"]) if "cosine_similarity" in data else None
            ),
            source_file=str(path),
        )


class SteeringConfig(CommandConfig):
    """Configuration for steering CLI commands."""

    model: str = Field(..., description="Model path or name")
    extract: bool = Field(default=False, description="Extract direction mode")
    positive: str | None = Field(default=None, description="Positive prompt")
    negative: str | None = Field(default=None, description="Negative prompt")
    direction: str | None = Field(default=None, description="Direction file path")
    neuron: int | None = Field(default=None, description="Neuron index for steering")
    prompts: str = Field(default="", description="Prompts to process")
    layer: int | None = Field(default=None, description="Layer for steering")
    coefficient: float = Field(
        default=SteeringDefaults.DEFAULT_COEFFICIENT,
        description="Steering coefficient",
    )
    compare: str | None = Field(default=None, description="Coefficients to compare")
    max_tokens: int = Field(default=100, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    output: str | None = Field(default=None, description="Output file path")
    name: str = Field(default=SteeringDefaults.DEFAULT_NAME, description="Direction name")
    positive_label: str = Field(
        default=SteeringDefaults.DEFAULT_POSITIVE_LABEL,
        description="Positive direction label",
    )
    negative_label: str = Field(
        default=SteeringDefaults.DEFAULT_NEGATIVE_LABEL,
        description="Negative direction label",
    )

    @classmethod
    def from_args(cls, args: Namespace) -> SteeringConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            extract=getattr(args, "extract", False),
            positive=getattr(args, "positive", None),
            negative=getattr(args, "negative", None),
            direction=getattr(args, "direction", None),
            neuron=getattr(args, "neuron", None),
            prompts=getattr(args, "prompts", "") or "",
            layer=getattr(args, "layer", None),
            coefficient=getattr(args, "coefficient", SteeringDefaults.DEFAULT_COEFFICIENT),
            compare=getattr(args, "compare", None),
            max_tokens=getattr(args, "max_tokens", 100),
            temperature=getattr(args, "temperature", 0.0),
            output=getattr(args, "output", None),
            name=getattr(args, "name", None) or SteeringDefaults.DEFAULT_NAME,
            positive_label=getattr(args, "positive_label", None)
            or SteeringDefaults.DEFAULT_POSITIVE_LABEL,
            negative_label=getattr(args, "negative_label", None)
            or SteeringDefaults.DEFAULT_NEGATIVE_LABEL,
        )


class SteeringExtractionResult(CommandResult):
    """Result of steering direction extraction."""

    layer: int
    norm: float
    cosine_similarity: float
    separation: float
    output_path: str | None = None

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "\nDirection extracted:",
            f"  Layer: {self.layer}",
            f"  Norm: {self.norm:.4f}",
            f"  Cosine similarity (pos, neg): {self.cosine_similarity:.4f}",
            f"  Separation: {self.separation:.4f}",
        ]
        if self.output_path:
            lines.append(f"\nDirection saved to: {self.output_path}")
        return "\n".join(lines)


class SteeringGenerationResult(CommandResult):
    """Result of steering generation."""

    prompt: str
    output: str
    layer: int
    coefficient: float

    def to_display(self) -> str:
        """Format result for display."""
        return f"\nPrompt: {self.prompt!r}\nOutput: {self.output!r}"


# =============================================================================
# Ablation Models
# =============================================================================


class AblationConfig(CommandConfig):
    """Configuration for ablation CLI commands."""

    model: str = Field(..., description="Model path or name")
    prompt: str | None = Field(default=None, description="Single prompt")
    prompts: str | None = Field(default=None, description="Multiple prompts with expected values")
    criterion: str | None = Field(default=None, description="Criterion for evaluation")
    layers: str | None = Field(default=None, description="Layers to ablate")
    component: str = Field(default="mlp", description="Component to ablate")
    multi: bool = Field(default=False, description="Multi-layer ablation mode")
    raw: bool = Field(default=False, description="Use raw mode (no chat template)")
    max_tokens: int = Field(default=50, description="Max tokens to generate")
    verbose: bool = Field(default=False, description="Verbose output")
    output: str | None = Field(default=None, description="Output file path")

    @classmethod
    def from_args(cls, args: Namespace) -> AblationConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            prompt=getattr(args, "prompt", None),
            prompts=getattr(args, "prompts", None),
            criterion=getattr(args, "criterion", None),
            layers=getattr(args, "layers", None),
            component=getattr(args, "component", "mlp"),
            multi=getattr(args, "multi", False),
            raw=getattr(args, "raw", False),
            max_tokens=getattr(args, "max_tokens", 50),
            verbose=getattr(args, "verbose", False),
            output=getattr(args, "output", None),
        )


class AblationResult(CommandResult):
    """Result of a single ablation test."""

    prompt: str
    expected: str
    ablation: str
    output: str
    correct: bool

    def to_display(self) -> str:
        """Format result for display."""
        status = "PASS" if self.correct else "FAIL"
        return f"[{status}] {self.ablation}: {self.output[:50]}..."


class MultiPromptAblationResult(CommandResult):
    """Result of multi-prompt ablation."""

    ablation_name: str
    results: list[AblationResult]

    def to_display(self) -> str:
        """Format result for display."""
        lines = [f"Ablation: {self.ablation_name}"]
        for r in self.results:
            lines.append(f"  {r.to_display()}")
        return "\n".join(lines)


# =============================================================================
# Neuron Analysis Models
# =============================================================================


class NeuronAnalysisConfig(CommandConfig):
    """Configuration for neuron analysis CLI commands."""

    model: str = Field(..., description="Model path or name")
    prompts: str = Field(..., description="Prompts to analyze")
    layer: int | None = Field(default=None, description="Single layer to analyze")
    layers: str | None = Field(default=None, description="Multiple layers to analyze")
    neurons: str | None = Field(default=None, description="Neuron indices to analyze")
    from_direction: str | None = Field(default=None, description="Direction file for neurons")
    auto_discover: bool = Field(default=False, description="Auto-discover discriminative neurons")
    labels: str | None = Field(default=None, description="Labels for prompts")
    top_k: int = Field(default=AnalysisDefaults.TOP_K, description="Top k neurons")
    neuron_names: str | None = Field(default=None, description="Names for neurons")
    steer: str | None = Field(default=None, description="Steering config")
    strength: float | None = Field(default=None, description="Steering strength")
    output: str | None = Field(default=None, description="Output file path")

    @classmethod
    def from_args(cls, args: Namespace) -> NeuronAnalysisConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            prompts=args.prompts,
            layer=getattr(args, "layer", None),
            layers=getattr(args, "layers", None),
            neurons=getattr(args, "neurons", None),
            from_direction=getattr(args, "from_direction", None),
            auto_discover=getattr(args, "auto_discover", False),
            labels=getattr(args, "labels", None),
            top_k=getattr(args, "top_k", AnalysisDefaults.TOP_K),
            neuron_names=getattr(args, "neuron_names", None),
            steer=getattr(args, "steer", None),
            strength=getattr(args, "strength", None),
            output=getattr(args, "output", None),
        )


class NeuronStats(BaseModel):
    """Statistics for a single neuron."""

    model_config = ConfigDict(frozen=True)

    index: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    weight: float | None = None
    separation: float | None = None


class NeuronAnalysisResult(CommandResult):
    """Result of neuron analysis."""

    model_id: str
    layers: list[int]
    neurons: list[int]
    prompts: list[str]
    labels: list[str] | None
    stats_by_layer: dict[int, list[NeuronStats]]

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\nNeuron Analysis: {self.model_id}",
            f"Layers: {self.layers}",
            f"Neurons: {self.neurons}",
        ]
        for layer, stats in self.stats_by_layer.items():
            lines.append(f"\nLayer {layer}:")
            for s in stats:
                lines.append(
                    f"  N{s.index}: min={s.min_val:+.1f}, max={s.max_val:+.1f}, "
                    f"mean={s.mean_val:+.1f}, std={s.std_val:.1f}"
                )
        return "\n".join(lines)


# =============================================================================
# Embedding Analysis Models
# =============================================================================


class EmbeddingAnalysisConfig(CommandConfig):
    """Configuration for embedding analysis CLI commands."""

    model: str = Field(..., description="Model path or name")
    operation: str | None = Field(default=None, description="Operation type to analyze")
    layers: str | None = Field(default=None, description="Layers to analyze")
    output: str | None = Field(default=None, description="Output file path")

    @classmethod
    def from_args(cls, args: Namespace) -> EmbeddingAnalysisConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            operation=getattr(args, "operation", None),
            layers=getattr(args, "layers", None),
            output=getattr(args, "output", None),
        )


class EmbeddingAnalysisResult(CommandResult):
    """Result of embedding analysis."""

    model_id: str
    task_from_embedding: float
    task_by_layer: dict[int, float]
    answer_r2_embedding: float
    answer_r2_by_layer: dict[int, float]
    within_arith_sim: float
    within_lang_sim: float
    between_task_sim: float

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "\n=== EMBEDDING ANALYSIS ===",
            f"Task type from embeddings: {self.task_from_embedding:.1%}",
            f"Answer R2 from embeddings: {self.answer_r2_embedding:.3f}",
            f"\nWithin arithmetic similarity: {self.within_arith_sim:.4f}",
            f"Within language similarity: {self.within_lang_sim:.4f}",
            f"Between task similarity: {self.between_task_sim:.4f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Direction Comparison Models
# =============================================================================


class DirectionComparisonConfig(CommandConfig):
    """Configuration for direction comparison CLI commands."""

    files: list[str] = Field(..., description="Direction files to compare")
    threshold: float = Field(default=0.1, description="Orthogonality threshold")
    output: str | None = Field(default=None, description="Output file path")

    @classmethod
    def from_args(cls, args: Namespace) -> DirectionComparisonConfig:
        """Create config from argparse Namespace."""
        return cls(
            files=args.files,
            threshold=getattr(args, "threshold", 0.1),
            output=getattr(args, "output", None),
        )


class DirectionPairSimilarity(BaseModel):
    """Similarity between two direction vectors."""

    model_config = ConfigDict(frozen=True)

    name_a: str
    name_b: str
    cosine_similarity: float
    orthogonal: bool


class DirectionComparisonResult(CommandResult):
    """Result of direction comparison."""

    files: list[str]
    names: list[str]
    pairs: list[DirectionPairSimilarity]
    mean_abs_similarity: float

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            "\n=== DIRECTION COMPARISON ===",
            f"Total pairs: {len(self.pairs)}",
            f"Mean |cosine similarity|: {self.mean_abs_similarity:.3f}",
        ]
        orthogonal = [p for p in self.pairs if p.orthogonal]
        if orthogonal:
            lines.append(f"\nOrthogonal pairs ({len(orthogonal)}):")
            for p in orthogonal:
                lines.append(f"  {p.name_a} ⊥ {p.name_b} (cos={p.cosine_similarity:+.3f})")
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================


def get_layer_phase(layer: int, total_layers: int | None = None) -> LayerPhase:
    """Determine the phase of a layer based on its index."""
    if layer < LayerPhaseDefaults.EARLY_END:
        return LayerPhase.EARLY
    elif layer < LayerPhaseDefaults.MIDDLE_END:
        return LayerPhase.MIDDLE
    else:
        return LayerPhase.LATE


def parse_layers_string(layers_str: str | None) -> list[int] | None:
    """Parse comma-separated layer list with support for ranges."""
    if not layers_str:
        return None

    layers = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


__all__ = [
    # Steering
    "SteeringDirectionConfig",
    "SteeringConfig",
    "SteeringExtractionResult",
    "SteeringGenerationResult",
    # Ablation
    "AblationConfig",
    "AblationResult",
    "MultiPromptAblationResult",
    # Neuron
    "NeuronAnalysisConfig",
    "NeuronStats",
    "NeuronAnalysisResult",
    # Embedding
    "EmbeddingAnalysisConfig",
    "EmbeddingAnalysisResult",
    # Direction
    "DirectionComparisonConfig",
    "DirectionPairSimilarity",
    "DirectionComparisonResult",
    # Utilities
    "get_layer_phase",
    "parse_layers_string",
]
