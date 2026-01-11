"""Pydantic models for activation patching analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import CommutativityLevel, PatchEffect


class CommutativityPair(BaseModel):
    """A pair of commutative prompts and their similarity."""

    prompt_a: str = Field(description="First prompt (e.g., '2*3=')")
    prompt_b: str = Field(description="Second prompt (e.g., '3*2=')")
    similarity: float = Field(description="Cosine similarity between activations")


class CommutativityResult(BaseModel):
    """Result of commutativity analysis."""

    model_id: str = Field(description="Model identifier")
    layer: int = Field(description="Layer analyzed")
    num_pairs: int = Field(description="Number of pairs tested")
    mean_similarity: float = Field(description="Mean cosine similarity")
    std_similarity: float = Field(description="Standard deviation of similarity")
    min_similarity: float = Field(description="Minimum similarity")
    max_similarity: float = Field(description="Maximum similarity")
    pairs: list[CommutativityPair] = Field(default_factory=list)

    @property
    def level(self) -> CommutativityLevel:
        """Classify the commutativity level."""
        if self.mean_similarity > 0.999:
            return CommutativityLevel.PERFECT
        elif self.mean_similarity > 0.99:
            return CommutativityLevel.HIGH
        elif self.mean_similarity > 0.9:
            return CommutativityLevel.MODERATE
        else:
            return CommutativityLevel.LOW

    @property
    def interpretation(self) -> str:
        """Get human-readable interpretation."""
        level = self.level
        if level == CommutativityLevel.PERFECT:
            return "Perfect commutativity (>0.999): Strong evidence for lookup table (memorization)"
        elif level == CommutativityLevel.HIGH:
            return "High commutativity (>0.99): Likely lookup table with slight representation differences"
        elif level == CommutativityLevel.MODERATE:
            return "Moderate commutativity (>0.9): Partial lookup table or learned symmetry"
        else:
            return "Low commutativity (<0.9): Model may use different algorithms for A*B vs B*A"


class PatchingLayerResult(BaseModel):
    """Result of patching at a single layer."""

    layer: int = Field(description="Layer where patching occurred")
    top_token: str = Field(description="Top predicted token after patching")
    top_prob: float = Field(description="Probability of top token")
    baseline_token: str = Field(description="Baseline top token (no patching)")
    baseline_prob: float = Field(description="Baseline probability")
    effect: PatchEffect = Field(description="Effect of the patching")
    notes: str = Field(default="", description="Additional notes")

    @property
    def changed(self) -> bool:
        """Check if patching changed the prediction."""
        return self.top_token != self.baseline_token


class PatchingResult(BaseModel):
    """Complete result of activation patching experiment."""

    model_id: str = Field(description="Model identifier")
    source_prompt: str = Field(description="Source prompt for activations")
    target_prompt: str = Field(description="Target prompt to patch into")
    source_answer: str | None = Field(default=None, description="Expected source answer")
    target_answer: str | None = Field(default=None, description="Expected target answer")
    blend: float = Field(default=1.0, description="Blend factor used")
    layers: list[int] = Field(default_factory=list, description="Layers tested")
    baseline_token: str = Field(description="Baseline prediction")
    baseline_prob: float = Field(description="Baseline probability")
    layer_results: list[PatchingLayerResult] = Field(default_factory=list)

    @property
    def transferred_layers(self) -> list[int]:
        """Get layers where source answer was transferred."""
        return [r.layer for r in self.layer_results if r.effect == PatchEffect.TRANSFERRED]

    @property
    def any_transfer(self) -> bool:
        """Check if any layer showed transfer."""
        return len(self.transferred_layers) > 0
