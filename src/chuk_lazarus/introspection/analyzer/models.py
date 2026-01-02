"""
Pydantic models for analysis results.

This module contains the data models returned by the analyzer,
including predictions, evolutions, and complete analysis results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str = Field(description="Decoded token string")
    token_id: int = Field(description="Token ID in vocabulary")
    probability: float = Field(ge=0, le=1, description="Probability")
    rank: int = Field(ge=1, description="Rank among all tokens")

    model_config = ConfigDict(frozen=True)


class LayerPredictionResult(BaseModel):
    """Predictions at a single layer."""

    layer_idx: int = Field(ge=0, description="Layer index")
    predictions: list[TokenPrediction] = Field(description="Top-k predictions")
    entropy: float = Field(default=0.0, description="Shannon entropy of the full distribution")
    entropy_normalized: float = Field(
        default=0.0, description="Entropy normalized by max entropy (0=certain, 1=uniform)"
    )

    @property
    def top_token(self) -> str:
        """Get the top predicted token."""
        return self.predictions[0].token if self.predictions else ""

    @property
    def top_probability(self) -> float:
        """Get the probability of top token."""
        return self.predictions[0].probability if self.predictions else 0.0

    @property
    def is_confident(self) -> bool:
        """Check if the layer is confident (normalized entropy < 0.3)."""
        return self.entropy_normalized < 0.3

    model_config = ConfigDict(frozen=True)


class LayerTransition(BaseModel):
    """Metrics for the transition between two consecutive layers."""

    from_layer: int = Field(ge=0, description="Source layer index")
    to_layer: int = Field(ge=0, description="Target layer index")
    kl_divergence: float = Field(ge=0, description="KL divergence from source to target")
    js_divergence: float = Field(ge=0, description="Jensen-Shannon divergence (symmetric)")
    top_token_changed: bool = Field(description="Whether the top prediction changed")
    entropy_delta: float = Field(description="Change in entropy (negative = more confident)")

    @property
    def is_significant(self) -> bool:
        """Check if this transition shows significant computation (JS > 0.1)."""
        return self.js_divergence > 0.1

    model_config = ConfigDict(frozen=True)


class ResidualContribution(BaseModel):
    """Contribution of a component (attention or FFN) to the residual stream.

    Computed by measuring how much each component changes the probability
    distribution when its contribution is ablated.
    """

    layer_idx: int = Field(ge=0, description="Layer index")
    attention_norm: float = Field(ge=0, description="L2 norm of attention contribution")
    ffn_norm: float = Field(ge=0, description="L2 norm of FFN contribution")
    total_norm: float = Field(ge=0, description="L2 norm of total layer contribution")
    attention_fraction: float = Field(
        ge=0, le=1, description="Attention's fraction of total contribution"
    )
    ffn_fraction: float = Field(ge=0, le=1, description="FFN's fraction of total contribution")

    @property
    def dominant_component(self) -> str:
        """Which component contributes more: 'attention' or 'ffn'."""
        return "attention" if self.attention_fraction > self.ffn_fraction else "ffn"

    model_config = ConfigDict(frozen=True)


class TokenEvolutionResult(BaseModel):
    """Evolution of a specific token's probability across layers."""

    token: str = Field(description="The tracked token")
    token_id: int = Field(description="Token ID")
    layer_probabilities: dict[int, float] = Field(description="Probability at each captured layer")
    layer_ranks: dict[int, int | None] = Field(
        description="Rank at each layer (None if not in top-100)"
    )
    emergence_layer: int | None = Field(
        default=None,
        description="First layer where token becomes top-1",
    )

    model_config = ConfigDict(frozen=True)


class AnalysisResult(BaseModel):
    """Complete analysis result for a prompt."""

    prompt: str = Field(description="The analyzed prompt")
    tokens: list[str] = Field(description="Tokenized prompt")
    num_layers: int = Field(ge=1, description="Total layers in model")
    captured_layers: list[int] = Field(description="Layers that were captured")
    final_prediction: list[TokenPrediction] = Field(
        description="Top-k predictions from final layer"
    )
    layer_predictions: list[LayerPredictionResult] = Field(
        description="Predictions at each captured layer"
    )
    layer_transitions: list[LayerTransition] = Field(
        default_factory=list,
        description="Transition metrics between consecutive captured layers",
    )
    token_evolutions: list[TokenEvolutionResult] = Field(
        default_factory=list,
        description="Evolution of tracked tokens",
    )
    residual_contributions: list[ResidualContribution] = Field(
        default_factory=list,
        description="Residual stream decomposition (attention vs FFN) per layer",
    )

    @property
    def predicted_token(self) -> str:
        """Get the model's top prediction."""
        return self.final_prediction[0].token if self.final_prediction else ""

    @property
    def predicted_probability(self) -> float:
        """Get probability of top prediction."""
        return self.final_prediction[0].probability if self.final_prediction else 0.0

    @property
    def decision_layer(self) -> int | None:
        """Find the layer where the final prediction first becomes confident.

        Returns the first layer where:
        1. The top token matches the final prediction
        2. The layer is confident (normalized entropy < 0.3)

        Returns None if the model never becomes confident.
        """
        final_token = self.predicted_token
        for pred in self.layer_predictions:
            if pred.top_token == final_token and pred.is_confident:
                return pred.layer_idx
        return None

    @property
    def max_kl_transition(self) -> LayerTransition | None:
        """Find the transition with highest KL divergence (where computation happens)."""
        if not self.layer_transitions:
            return None
        return max(self.layer_transitions, key=lambda t: t.kl_divergence)

    @property
    def significant_transitions(self) -> list[LayerTransition]:
        """Get all transitions with JS divergence > 0.1."""
        return [t for t in self.layer_transitions if t.is_significant]

    @property
    def attention_dominant_layers(self) -> list[int]:
        """Layers where attention contributes more than FFN."""
        return [
            c.layer_idx for c in self.residual_contributions if c.dominant_component == "attention"
        ]

    @property
    def ffn_dominant_layers(self) -> list[int]:
        """Layers where FFN contributes more than attention."""
        return [c.layer_idx for c in self.residual_contributions if c.dominant_component == "ffn"]

    @property
    def max_attention_layer(self) -> int | None:
        """Layer with highest attention contribution."""
        if not self.residual_contributions:
            return None
        return max(self.residual_contributions, key=lambda c: c.attention_norm).layer_idx

    @property
    def max_ffn_layer(self) -> int | None:
        """Layer with highest FFN contribution."""
        if not self.residual_contributions:
            return None
        return max(self.residual_contributions, key=lambda c: c.ffn_norm).layer_idx

    model_config = ConfigDict(frozen=True)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(ge=1, description="Number of layers")
    hidden_size: int = Field(ge=1, description="Hidden dimension size")
    vocab_size: int = Field(ge=1, description="Vocabulary size")
    has_tied_embeddings: bool = Field(description="Whether embeddings are tied")

    model_config = ConfigDict(frozen=True)
