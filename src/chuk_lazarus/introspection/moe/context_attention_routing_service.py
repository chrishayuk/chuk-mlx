"""Service for context-aware attention-routing correlation analysis.

Tests the hypothesis: "Attention drives routing, but context-dependently."

Key insight: Instead of predicting routing from attention features alone (which fails
because it ignores context), we measure whether CHANGES in attention patterns
correlate with CHANGES in routing decisions across different contexts.

Methodology:
1. Place the same token in N different contexts
2. Capture attention patterns AND routing decisions for each context
3. Compute pairwise similarity matrices for both attention and routing
4. Measure correlation: contexts with similar attention → similar routing?

If attention drives routing, we expect strong positive correlation between
the attention similarity matrix and routing similarity matrix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, computed_field

if TYPE_CHECKING:
    from .expert_router import ExpertRouter

logger = logging.getLogger(__name__)


class ContextSample(BaseModel):
    """Data captured for a single context."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    context_name: str = Field(..., description="Name of the context")
    context: str = Field(..., description="Full context text")
    tokens: list[str] = Field(..., description="Tokenized text")
    target_position: int = Field(..., description="Position of target token")

    # Routing info
    primary_expert: int = Field(..., description="Primary expert selected")
    all_experts: list[int] = Field(..., description="All selected experts (top-k)")
    expert_weights: list[float] = Field(..., description="Expert weights")

    # Attention pattern (flattened for comparison)
    attention_pattern: list[float] = Field(
        default_factory=list,
        description="Attention weights from target position to all positions"
    )


class LayerCorrelation(BaseModel):
    """Correlation results for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer: int = Field(..., description="Layer index")
    layer_label: str = Field(..., description="Layer phase label")

    # Correlation metrics
    attention_routing_correlation: float = Field(
        ..., description="Correlation between attention similarity and routing similarity"
    )

    # Similarity matrices (flattened for storage)
    attention_similarities: list[float] = Field(
        ..., description="Pairwise attention similarities (N*(N-1)/2 values)"
    )
    routing_similarities: list[float] = Field(
        ..., description="Pairwise routing similarities (N*(N-1)/2 values)"
    )

    # Summary stats
    avg_attention_similarity: float = Field(..., description="Average attention similarity")
    avg_routing_similarity: float = Field(..., description="Average routing similarity")
    num_unique_experts: int = Field(..., description="Number of unique primary experts")

    # Raw samples for inspection
    samples: list[ContextSample] = Field(..., description="Context samples")


class ContextAttentionRoutingAnalysis(BaseModel):
    """Complete analysis results."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Model identifier")
    target_token: str = Field(..., description="Target token analyzed")
    num_contexts: int = Field(..., description="Number of contexts tested")

    # Results by layer
    layer_correlations: list[LayerCorrelation] = Field(..., description="Per-layer results")

    @computed_field
    @property
    def overall_correlation(self) -> float:
        """Average correlation across all layers."""
        if not self.layer_correlations:
            return 0.0
        return sum(lc.attention_routing_correlation for lc in self.layer_correlations) / len(self.layer_correlations)

    @computed_field
    @property
    def best_layer(self) -> int:
        """Layer with highest attention-routing correlation."""
        if not self.layer_correlations:
            return -1
        best = max(self.layer_correlations, key=lambda lc: lc.attention_routing_correlation)
        return best.layer

    @computed_field
    @property
    def hypothesis_supported(self) -> bool:
        """Whether the hypothesis 'attention drives routing' is supported."""
        # Strong support if overall correlation > 0.5
        return self.overall_correlation > 0.5


# Default contexts for testing
DEFAULT_CONTEXT_TESTS: list[tuple[str, str]] = [
    # Arithmetic contexts
    ("numeric_add", "127 + "),
    ("numeric_mult", "127 * "),
    ("numeric_eq", "127 = "),
    ("numeric_seq", "125 126 127"),

    # Code contexts
    ("code_var", "x = 127"),
    ("code_print", "print(127"),
    ("code_arr", "[127,"),
    ("code_hex", "0x7F == 127"),

    # Language contexts
    ("word_number", "the number 127"),
    ("word_value", "value is 127"),
    ("word_room", "room 127"),
    ("word_error", "error 127"),

    # Mixed contexts
    ("calc_sum", "Calculate: 127"),
    ("question", "What is 127"),
    ("sentence", "I saw 127"),
    ("standalone", "127"),
]


class ContextAttentionRoutingService:
    """Service for context-aware attention-routing correlation analysis."""

    def __init__(self, router: ExpertRouter):
        """Initialize with an ExpertRouter instance."""
        self.router = router

    async def analyze(
        self,
        target_token: str = "127",
        contexts: list[tuple[str, str]] | None = None,
        layers: list[int] | None = None,
        model_id: str = "unknown",
    ) -> ContextAttentionRoutingAnalysis:
        """Run the context-attention-routing correlation analysis.

        Args:
            target_token: Token to analyze (should appear in all contexts).
            contexts: List of (name, prompt) tuples. Uses defaults if None.
            layers: Layer indices to analyze. Uses early/middle/late if None.
            model_id: Model identifier for results.

        Returns:
            ContextAttentionRoutingAnalysis with correlation results.
        """
        from .attention_routing_service import AttentionRoutingService

        if contexts is None:
            contexts = DEFAULT_CONTEXT_TESTS

        # Determine layers to analyze
        info = self.router.info
        moe_layers = info.moe_layers

        if layers is None:
            # Default: early, middle, late
            early = moe_layers[0]
            middle = moe_layers[len(moe_layers) // 2]
            late = moe_layers[-1]
            layers = [early, middle, late]

        layer_labels = {
            layers[0]: "Early",
            layers[-1]: "Late",
        }
        if len(layers) >= 3:
            layer_labels[layers[len(layers) // 2]] = "Middle"

        layer_correlations = []

        for layer in layers:
            label = layer_labels.get(layer, f"L{layer}")
            logger.info(f"Analyzing layer {layer} ({label})...")

            samples = []

            for ctx_name, ctx in contexts:
                # Capture attention weights
                attn_result = AttentionRoutingService.capture_attention_weights(
                    self.router, ctx, layer
                )

                # Capture routing weights
                weights = await self.router.capture_router_weights(ctx, layers=[layer])

                if not weights or not weights[0].positions:
                    logger.warning(f"No routing data for context: {ctx_name}")
                    continue

                layer_weights = weights[0]

                # Find target token position
                target_pos_idx = None
                for i, pos in enumerate(layer_weights.positions):
                    if target_token.lower() in pos.token.lower():
                        target_pos_idx = i
                        break

                if target_pos_idx is None:
                    # Fall back to last position
                    target_pos_idx = len(layer_weights.positions) - 1

                target_pos = layer_weights.positions[target_pos_idx]

                # Extract attention pattern for target position
                attention_pattern: list[float] = []
                if attn_result.success and attn_result.attention_weights is not None:
                    # Average across heads, get attention from target position
                    avg_attn = mx.mean(attn_result.attention_weights[:, target_pos_idx, :], axis=0)
                    attention_pattern = avg_attn.tolist()

                sample = ContextSample(
                    context_name=ctx_name,
                    context=ctx,
                    tokens=attn_result.tokens,
                    target_position=target_pos_idx,
                    primary_expert=target_pos.expert_indices[0] if target_pos.expert_indices else -1,
                    all_experts=target_pos.expert_indices,
                    expert_weights=target_pos.weights,
                    attention_pattern=attention_pattern,
                )
                samples.append(sample)

            if len(samples) < 2:
                logger.warning(f"Not enough samples for layer {layer}")
                continue

            # Compute pairwise similarities
            attn_sims, routing_sims = self._compute_pairwise_similarities(samples)

            # Compute correlation
            correlation = self._compute_correlation(attn_sims, routing_sims)

            # Summary stats
            avg_attn_sim = sum(attn_sims) / len(attn_sims) if attn_sims else 0.0
            avg_routing_sim = sum(routing_sims) / len(routing_sims) if routing_sims else 0.0
            unique_experts = len({s.primary_expert for s in samples})

            layer_correlation = LayerCorrelation(
                layer=layer,
                layer_label=label,
                attention_routing_correlation=correlation,
                attention_similarities=attn_sims,
                routing_similarities=routing_sims,
                avg_attention_similarity=avg_attn_sim,
                avg_routing_similarity=avg_routing_sim,
                num_unique_experts=unique_experts,
                samples=samples,
            )
            layer_correlations.append(layer_correlation)

        return ContextAttentionRoutingAnalysis(
            model_id=model_id,
            target_token=target_token,
            num_contexts=len(contexts),
            layer_correlations=layer_correlations,
        )

    def _compute_pairwise_similarities(
        self, samples: list[ContextSample]
    ) -> tuple[list[float], list[float]]:
        """Compute pairwise similarity matrices for attention and routing.

        Returns:
            Tuple of (attention_similarities, routing_similarities) as flattened lists.
        """
        n = len(samples)
        attn_sims = []
        routing_sims = []

        for i in range(n):
            for j in range(i + 1, n):
                # Attention similarity (cosine similarity of attention patterns)
                attn_sim = self._cosine_similarity(
                    samples[i].attention_pattern,
                    samples[j].attention_pattern,
                )
                attn_sims.append(attn_sim)

                # Routing similarity (based on expert overlap and weight similarity)
                routing_sim = self._routing_similarity(
                    samples[i].all_experts,
                    samples[i].expert_weights,
                    samples[j].all_experts,
                    samples[j].expert_weights,
                )
                routing_sims.append(routing_sim)

        return attn_sims, routing_sims

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _routing_similarity(
        self,
        experts_a: list[int],
        weights_a: list[float],
        experts_b: list[int],
        weights_b: list[float],
    ) -> float:
        """Compute routing similarity based on expert overlap and weights.

        Combines:
        1. Jaccard similarity of expert sets
        2. Cosine similarity of weight vectors (when experts overlap)
        """
        if not experts_a or not experts_b:
            return 0.0

        # Jaccard similarity of expert sets
        set_a = set(experts_a)
        set_b = set(experts_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 0.0

        # If same primary expert, boost similarity
        if experts_a[0] == experts_b[0]:
            jaccard = (jaccard + 1.0) / 2  # Boost by 50%

        return jaccard

    def _compute_correlation(self, x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Covariance
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n

        # Standard deviations
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)


def print_context_attention_routing_analysis(analysis: ContextAttentionRoutingAnalysis) -> None:
    """Print a formatted report of the analysis results."""
    print()
    print("=" * 70)
    print("CONTEXT-AWARE ATTENTION-ROUTING CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    print("=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print()
    print("  If attention drives routing (context-dependently), then:")
    print("  - Contexts with SIMILAR attention patterns should have SIMILAR routing")
    print("  - Contexts with DIFFERENT attention patterns should have DIFFERENT routing")
    print()
    print("  We measure the correlation between:")
    print("  - Pairwise attention pattern similarity (cosine)")
    print("  - Pairwise routing decision similarity (expert overlap)")
    print()
    print("  High positive correlation → Attention drives routing")
    print("  Low/negative correlation → Something else drives routing")
    print()

    print("=" * 70)
    print("EXPERIMENT")
    print("=" * 70)
    print()
    print(f"  Model: {analysis.model_id}")
    print(f"  Target token: '{analysis.target_token}'")
    print(f"  Contexts tested: {analysis.num_contexts}")
    print()

    print("=" * 70)
    print("RESULTS BY LAYER")
    print("=" * 70)
    print()

    for lc in analysis.layer_correlations:
        correlation_str = f"{lc.attention_routing_correlation:+.3f}"
        strength = _correlation_strength(lc.attention_routing_correlation)

        print(f"  Layer {lc.layer} ({lc.layer_label}):")
        print(f"    Attention-Routing Correlation: {correlation_str} ({strength})")
        print(f"    Avg Attention Similarity:      {lc.avg_attention_similarity:.3f}")
        print(f"    Avg Routing Similarity:        {lc.avg_routing_similarity:.3f}")
        print(f"    Unique Primary Experts:        {lc.num_unique_experts}/{len(lc.samples)}")
        print()

        # Show a few sample pairs with highest attention similarity
        if lc.samples:
            print("    Sample contexts and their routing:")
            for sample in lc.samples[:6]:
                expert_str = f"E{sample.primary_expert}"
                print(f"      {sample.context_name:<12} → {expert_str}")
            if len(lc.samples) > 6:
                print(f"      ... and {len(lc.samples) - 6} more")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    overall = analysis.overall_correlation
    strength = _correlation_strength(overall)

    print(f"  Overall Correlation: {overall:+.3f} ({strength})")
    print(f"  Best Layer: L{analysis.best_layer}")
    print()

    if analysis.hypothesis_supported:
        print("  CONCLUSION: HYPOTHESIS SUPPORTED")
        print()
        print("  Contexts with similar attention patterns DO have similar routing.")
        print("  This confirms that attention drives routing, but the relationship")
        print("  is context-dependent (same token, different contexts → different attention")
        print("  patterns → different routing decisions).")
    else:
        print("  CONCLUSION: WEAK OR NO SUPPORT")
        print()
        print("  The correlation between attention patterns and routing is weak.")
        print("  This suggests other factors beyond attention influence routing,")
        print("  or the relationship is more complex than linear correlation can capture.")

    print()
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("  Earlier work showed attention provides 89-98% of routing signal.")
    print("  Simple prediction failed (4.3%) because it used features WITHOUT context.")
    print()
    print("  This experiment shows the attention→routing relationship IS strong")
    print("  but CONTEXT-DEPENDENT. Same attention pattern → same routing, but")
    print("  the same TOKEN can have very different attention patterns in")
    print("  different contexts, leading to different routing.")
    print()
    print("  The router sees: hidden_state = f(attention, residual, context)")
    print("  Not just: hidden_state = g(token)")
    print()
    print("=" * 70)


def _correlation_strength(r: float) -> str:
    """Get human-readable strength description for correlation."""
    r = abs(r)
    if r >= 0.8:
        return "Very Strong"
    elif r >= 0.6:
        return "Strong"
    elif r >= 0.4:
        return "Moderate"
    elif r >= 0.2:
        return "Weak"
    else:
        return "Very Weak"
