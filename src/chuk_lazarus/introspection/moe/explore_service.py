"""Service layer for MoE expert exploration.

Provides business logic for interactive exploration of expert routing patterns.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .._shared_constants import LayerPhaseDefaults
from .analysis_service import classify_token

# =============================================================================
# Result Models
# =============================================================================


class TokenAnalysis(BaseModel):
    """Analysis of a single token."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    token_type: str = Field(..., description="Semantic type")
    trigram: str = Field(..., description="Trigram pattern")
    top_expert: int | None = Field(default=None, description="Top expert index")
    all_experts: list[int] = Field(default_factory=list, description="All selected experts")
    expert_weights: list[float] = Field(default_factory=list, description="Expert weights")


class PatternMatch(BaseModel):
    """A matched pattern in the routing."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    trigram: str = Field(..., description="Trigram pattern")
    pattern_type: str = Field(..., description="Type of pattern detected")
    top_expert: int | None = Field(default=None, description="Top expert for this position")


class LayerPhaseData(BaseModel):
    """Expert routing data for a layer phase."""

    model_config = ConfigDict(frozen=True)

    phase_name: str = Field(..., description="Phase name (early/middle/late)")
    layer_range: str = Field(..., description="Layer range description")
    layer_experts: list[tuple[int, int]] = Field(
        default_factory=list, description="(layer, expert) pairs"
    )
    dominant_expert: int | None = Field(default=None, description="Most common expert in phase")


class PositionEvolution(BaseModel):
    """Evolution of a position across layer phases."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    trigram: str = Field(..., description="Trigram pattern")
    early: LayerPhaseData = Field(..., description="Early phase data")
    middle: LayerPhaseData = Field(..., description="Middle phase data")
    late: LayerPhaseData = Field(..., description="Late phase data")
    has_transition: bool = Field(default=False, description="Whether expert changes between phases")
    transitions: list[str] = Field(default_factory=list, description="Transition descriptions")


class ComparisonResult(BaseModel):
    """Result of comparing two prompts."""

    model_config = ConfigDict(frozen=True)

    prompt1: str = Field(..., description="First prompt")
    prompt2: str = Field(..., description="Second prompt")
    layer: int = Field(..., description="Layer analyzed")
    tokens1: list[TokenAnalysis] = Field(default_factory=list, description="Tokens from prompt 1")
    tokens2: list[TokenAnalysis] = Field(default_factory=list, description="Tokens from prompt 2")
    shared_experts: list[int] = Field(default_factory=list, description="Experts used by both")
    only_prompt1: list[int] = Field(default_factory=list, description="Experts only in prompt 1")
    only_prompt2: list[int] = Field(default_factory=list, description="Experts only in prompt 2")
    overlap_ratio: float = Field(default=0.0, description="Overlap ratio")


class DeepDiveResult(BaseModel):
    """Result of deep diving into a specific position."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    token_type: str = Field(..., description="Semantic type")
    trigram: str = Field(..., description="Trigram pattern")
    prev_token: str = Field(..., description="Previous token")
    prev_type: str = Field(..., description="Previous token type")
    next_token: str = Field(..., description="Next token")
    next_type: str = Field(..., description="Next token type")
    layer_routing: list[tuple[int, list[tuple[int, float]]]] = Field(
        default_factory=list, description="(layer, [(expert, weight)]) data"
    )
    all_experts: list[int] = Field(default_factory=list, description="All experts used")
    dominant_expert: int | None = Field(default=None, description="Most common expert")
    peak_layer: int | None = Field(default=None, description="Layer with peak activity")


# =============================================================================
# Pattern Detection Constants
# =============================================================================

# Patterns to detect and their descriptions
PATTERN_TRIGGERS = {
    "analogy_marker": (lambda t, s, p, n: "TO" in t and "AS" in str(s)),
    "analogy_pivot": (lambda t, s, p, n: "→AS→" in t),
    "arithmetic_operator": (lambda t, s, p, n: "→OP→" in t or "OP→" in t),
    "number_before_op": (lambda t, s, p, n: "NUM→OP" in t),
    "code_start": (lambda t, s, p, n: "^→KW" in t),
    "sequence_start": (lambda t, s, p, n: p == "^"),
    "sequence_end": (lambda t, s, p, n: n == "$"),
    "synonym_relation": (lambda t, s, p, n: "→SYN→" in t),
    "antonym_relation": (lambda t, s, p, n: "→ANT→" in t),
}

# Semantic types that indicate interesting positions
INTERESTING_SEMANTIC_TYPES = frozenset({"AS", "TO", "SYN", "ANT", "CAUSE", "THAN"})

# Content word types
CONTENT_WORD_TYPES = frozenset({"NOUN", "ADJ", "VERB"})


# =============================================================================
# Service Class
# =============================================================================


class ExploreService:
    """Service for MoE expert exploration analysis."""

    @staticmethod
    def analyze_routing(
        tokens: list[str],
        positions: list[Any],
    ) -> list[TokenAnalysis]:
        """Analyze routing for a list of tokens.

        Args:
            tokens: List of token strings.
            positions: List of position routing data.

        Returns:
            List of TokenAnalysis results.
        """
        sem_types = [classify_token(t).value for t in tokens]
        results = []

        for i, (tok, pos) in enumerate(zip(tokens, positions)):
            prev_t = sem_types[i - 1] if i > 0 else "^"
            curr_t = sem_types[i]
            next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
            trigram = f"{prev_t}→{curr_t}→{next_t}"

            expert_indices = pos.expert_indices if hasattr(pos, "expert_indices") else []
            expert_weights = list(pos.weights) if hasattr(pos, "weights") and pos.weights else []

            results.append(
                TokenAnalysis(
                    position=i,
                    token=tok,
                    token_type=curr_t,
                    trigram=trigram,
                    top_expert=expert_indices[0] if expert_indices else None,
                    all_experts=list(expert_indices),
                    expert_weights=expert_weights,
                )
            )

        return results

    @staticmethod
    def find_patterns(
        tokens: list[str],
        positions: list[Any],
    ) -> list[PatternMatch]:
        """Find interesting patterns in routing data.

        Args:
            tokens: List of token strings.
            positions: List of position routing data.

        Returns:
            List of PatternMatch results.
        """
        sem_types = [classify_token(t).value for t in tokens]
        patterns_found = []

        for i, (tok, pos) in enumerate(zip(tokens, positions)):
            prev_t = sem_types[i - 1] if i > 0 else "^"
            curr_t = sem_types[i]
            next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
            trigram = f"{prev_t}→{curr_t}→{next_t}"

            top_exp = (
                pos.expert_indices[0]
                if hasattr(pos, "expert_indices") and pos.expert_indices
                else None
            )

            # Check each pattern
            for pattern_name, check_fn in PATTERN_TRIGGERS.items():
                if check_fn(trigram, sem_types, prev_t, next_t):
                    patterns_found.append(
                        PatternMatch(
                            position=i,
                            token=tok.strip(),
                            trigram=trigram,
                            pattern_type=pattern_name.replace("_", " "),
                            top_expert=top_exp,
                        )
                    )
                    break  # Only one pattern per position

        return patterns_found

    @staticmethod
    def find_interesting_positions(
        tokens: list[str],
        top_k: int = 4,
    ) -> list[int]:
        """Find positions with interesting patterns.

        Args:
            tokens: List of token strings.
            top_k: Number of top positions to return.

        Returns:
            List of position indices sorted by interest score.
        """
        sem_types = [classify_token(t).value for t in tokens]
        scored = []

        for i, (tok, sem_type) in enumerate(zip(tokens, sem_types)):
            score = 0
            prev_t = sem_types[i - 1] if i > 0 else "^"
            next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"

            # Position markers
            if prev_t == "^":
                score += 2
            if next_t == "$":
                score += 2

            # Semantic relations
            if sem_type in INTERESTING_SEMANTIC_TYPES:
                score += 3

            # Operators
            if sem_type == "OP":
                score += 2

            # Content words in specific patterns
            if sem_type in CONTENT_WORD_TYPES and prev_t in {"AS", "TO"}:
                score += 2

            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        return [idx for _, idx in scored[:top_k]]

    @staticmethod
    def analyze_layer_evolution(
        tokens: list[str],
        weights_by_layer: list[Any],
        position: int,
    ) -> PositionEvolution:
        """Analyze how a position's routing evolves across layers.

        Args:
            tokens: List of token strings.
            weights_by_layer: List of layer routing data.
            position: Position index to analyze.

        Returns:
            PositionEvolution result.
        """
        sem_types = [classify_token(t).value for t in tokens]
        prev_t = sem_types[position - 1] if position > 0 else "^"
        curr_t = sem_types[position]
        next_t = sem_types[position + 1] if position < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"

        # Collect layer-expert pairs
        layer_experts = []
        for layer_weights in weights_by_layer:
            layer_idx = layer_weights.layer_idx
            if position < len(layer_weights.positions):
                pos = layer_weights.positions[position]
                top = pos.expert_indices[0] if pos.expert_indices else None
                if top is not None:
                    layer_experts.append((layer_idx, top))

        # Split by phase using constants
        early_data = [
            (layer, exp) for layer, exp in layer_experts if layer < LayerPhaseDefaults.EARLY_END
        ]
        middle_data = [
            (layer, exp)
            for layer, exp in layer_experts
            if LayerPhaseDefaults.EARLY_END <= layer < LayerPhaseDefaults.MIDDLE_END
        ]
        late_data = [
            (layer, exp) for layer, exp in layer_experts if layer >= LayerPhaseDefaults.MIDDLE_END
        ]

        def get_dominant(data: list[tuple[int, int]]) -> int | None:
            if not data:
                return None
            counts = Counter(exp for _, exp in data)
            return counts.most_common(1)[0][0]

        early_dom = get_dominant(early_data)
        mid_dom = get_dominant(middle_data)
        late_dom = get_dominant(late_data)

        # Check for transitions
        transitions = []
        if early_dom != mid_dom and early_dom is not None and mid_dom is not None:
            transitions.append(f"E{early_dom}→E{mid_dom}")
        if mid_dom != late_dom and mid_dom is not None and late_dom is not None:
            transitions.append(f"E{mid_dom}→E{late_dom}")

        return PositionEvolution(
            position=position,
            token=tokens[position],
            trigram=trigram,
            early=LayerPhaseData(
                phase_name="early",
                layer_range=f"L0-{LayerPhaseDefaults.EARLY_END - 1}",
                layer_experts=early_data,
                dominant_expert=early_dom,
            ),
            middle=LayerPhaseData(
                phase_name="middle",
                layer_range=f"L{LayerPhaseDefaults.EARLY_END}-{LayerPhaseDefaults.MIDDLE_END - 1}",
                layer_experts=middle_data,
                dominant_expert=mid_dom,
            ),
            late=LayerPhaseData(
                phase_name="late",
                layer_range=f"L{LayerPhaseDefaults.MIDDLE_END}+",
                layer_experts=late_data,
                dominant_expert=late_dom,
            ),
            has_transition=len(transitions) > 0,
            transitions=transitions,
        )

    @staticmethod
    def compare_routing(
        tokens1: list[str],
        positions1: list[Any],
        tokens2: list[str],
        positions2: list[Any],
        prompt1: str,
        prompt2: str,
        layer: int,
    ) -> ComparisonResult:
        """Compare routing between two prompts.

        Args:
            tokens1: Tokens from first prompt.
            positions1: Routing data from first prompt.
            tokens2: Tokens from second prompt.
            positions2: Routing data from second prompt.
            prompt1: First prompt string.
            prompt2: Second prompt string.
            layer: Layer being compared.

        Returns:
            ComparisonResult.
        """
        analysis1 = ExploreService.analyze_routing(tokens1, positions1)
        analysis2 = ExploreService.analyze_routing(tokens2, positions2)

        # Collect expert sets
        experts1 = set()
        for pos in positions1:
            if hasattr(pos, "expert_indices"):
                experts1.update(pos.expert_indices)

        experts2 = set()
        for pos in positions2:
            if hasattr(pos, "expert_indices"):
                experts2.update(pos.expert_indices)

        shared = experts1 & experts2
        only1 = experts1 - experts2
        only2 = experts2 - experts1
        total = len(experts1 | experts2)

        return ComparisonResult(
            prompt1=prompt1,
            prompt2=prompt2,
            layer=layer,
            tokens1=analysis1,
            tokens2=analysis2,
            shared_experts=sorted(shared),
            only_prompt1=sorted(only1),
            only_prompt2=sorted(only2),
            overlap_ratio=len(shared) / max(1, total),
        )

    @staticmethod
    def deep_dive_position(
        tokens: list[str],
        weights_by_layer: list[Any],
        position: int,
    ) -> DeepDiveResult:
        """Deep dive into a specific position's routing.

        Args:
            tokens: List of token strings.
            weights_by_layer: List of layer routing data.
            position: Position index to analyze.

        Returns:
            DeepDiveResult.
        """
        sem_types = [classify_token(t).value for t in tokens]

        tok = tokens[position]
        curr_t = sem_types[position]
        prev_t = sem_types[position - 1] if position > 0 else "^"
        next_t = sem_types[position + 1] if position < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"

        prev_tok = tokens[position - 1] if position > 0 else "^"
        next_tok = tokens[position + 1] if position < len(tokens) - 1 else "$"

        # Collect routing data across layers
        all_experts: set[int] = set()
        layer_routing = []
        exp_layer_counts: dict[int, list[int]] = {}

        for layer_weights in weights_by_layer:
            if position < len(layer_weights.positions):
                pos = layer_weights.positions[position]
                all_experts.update(pos.expert_indices)

                if hasattr(pos, "weights") and pos.weights:
                    exp_weights = list(zip(pos.expert_indices, pos.weights))
                else:
                    n = len(pos.expert_indices)
                    exp_weights = [(e, 1.0 / n) for e in pos.expert_indices]

                layer_routing.append((layer_weights.layer_idx, exp_weights))

                for exp, _ in exp_weights:
                    if exp not in exp_layer_counts:
                        exp_layer_counts[exp] = []
                    exp_layer_counts[exp].append(layer_weights.layer_idx)

        # Find dominant expert
        all_exp_counts = Counter()
        for _, exp_weights in layer_routing:
            for exp, _ in exp_weights:
                all_exp_counts[exp] += 1

        dominant_exp = None
        peak_layer = None
        if all_exp_counts:
            dominant_exp, _ = all_exp_counts.most_common(1)[0]
            layers_active = exp_layer_counts.get(dominant_exp, [])
            if layers_active:
                peak_layer = layers_active[len(layers_active) // 2]

        return DeepDiveResult(
            position=position,
            token=tok,
            token_type=curr_t,
            trigram=trigram,
            prev_token=prev_tok,
            prev_type=prev_t,
            next_token=next_tok,
            next_type=next_t,
            layer_routing=layer_routing,
            all_experts=sorted(all_experts),
            dominant_expert=dominant_exp,
            peak_layer=peak_layer,
        )


__all__ = [
    "ExploreService",
    "TokenAnalysis",
    "PatternMatch",
    "LayerPhaseData",
    "PositionEvolution",
    "ComparisonResult",
    "DeepDiveResult",
]
