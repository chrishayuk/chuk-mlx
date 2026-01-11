"""MoE-specific logit lens analysis.

Extends the base logit lens with MoE-specific analysis:
- Per-expert contribution to final logits
- Router decision evolution across layers
- Expert specialization through vocabulary analysis
- Token-to-expert preference mapping

Example:
    >>> from chuk_lazarus.introspection.moe import MoEHooks
    >>> from chuk_lazarus.introspection.moe.logit_lens import (
    ...     MoELogitLens,
    ...     analyze_expert_vocabulary,
    ...     compute_expert_vocab_contribution,
    ...     find_expert_specialists,
    ... )
    >>>
    >>> hooks = MoEHooks(model)
    >>> lens = MoELogitLens(hooks, tokenizer)
    >>>
    >>> # Get expert vocabulary contributions
    >>> contrib = compute_expert_vocab_contribution(model, tokenizer, layer_idx=10)
    >>> for exp in contrib.expert_contributions[:3]:
    ...     print(f"Expert {exp.expert_idx}: {exp.top_tokens[:5]}")
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .hooks import MoEHooks


class ExpertLogitContribution(BaseModel):
    """Contribution of a single expert to logit predictions."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    expert_idx: int = Field(ge=0)
    top_tokens: tuple[str, ...] = Field(default_factory=tuple)
    top_logits: tuple[float, ...] = Field(default_factory=tuple)
    top_token_ids: tuple[int, ...] = Field(default_factory=tuple)
    activation_weight: float = Field(ge=0, le=1)


class LayerRoutingSnapshot(BaseModel):
    """Routing snapshot at a layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0)
    selected_experts: tuple[int, ...] = Field(default_factory=tuple)
    expert_weights: tuple[float, ...] = Field(default_factory=tuple)
    router_entropy: float = Field(ge=0)
    top_token: str = ""
    top_token_prob: float = Field(ge=0, le=1, default=0.0)


class MoELogitLens:
    """
    MoE-specific logit lens analysis.

    Provides insight into how expert routing affects predictions
    and how different experts contribute to the output vocabulary.
    """

    def __init__(
        self,
        hooks: MoEHooks,
        tokenizer: Any | None = None,
    ):
        """
        Initialize MoE logit lens.

        Args:
            hooks: MoEHooks with captured state
            tokenizer: Tokenizer for decoding
        """
        self.hooks = hooks
        self.tokenizer = tokenizer

    def get_expert_contributions(
        self,
        layer_idx: int,
        position: int = -1,
        top_k: int = 10,
    ) -> list[ExpertLogitContribution]:
        """
        Analyze how each selected expert contributes to predictions.

        Args:
            layer_idx: Layer to analyze
            position: Sequence position
            top_k: Number of top tokens per expert

        Returns:
            List of ExpertLogitContribution for selected experts
        """
        if layer_idx not in self.hooks.moe_state.selected_experts:
            return []

        selected = self.hooks.moe_state.selected_experts[layer_idx]
        weights = self.hooks.moe_state.router_weights.get(layer_idx)

        if selected.ndim == 3:
            # [batch, seq, k] -> get position
            sel_at_pos = selected[0, position, :].tolist()
            if weights is not None:
                w_at_pos = weights.reshape(selected.shape)[0, position, :].tolist()
            else:
                w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)
        else:
            sel_at_pos = selected[position, :].tolist()
            w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)

        contributions = []
        for expert_idx, weight in zip(sel_at_pos, w_at_pos):
            # Get expert's vocabulary preference
            # This requires capturing expert outputs, which may not be available
            contributions.append(
                ExpertLogitContribution(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    top_tokens=(),
                    top_logits=(),
                    top_token_ids=(),
                    activation_weight=weight,
                )
            )

        return contributions

    def get_routing_evolution(
        self,
        position: int = -1,
    ) -> list[LayerRoutingSnapshot]:
        """
        Get routing decisions across all captured layers.

        Args:
            position: Sequence position

        Returns:
            List of LayerRoutingSnapshot, one per layer
        """
        snapshots = []

        for layer_idx in sorted(self.hooks.moe_state.selected_experts.keys()):
            selected = self.hooks.moe_state.selected_experts[layer_idx]
            weights = self.hooks.moe_state.router_weights.get(layer_idx)

            if selected.ndim == 3:
                sel_at_pos = selected[0, position, :].tolist()
                if weights is not None:
                    w_at_pos = weights.reshape(selected.shape)[0, position, :].tolist()
                else:
                    w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)
            else:
                sel_at_pos = selected[position, :].tolist()
                w_at_pos = [1.0 / len(sel_at_pos)] * len(sel_at_pos)

            # Compute entropy from router logits if available
            entropy = 0.0
            if layer_idx in self.hooks.moe_state.router_logits:
                logits = self.hooks.moe_state.router_logits[layer_idx]
                probs = mx.softmax(logits, axis=-1)
                log_probs = mx.log(probs + 1e-10)
                ent = -mx.sum(probs * log_probs, axis=-1)
                entropy = float(mx.mean(ent))

            # Get top prediction at this layer
            top_token = ""
            top_prob = 0.0

            snapshots.append(
                LayerRoutingSnapshot(
                    layer_idx=layer_idx,
                    selected_experts=tuple(sel_at_pos),
                    expert_weights=tuple(w_at_pos),
                    router_entropy=entropy,
                    top_token=top_token,
                    top_token_prob=top_prob,
                )
            )

        return snapshots

    def find_routing_divergence(
        self,
        position: int = -1,
    ) -> list[tuple[int, int, set[int]]]:
        """
        Find layers where routing changes significantly.

        Args:
            position: Sequence position

        Returns:
            List of (layer_a, layer_b, expert_difference) tuples
        """
        snapshots = self.get_routing_evolution(position)
        divergences = []

        for i in range(len(snapshots) - 1):
            a, b = snapshots[i], snapshots[i + 1]
            set_a = set(a.selected_experts)
            set_b = set(b.selected_experts)

            if set_a != set_b:
                diff = set_a.symmetric_difference(set_b)
                divergences.append((a.layer_idx, b.layer_idx, diff))

        return divergences

    def print_routing_evolution(self, position: int = -1) -> None:
        """Print routing evolution in human-readable format."""
        snapshots = self.get_routing_evolution(position)

        if not snapshots:
            print("No routing data captured")
            return

        print(f"\nMoE Routing Evolution (position {position})")
        print("=" * 60)

        for snap in snapshots:
            experts_str = ", ".join(
                f"E{e}({w:.2f})" for e, w in zip(snap.selected_experts, snap.expert_weights)
            )
            print(f"Layer {snap.layer_idx:2d}: [{experts_str}] entropy={snap.router_entropy:.3f}")


def analyze_expert_vocabulary(
    model: nn.Module,
    layer_idx: int,
    expert_idx: int,
    tokenizer: Any,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Analyze what vocabulary an expert specializes in.

    This examines the expert's output projection to find
    which tokens it most strongly promotes.

    Args:
        model: The model
        layer_idx: Layer index
        expert_idx: Expert index
        tokenizer: Tokenizer
        top_k: Number of top tokens

    Returns:
        Dict with vocabulary analysis
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return {"error": "layer out of range"}

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return {"error": "no mlp"}

    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        return {"error": "no experts list"}

    if expert_idx >= len(experts):
        return {"error": "expert out of range"}

    expert = experts[expert_idx]
    down_proj = getattr(expert, "down_proj", None)
    if down_proj is None:
        return {"error": "no down_proj"}

    # Get the output weight
    weight = down_proj.weight  # [hidden, intermediate]

    # Compute which output dimensions have strongest weights
    # This is a proxy for vocabulary preference
    output_norms = mx.linalg.norm(weight, axis=1)
    top_dims = mx.argsort(output_norms)[::-1][:top_k].tolist()

    return {
        "expert_idx": expert_idx,
        "layer_idx": layer_idx,
        "top_output_dimensions": top_dims,
        "dimension_norms": output_norms[top_dims[:10]].tolist(),
    }


def _get_model_layers(model: nn.Module) -> list[nn.Module]:
    """Get transformer layers from model."""
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            layers = getattr(submodel, "layers", None)
            if layers is not None:
                return list(layers)
    return list(getattr(model, "layers", []))


# =============================================================================
# Expert Vocabulary Contribution Models
# =============================================================================


class ExpertVocabContribution(BaseModel):
    """Vocabulary contribution analysis for a single expert."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(ge=0, description="Expert index")
    layer_idx: int = Field(ge=0, description="Layer index")
    top_tokens: tuple[str, ...] = Field(
        default_factory=tuple, description="Tokens this expert most strongly promotes"
    )
    top_token_ids: tuple[int, ...] = Field(
        default_factory=tuple, description="Token IDs for top tokens"
    )
    top_scores: tuple[float, ...] = Field(
        default_factory=tuple, description="Scores for top tokens (projection norm)"
    )
    vocab_entropy: float = Field(
        ge=0, default=0.0, description="Entropy of expert's vocabulary preference"
    )
    specialization_score: float = Field(
        ge=0,
        le=1,
        default=0.0,
        description="How specialized (vs. generalist) this expert is",
    )
    dominant_categories: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Inferred token categories (numbers, punctuation, etc.)",
    )


class LayerVocabAnalysis(BaseModel):
    """Vocabulary contribution analysis for all experts in a layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    num_experts: int = Field(ge=1, description="Number of experts")
    expert_contributions: tuple[ExpertVocabContribution, ...] = Field(
        default_factory=tuple, description="Per-expert vocabulary contributions"
    )
    vocab_coverage: float = Field(
        ge=0,
        le=1,
        default=0.0,
        description="Fraction of vocabulary covered by top-k per expert",
    )
    expert_overlap: float = Field(
        ge=0,
        le=1,
        default=0.0,
        description="Average overlap between expert vocabularies",
    )


class TokenExpertPreference(BaseModel):
    """Which experts prefer a specific token."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(description="Token text")
    token_id: int = Field(ge=0, description="Token ID")
    preferred_experts: tuple[int, ...] = Field(
        default_factory=tuple, description="Experts that most prefer this token"
    )
    preference_scores: tuple[float, ...] = Field(
        default_factory=tuple, description="Preference scores for each expert"
    )


class VocabExpertMapping(BaseModel):
    """Complete vocabulary-to-expert mapping for a layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(ge=0, description="Layer index")
    num_experts: int = Field(ge=1, description="Number of experts")
    num_tokens: int = Field(ge=0, description="Vocabulary size analyzed")
    token_preferences: tuple[TokenExpertPreference, ...] = Field(
        default_factory=tuple, description="Per-token expert preferences"
    )
    expert_vocab_sizes: tuple[int, ...] = Field(
        default_factory=tuple, description="Number of tokens each expert 'owns'"
    )


# =============================================================================
# Expert Vocabulary Contribution Functions
# =============================================================================


def compute_expert_vocab_contribution(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    top_k: int = 50,
    vocab_sample_size: int | None = None,
) -> LayerVocabAnalysis:
    """
    Compute vocabulary contribution for each expert in a layer.

    This analyzes how each expert's output projection weights map to
    the vocabulary, revealing which tokens each expert specializes in.

    Args:
        model: The MoE model
        tokenizer: Tokenizer for decoding tokens
        layer_idx: Layer to analyze
        top_k: Number of top tokens per expert
        vocab_sample_size: If set, sample this many tokens from vocabulary

    Returns:
        LayerVocabAnalysis with per-expert vocabulary contributions
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return LayerVocabAnalysis(
            layer_idx=layer_idx,
            num_experts=1,
            expert_contributions=(),
            vocab_coverage=0.0,
            expert_overlap=0.0,
        )

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return LayerVocabAnalysis(
            layer_idx=layer_idx,
            num_experts=1,
            expert_contributions=(),
        )

    # Get experts
    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        return LayerVocabAnalysis(
            layer_idx=layer_idx,
            num_experts=1,
            expert_contributions=(),
        )

    num_experts = len(experts)

    # Get language model head for projecting to vocabulary
    lm_head = _get_lm_head(model)
    if lm_head is None:
        return LayerVocabAnalysis(
            layer_idx=layer_idx,
            num_experts=num_experts,
            expert_contributions=(),
        )

    lm_weight = lm_head.weight  # [vocab, hidden]
    vocab_size = lm_weight.shape[0]

    # Sample vocabulary if too large
    if vocab_sample_size is not None and vocab_size > vocab_sample_size:
        import random

        sample_indices = sorted(random.sample(range(vocab_size), vocab_sample_size))
        lm_weight_sample = lm_weight[sample_indices]
        token_ids = sample_indices
    else:
        lm_weight_sample = lm_weight
        token_ids = list(range(vocab_size))

    expert_contributions: list[ExpertVocabContribution] = []
    all_expert_top_tokens: list[set[int]] = []

    for expert_idx, expert in enumerate(experts):
        # Get expert's down projection (output)
        down_proj = getattr(expert, "down_proj", None)
        if down_proj is None:
            continue

        down_weight = down_proj.weight  # [hidden, intermediate]

        # Compute how each vocabulary token aligns with this expert's output
        # Score = ||lm_head_row · down_proj||
        # This measures how much this expert can influence each vocabulary token
        expert_vocab_scores = _compute_vocab_scores(down_weight, lm_weight_sample)

        # Get top-k tokens for this expert
        top_indices = mx.argsort(expert_vocab_scores)[::-1][:top_k].tolist()
        top_scores_list = [float(expert_vocab_scores[i]) for i in top_indices]
        top_token_ids_list = [token_ids[i] for i in top_indices]

        # Decode tokens
        top_tokens = []
        for tid in top_token_ids_list:
            try:
                decoded = tokenizer.decode([tid])
                top_tokens.append(decoded)
            except Exception:
                top_tokens.append(f"[{tid}]")

        # Compute vocabulary entropy for specialization
        vocab_probs = mx.softmax(expert_vocab_scores)
        log_probs = mx.log(vocab_probs + 1e-10)
        entropy = float(-mx.sum(vocab_probs * log_probs))

        # Max entropy for uniform distribution
        max_entropy = float(mx.log(mx.array(len(token_ids))))
        specialization = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        # Categorize top tokens
        categories = _categorize_tokens(top_tokens[:20])

        expert_contributions.append(
            ExpertVocabContribution(
                expert_idx=expert_idx,
                layer_idx=layer_idx,
                top_tokens=tuple(top_tokens),
                top_token_ids=tuple(top_token_ids_list),
                top_scores=tuple(top_scores_list),
                vocab_entropy=entropy,
                specialization_score=min(1.0, max(0.0, specialization)),
                dominant_categories=tuple(categories),
            )
        )

        all_expert_top_tokens.append(set(top_token_ids_list))

    # Compute coverage and overlap
    if all_expert_top_tokens:
        all_covered = set()
        for tokens in all_expert_top_tokens:
            all_covered.update(tokens)
        vocab_coverage = len(all_covered) / len(token_ids) if token_ids else 0.0

        # Average pairwise overlap
        overlaps = []
        for i, tokens_i in enumerate(all_expert_top_tokens):
            for j, tokens_j in enumerate(all_expert_top_tokens):
                if i < j:
                    intersection = len(tokens_i & tokens_j)
                    union = len(tokens_i | tokens_j)
                    overlaps.append(intersection / union if union > 0 else 0.0)

        expert_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    else:
        vocab_coverage = 0.0
        expert_overlap = 0.0

    return LayerVocabAnalysis(
        layer_idx=layer_idx,
        num_experts=num_experts,
        expert_contributions=tuple(expert_contributions),
        vocab_coverage=vocab_coverage,
        expert_overlap=expert_overlap,
    )


def compute_token_expert_mapping(
    model: nn.Module,
    tokenizer: Any,
    layer_idx: int,
    tokens_to_analyze: list[str] | None = None,
    top_experts_per_token: int = 3,
) -> VocabExpertMapping:
    """
    Compute which experts prefer specific tokens.

    This is the inverse of compute_expert_vocab_contribution - instead
    of asking "what tokens does each expert prefer?", we ask
    "which experts prefer each token?"

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        layer_idx: Layer to analyze
        tokens_to_analyze: Specific tokens to analyze (if None, uses common tokens)
        top_experts_per_token: Number of top experts per token

    Returns:
        VocabExpertMapping with per-token expert preferences
    """
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return VocabExpertMapping(
            layer_idx=layer_idx,
            num_experts=1,
            num_tokens=0,
            token_preferences=(),
            expert_vocab_sizes=(),
        )

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return VocabExpertMapping(
            layer_idx=layer_idx,
            num_experts=1,
            num_tokens=0,
        )

    experts = getattr(mlp, "experts", None)
    if experts is None or not isinstance(experts, list):
        return VocabExpertMapping(
            layer_idx=layer_idx,
            num_experts=1,
            num_tokens=0,
        )

    num_experts = len(experts)

    # Default tokens to analyze
    if tokens_to_analyze is None:
        tokens_to_analyze = [
            # Common words
            "the",
            "a",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            # Punctuation
            ".",
            ",",
            "!",
            "?",
            ":",
            ";",
            "(",
            ")",
            # Numbers
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            # Code tokens
            "def",
            "class",
            "return",
            "if",
            "else",
            "for",
            # Math
            "+",
            "-",
            "*",
            "/",
            "=",
        ]

    # Encode tokens
    token_ids_map: dict[str, int] = {}
    for tok in tokens_to_analyze:
        try:
            encoded = tokenizer.encode(tok)
            if encoded:
                token_ids_map[tok] = encoded[0] if isinstance(encoded, list) else encoded
        except Exception:
            pass

    if not token_ids_map:
        return VocabExpertMapping(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_tokens=0,
            token_preferences=(),
            expert_vocab_sizes=(),
        )

    # Get LM head
    lm_head = _get_lm_head(model)
    if lm_head is None:
        return VocabExpertMapping(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_tokens=len(token_ids_map),
        )

    lm_weight = lm_head.weight

    # Compute per-expert scores for each token
    token_preferences: list[TokenExpertPreference] = []
    expert_token_counts: dict[int, int] = defaultdict(int)

    for token, token_id in token_ids_map.items():
        token_lm_row = lm_weight[token_id]  # [hidden]

        expert_scores = []
        for expert_idx, expert in enumerate(experts):
            down_proj = getattr(expert, "down_proj", None)
            if down_proj is None:
                expert_scores.append(0.0)
                continue

            down_weight = down_proj.weight  # [hidden, intermediate]

            # Score: how well can this expert influence this token?
            # Use the max projection along the intermediate dimension
            projections = mx.abs(down_weight @ token_lm_row)
            score = float(mx.max(projections))
            expert_scores.append(score)

        # Normalize and get top experts
        if max(expert_scores) > 0:
            normalized = [s / max(expert_scores) for s in expert_scores]
        else:
            normalized = expert_scores

        sorted_experts = sorted(enumerate(normalized), key=lambda x: x[1], reverse=True)[
            :top_experts_per_token
        ]

        preferred_experts = [e[0] for e in sorted_experts]
        preference_scores = [e[1] for e in sorted_experts]

        # Track which expert "owns" this token
        if preferred_experts:
            expert_token_counts[preferred_experts[0]] += 1

        token_preferences.append(
            TokenExpertPreference(
                token=token,
                token_id=token_id,
                preferred_experts=tuple(preferred_experts),
                preference_scores=tuple(preference_scores),
            )
        )

    # Compute vocab sizes per expert
    expert_vocab_sizes = tuple(expert_token_counts.get(i, 0) for i in range(num_experts))

    return VocabExpertMapping(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_tokens=len(token_ids_map),
        token_preferences=tuple(token_preferences),
        expert_vocab_sizes=expert_vocab_sizes,
    )


def find_expert_specialists(
    analysis: LayerVocabAnalysis,
    min_specialization: float = 0.3,
) -> list[tuple[int, str, float]]:
    """
    Find experts that specialize in specific vocabulary categories.

    Args:
        analysis: LayerVocabAnalysis from compute_expert_vocab_contribution
        min_specialization: Minimum specialization score

    Returns:
        List of (expert_idx, category, specialization_score)
    """
    specialists = []

    for contrib in analysis.expert_contributions:
        if contrib.specialization_score >= min_specialization:
            primary_category = (
                contrib.dominant_categories[0] if contrib.dominant_categories else "general"
            )
            specialists.append(
                (
                    contrib.expert_idx,
                    primary_category,
                    contrib.specialization_score,
                )
            )

    specialists.sort(key=lambda x: x[2], reverse=True)
    return specialists


def print_expert_vocab_summary(analysis: LayerVocabAnalysis) -> None:
    """Print a summary of expert vocabulary contributions."""
    print(f"\nExpert Vocabulary Contributions - Layer {analysis.layer_idx}")
    print("=" * 70)
    print(f"Experts: {analysis.num_experts}")
    print(f"Vocabulary Coverage: {analysis.vocab_coverage:.1%}")
    print(f"Expert Overlap: {analysis.expert_overlap:.1%}")
    print("-" * 70)

    for contrib in analysis.expert_contributions:
        spec_bar = "█" * int(contrib.specialization_score * 10)
        spec_bar += "░" * (10 - len(spec_bar))

        categories = ", ".join(contrib.dominant_categories[:3]) or "mixed"

        print(f"\nExpert {contrib.expert_idx}:")
        print(f"  Specialization: [{spec_bar}] {contrib.specialization_score:.2f}")
        print(f"  Categories: {categories}")
        print(f"  Top tokens: {' '.join(repr(t) for t in contrib.top_tokens[:8])}")


def print_token_expert_preferences(mapping: VocabExpertMapping) -> None:
    """Print token-to-expert preference mapping."""
    print(f"\nToken-Expert Preferences - Layer {mapping.layer_idx}")
    print("=" * 60)
    print(f"Tokens analyzed: {mapping.num_tokens}")
    print(f"Experts: {mapping.num_experts}")
    print("-" * 60)

    # Group by dominant expert
    expert_tokens: dict[int, list[str]] = defaultdict(list)

    for pref in mapping.token_preferences:
        if pref.preferred_experts:
            expert_tokens[pref.preferred_experts[0]].append(pref.token)

    for expert_idx in range(mapping.num_experts):
        tokens = expert_tokens.get(expert_idx, [])
        if tokens:
            tokens_str = " ".join(repr(t) for t in tokens[:10])
            more = f" (+{len(tokens) - 10} more)" if len(tokens) > 10 else ""
            print(f"Expert {expert_idx}: {tokens_str}{more}")
        else:
            print(f"Expert {expert_idx}: (no dominant tokens)")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_lm_head(model: nn.Module) -> nn.Module | None:
    """Get the language model head from model."""
    # Try common names
    for attr in ["lm_head", "output", "head"]:
        head = getattr(model, attr, None)
        if head is not None and hasattr(head, "weight"):
            return head

    # Try tied embeddings
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            embed = getattr(submodel, "embed_tokens", None)
            if embed is not None and hasattr(embed, "weight"):
                return embed

    return None


def _compute_vocab_scores(
    down_weight: mx.array,
    lm_weight: mx.array,
) -> mx.array:
    """
    Compute vocabulary scores for an expert.

    Args:
        down_weight: Expert's down projection [hidden, intermediate]
        lm_weight: LM head weights [vocab, hidden]

    Returns:
        Scores for each vocabulary token [vocab]
    """
    # Compute influence: how much can this expert affect each vocab token?
    # Approach: For each vocab token, compute the max projection through expert

    # Efficient: Compute LM × down_proj^T, then take row norms
    # Result: [vocab, intermediate]
    combined = lm_weight @ down_weight

    # Take L2 norm along intermediate dimension -> [vocab]
    scores = mx.linalg.norm(combined, axis=1)

    return scores


def _categorize_tokens(tokens: list[str]) -> list[str]:
    """Categorize tokens into categories."""
    categories: dict[str, int] = defaultdict(int)

    for token in tokens:
        token_stripped = token.strip()

        if not token_stripped:
            categories["whitespace"] += 1
        elif token_stripped.isdigit():
            categories["numbers"] += 1
        elif token_stripped.isalpha():
            if token_stripped.isupper():
                categories["uppercase"] += 1
            elif token_stripped.islower():
                categories["lowercase"] += 1
            else:
                categories["mixed_case"] += 1
        elif all(c in ".,!?;:'\"-()[]{}/" for c in token_stripped):
            categories["punctuation"] += 1
        elif any(c in "+-*/=<>^%&|" for c in token_stripped):
            categories["operators"] += 1
        else:
            categories["mixed"] += 1

    # Return top categories
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in sorted_cats[:3]]
