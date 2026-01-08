"""Service layer for MoE expert analysis CLI commands.

This module provides the MoEAnalysisService class that provides
functionality for analyzing MoE expert routing and patterns.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

from ...cli.commands._constants import (
    LayerPhase,
    LayerPhaseDefaults,
    PatternCategory,
    TokenType,
)
from . import ExpertRouter
from .test_data import (
    ANSWER_WORDS,
    BOOLEAN_LITERALS,
    CAUSATION_WORDS,
    CODE_KEYWORDS,
    COMPARISON_WORDS,
    CONDITIONAL_WORDS,
    COORDINATION_WORDS,
    NEGATION_WORDS,
    QUANTIFIER_WORDS,
    QUESTION_WORDS,
    TIME_WORDS,
    TYPE_KEYWORDS,
)


# =============================================================================
# Result Models
# =============================================================================


class ExpertWeightInfo(BaseModel):
    """Expert weight information."""

    model_config = ConfigDict(frozen=True)

    expert_idx: int = Field(..., description="Expert index")
    weight: float = Field(..., description="Routing weight")


class PositionRoutingInfo(BaseModel):
    """Routing information for a token position."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(..., description="Token position")
    token: str = Field(..., description="Token string")
    token_type: str = Field(..., description="Semantic token type")
    trigram: str = Field(..., description="Trigram pattern")
    experts: list[ExpertWeightInfo] = Field(default_factory=list, description="Expert routing")


class LayerRoutingInfo(BaseModel):
    """Routing information for a layer."""

    model_config = ConfigDict(frozen=True)

    layer_idx: int = Field(..., description="Layer index")
    positions: list[PositionRoutingInfo] = Field(default_factory=list)


class AttentionCaptureResult(BaseModel):
    """Result of attention weight capture."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    layer: int = Field(..., description="Layer index")
    query_position: int = Field(..., description="Query position")
    query_token: str = Field(..., description="Query token")
    attention_weights: list[tuple[int, float]] = Field(
        default_factory=list, description="Position-weight pairs sorted by weight"
    )
    self_attention: float = Field(..., description="Self-attention weight")


class TaxonomyExpertMapping(BaseModel):
    """Mapping of pattern categories to experts."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="Pattern category")
    layer: int = Field(..., description="Layer index")
    experts: list[int] = Field(default_factory=list, description="Expert indices")
    trigrams: list[str] = Field(default_factory=list, description="Trigram patterns")


# =============================================================================
# Token Classification
# =============================================================================


def classify_token(token: str) -> TokenType:
    """Classify a token into a semantic type.

    Args:
        token: Token string to classify.

    Returns:
        TokenType enum value.
    """
    # Whitespace
    if not token.strip():
        return TokenType.WS

    # Numbers
    if token.strip().isdigit():
        return TokenType.NUM

    # Operators
    if token.strip() in {"+", "-", "*", "/", "=", "<", ">", "^", "%"}:
        return TokenType.OP

    # Brackets
    if token.strip() in {"(", ")", "[", "]", "{", "}", "<", ">"}:
        return TokenType.BR

    # Punctuation
    if token.strip() in {".", ",", ":", ";", "!", "?", "-", "'"}:
        return TokenType.PN

    # Quotes
    if token.strip() in {'"', "'", "`", "'''", '"""'}:
        return TokenType.QUOTE

    # Lowercase for keyword checks
    lower = token.strip().lower()

    # Code keywords
    if lower in CODE_KEYWORDS:
        return TokenType.KW

    # Boolean literals
    if token.strip() in BOOLEAN_LITERALS:
        return TokenType.BOOL

    # Type keywords
    if token.strip() in TYPE_KEYWORDS:
        return TokenType.TYPE

    # Question words
    if lower in QUESTION_WORDS:
        return TokenType.QW

    # Answer words
    if lower in ANSWER_WORDS:
        return TokenType.ANS

    # Negation
    if lower in NEGATION_WORDS:
        return TokenType.NEG

    # Time words
    if lower in TIME_WORDS:
        return TokenType.TIME

    # Quantifiers
    if lower in QUANTIFIER_WORDS:
        return TokenType.QUANT

    # Comparison
    if lower in COMPARISON_WORDS:
        return TokenType.COMP

    # Coordination
    if lower in COORDINATION_WORDS:
        return TokenType.COORD

    # Causation
    if lower in CAUSATION_WORDS:
        return TokenType.CAUSE

    # Conditional
    if lower in CONDITIONAL_WORDS:
        return TokenType.COND

    # Special markers
    if lower == "as":
        return TokenType.AS
    if lower == "to":
        return TokenType.TO
    if lower == "than":
        return TokenType.THAN

    # Synonym/antonym markers
    if lower in {"like", "similar", "same", "means", "equals"}:
        return TokenType.SYN
    if lower in {"opposite", "versus", "unlike", "contrasts", "but"}:
        return TokenType.ANT

    # Capitalized (proper noun)
    if token[0].isupper() and len(token) > 1:
        return TokenType.CAP

    # Default: content word
    return TokenType.CW


def get_trigram(
    tokens: list[str],
    position: int,
) -> str:
    """Get the trigram pattern for a position.

    Args:
        tokens: List of token strings.
        position: Position to get trigram for.

    Returns:
        Trigram pattern string (prev→curr→next).
    """
    prev_type = "^" if position == 0 else classify_token(tokens[position - 1]).value
    curr_type = classify_token(tokens[position]).value
    next_type = "$" if position >= len(tokens) - 1 else classify_token(tokens[position + 1]).value

    return f"{prev_type}→{curr_type}→{next_type}"


def get_layer_phase(layer: int) -> LayerPhase:
    """Determine the phase of a layer.

    Args:
        layer: Layer index.

    Returns:
        LayerPhase enum value.
    """
    if layer < LayerPhaseDefaults.EARLY_END:
        return LayerPhase.EARLY
    elif layer < LayerPhaseDefaults.MIDDLE_END:
        return LayerPhase.MIDDLE
    else:
        return LayerPhase.LATE


# =============================================================================
# Service Class
# =============================================================================


class MoEAnalysisServiceConfig(BaseModel):
    """Configuration for MoEAnalysisService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")


class MoEAnalysisService:
    """Service class for MoE expert analysis.

    Provides a high-level interface for CLI commands to analyze MoE models
    without needing to understand the internal architecture.
    """

    Config = MoEAnalysisServiceConfig

    @classmethod
    async def capture_router_weights(
        cls,
        model: str,
        prompt: str,
        layers: list[int] | None = None,
    ) -> list[LayerRoutingInfo]:
        """Capture router weights for a prompt.

        Args:
            model: Model path or name.
            prompt: Prompt to analyze.
            layers: Specific layers to capture (default: all MoE layers).

        Returns:
            List of LayerRoutingInfo for each layer.
        """
        async with await ExpertRouter.from_pretrained(model) as router:
            info = router.info
            target_layers = layers or list(info.moe_layers)

            # Tokenize
            tokens = [router.tokenizer.decode([t]) for t in router.tokenizer.encode(prompt)]

            # Capture weights
            weights_list = await router.capture_router_weights(prompt, layers=target_layers)

            results = []
            for layer_weights in weights_list:
                positions = []
                for i, pos in enumerate(layer_weights.positions):
                    token = tokens[i] if i < len(tokens) else ""
                    token_type = classify_token(token)
                    trigram = get_trigram(tokens, i)

                    experts = [
                        ExpertWeightInfo(expert_idx=idx, weight=w)
                        for idx, w in zip(pos.expert_indices, pos.weights)
                    ]

                    positions.append(
                        PositionRoutingInfo(
                            position=i,
                            token=token,
                            token_type=token_type.value,
                            trigram=trigram,
                            experts=experts,
                        )
                    )

                results.append(
                    LayerRoutingInfo(layer_idx=layer_weights.layer, positions=positions)
                )

            return results

    @classmethod
    async def capture_attention_weights(
        cls,
        model: str,
        prompt: str,
        layer: int,
        query_position: int | None = None,
        head: int | None = None,
        top_k: int = 5,
    ) -> AttentionCaptureResult:
        """Capture attention weights for a specific position.

        Args:
            model: Model path or name.
            prompt: Prompt to analyze.
            layer: Layer to capture attention from.
            query_position: Position to analyze (default: last).
            head: Specific head to use (default: average across heads).
            top_k: Number of top attention positions to return.

        Returns:
            AttentionCaptureResult with attention weights.
        """
        async with await ExpertRouter.from_pretrained(model) as router:
            # Tokenize
            input_ids = mx.array(router.tokenizer.encode(prompt))[None, :]
            tokens = [router.tokenizer.decode([t]) for t in input_ids[0].tolist()]

            # Determine query position
            if query_position is None:
                query_pos = len(tokens) - 1
            elif query_position < 0:
                query_pos = len(tokens) + query_position
            else:
                query_pos = min(query_position, len(tokens) - 1)

            # Get attention layer
            target_block = router._model.model.layers[layer]
            attn = target_block.self_attn
            attn_class = type(attn)
            original_call = attn_class.__call__

            captured_qk: dict[int, tuple[mx.array, mx.array]] = {}

            def patched_attn_call(attn_self, x, mask=None, cache=None):
                batch, seq_len, _ = x.shape
                q = attn_self.q_proj(x)
                k = attn_self.k_proj(x)

                q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
                k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)

                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)

                if cache is not None:
                    q = attn_self.rope(q, offset=cache[0].shape[2])
                    k = attn_self.rope(k, offset=cache[0].shape[2])
                else:
                    q = attn_self.rope(q)
                    k = attn_self.rope(k)

                captured_qk[layer] = (q, k)
                return original_call(attn_self, x, mask=mask, cache=cache)

            try:
                attn_class.__call__ = patched_attn_call
                router._model(input_ids)
            finally:
                attn_class.__call__ = original_call

            if layer not in captured_qk:
                raise RuntimeError(f"Could not capture attention for layer {layer}")

            q, k = captured_qk[layer]

            # Handle GQA
            num_heads = q.shape[1]
            num_kv_heads = k.shape[1]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                k = mx.repeat(k, repeat_factor, axis=1)

            # Compute attention scores
            head_dim = q.shape[-1]
            scale = 1.0 / math.sqrt(head_dim)
            attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask
            seq_len = attn_scores.shape[-1]
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
            attn_scores = attn_scores + causal_mask

            # Softmax
            attn_weights = mx.softmax(attn_scores, axis=-1)

            # Get weights for query position
            query_attn = attn_weights[0, :, query_pos, :]

            # Aggregate across heads or select specific head
            if head is not None:
                head_idx = min(head, num_heads - 1)
                attn_for_pos = query_attn[head_idx]
            else:
                attn_for_pos = mx.mean(query_attn, axis=0)

            # Get top-k
            attn_list = attn_for_pos.tolist()
            indexed = list(enumerate(attn_list))
            sorted_attn = sorted(indexed, key=lambda x: x[1], reverse=True)

            return AttentionCaptureResult(
                layer=layer,
                query_position=query_pos,
                query_token=tokens[query_pos],
                attention_weights=sorted_attn[:top_k],
                self_attention=attn_list[query_pos],
            )

    @classmethod
    async def analyze_domain_routing(
        cls,
        model: str,
        domain_prompts: dict[str, list[str]],
        layers: list[int] | None = None,
    ) -> dict[int, dict[str, list[int]]]:
        """Analyze which experts handle which domains.

        Args:
            model: Model path or name.
            domain_prompts: Dict mapping domain -> list of prompts.
            layers: Layers to analyze (default: all MoE layers).

        Returns:
            Dict mapping layer -> domain -> list of primary experts.
        """
        async with await ExpertRouter.from_pretrained(model) as router:
            info = router.info
            target_layers = layers or list(info.moe_layers)

            results: dict[int, dict[str, list[int]]] = {l: {} for l in target_layers}

            for domain, prompts in domain_prompts.items():
                for layer in target_layers:
                    if domain not in results[layer]:
                        results[layer][domain] = []

                    for prompt in prompts:
                        weights_list = await router.capture_router_weights(prompt, layers=[layer])
                        if weights_list and weights_list[0].positions:
                            # Get primary expert from last position
                            last_pos = weights_list[0].positions[-1]
                            primary = last_pos.expert_indices[0]
                            results[layer][domain].append(primary)

            return results

    @classmethod
    async def get_model_info(cls, model: str) -> dict[str, Any]:
        """Get basic model info.

        Args:
            model: Model path or name.

        Returns:
            Dict with model info.
        """
        async with await ExpertRouter.from_pretrained(model) as router:
            info = router.info
            return {
                "model": model,
                "num_experts": info.num_experts,
                "top_k": info.top_k,
                "total_layers": info.total_layers,
                "moe_layers": list(info.moe_layers),
            }


__all__ = [
    "MoEAnalysisService",
    "MoEAnalysisServiceConfig",
    "ExpertWeightInfo",
    "PositionRoutingInfo",
    "LayerRoutingInfo",
    "AttentionCaptureResult",
    "TaxonomyExpertMapping",
    "classify_token",
    "get_trigram",
    "get_layer_phase",
]
