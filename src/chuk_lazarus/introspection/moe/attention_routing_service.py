"""Service layer for attention-routing analysis.

Provides business logic for analyzing how attention patterns drive expert routing.
This separates CLI concerns from the core attention capture and analysis algorithms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, computed_field

if TYPE_CHECKING:
    from .expert_router import ExpertRouter


class AttentionCaptureResult(BaseModel):
    """Result of capturing attention weights for a prompt."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tokens: list[str] = Field(..., description="Token strings")
    attention_weights: mx.array | None = Field(
        default=None, description="Attention weights (num_heads, seq_len, seq_len)"
    )
    layer: int = Field(..., description="Layer index")

    @computed_field
    @property
    def success(self) -> bool:
        """Whether attention weights were successfully captured."""
        return self.attention_weights is not None


class AttentionSummary(BaseModel):
    """Summary of attention pattern for a position."""

    model_config = ConfigDict(frozen=True)

    top_attended: list[tuple[str, float]] = Field(..., description="(token, weight) pairs")
    self_attention_weight: float = Field(..., description="Self-attention weight")


class ContextRoutingResult(BaseModel):
    """Result of analyzing routing for a single context."""

    model_config = ConfigDict(frozen=True)

    context_name: str = Field(..., description="Name of the context")
    context: str = Field(..., description="Context text")
    tokens: list[str] = Field(..., description="Token strings")
    target_pos: int = Field(..., description="Target position")
    target_token: str = Field(..., description="Target token")
    primary_expert: int = Field(..., description="Primary expert index")
    all_experts: list[int] = Field(..., description="All selected experts")
    weights: list[float] = Field(..., description="Expert weights")
    attention_summary: AttentionSummary | None = Field(
        default=None, description="Attention summary"
    )


class LayerRoutingResults(BaseModel):
    """Results for a single layer across all contexts."""

    model_config = ConfigDict(frozen=True)

    layer: int = Field(..., description="Layer index")
    label: str = Field(..., description="Layer phase label (Early, Middle, Late)")
    results: list[ContextRoutingResult] = Field(
        default_factory=list, description="Routing results per context"
    )

    @computed_field
    @property
    def unique_expert_count(self) -> int:
        """Number of unique primary experts across contexts."""
        return len({r.primary_expert for r in self.results})

    @computed_field
    @property
    def is_context_sensitive(self) -> bool:
        """Whether this layer shows context sensitivity."""
        return self.unique_expert_count > 1


class AttentionRoutingAnalysis(BaseModel):
    """Complete analysis results across layers."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Model identifier")
    target_token: str = Field(..., description="Target token being analyzed")
    layers: list[LayerRoutingResults] = Field(..., description="Layer results")

    @computed_field
    @property
    def early_layer(self) -> LayerRoutingResults | None:
        """Get early layer results."""
        return self.layers[0] if self.layers else None

    @computed_field
    @property
    def middle_layer(self) -> LayerRoutingResults | None:
        """Get middle layer results."""
        if len(self.layers) >= 2:
            return self.layers[len(self.layers) // 2]
        return self.layers[0] if self.layers else None

    @computed_field
    @property
    def late_layer(self) -> LayerRoutingResults | None:
        """Get late layer results."""
        return self.layers[-1] if self.layers else None


# Default test contexts
DEFAULT_ATTENTION_CONTEXTS: list[tuple[str, str]] = [
    ("minimal", "2 + 3"),
    ("instruction", "Calculate: 2 + 3"),
    ("sentence", "The sum is 2 + 3"),
    ("code", "result = 2 + 3"),
]


class AttentionRoutingService:
    """Service for attention-routing analysis."""

    @staticmethod
    def capture_attention_weights(
        router: ExpertRouter,
        prompt: str,
        target_layer: int,
    ) -> AttentionCaptureResult:
        """Capture attention weights for a prompt at a specific layer.

        This patches the attention layer to capture Q and K projections,
        then computes attention weights.

        Args:
            router: The ExpertRouter instance.
            prompt: The input prompt.
            target_layer: The layer index to capture attention from.

        Returns:
            AttentionCaptureResult with tokens and attention weights.
        """
        input_ids = mx.array(router.tokenizer.encode(prompt))[None, :]
        tokens = [router.tokenizer.decode([t]) for t in input_ids[0].tolist()]

        # Storage for captured Q, K
        captured_qk: dict[int, tuple[mx.array, mx.array]] = {}

        # Get the attention layer for the target block
        target_block = router._model.model.layers[target_layer]
        attn = target_block.self_attn
        attn_class = type(attn)
        original_call = attn_class.__call__

        def patched_attn_call(
            attn_self: Any, x: mx.array, mask: Any = None, cache: Any = None
        ) -> Any:
            """Patch to capture Q and K."""
            batch, seq_len, _ = x.shape

            # Project Q, K
            q = attn_self.q_proj(x)
            k = attn_self.k_proj(x)

            # Reshape to (batch, seq_len, num_heads, head_dim)
            q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
            k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)

            # Transpose to (batch, num_heads, seq_len, head_dim)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)

            # Apply RoPE
            if cache is not None:
                q = attn_self.rope(q, offset=cache[0].shape[2])
                k = attn_self.rope(k, offset=cache[0].shape[2])
            else:
                q = attn_self.rope(q)
                k = attn_self.rope(k)

            # Store the Q, K for later analysis
            captured_qk[target_layer] = (q, k)

            # Call original
            return original_call(attn_self, x, mask=mask, cache=cache)

        try:
            attn_class.__call__ = patched_attn_call
            router._model(input_ids)
        finally:
            attn_class.__call__ = original_call

        if target_layer not in captured_qk:
            return AttentionCaptureResult(tokens=tokens, attention_weights=None, layer=target_layer)

        q, k = captured_qk[target_layer]

        # Handle GQA (Grouped Query Attention)
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
        attn_weights = mx.softmax(attn_scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)

        return AttentionCaptureResult(
            tokens=tokens,
            attention_weights=attn_weights[0],  # Remove batch dim
            layer=target_layer,
        )

    @staticmethod
    def compute_attention_summary(
        attn_weights: mx.array,
        tokens: list[str],
        position: int,
        top_k: int = 3,
    ) -> AttentionSummary:
        """Compute attention summary for a specific position.

        Args:
            attn_weights: Attention weights (num_heads, seq_len, seq_len).
            tokens: List of token strings.
            position: The query position to analyze.
            top_k: Number of top attended tokens to return.

        Returns:
            AttentionSummary with top attended tokens.
        """
        # Average attention across heads for this position
        pos_attn = mx.mean(attn_weights[:, position, :], axis=0).tolist()

        # Get self-attention weight
        self_attn = pos_attn[position] if position < len(pos_attn) else 0.0

        # Sort by attention weight
        indexed = list(enumerate(pos_attn))
        sorted_attn = sorted(indexed, key=lambda x: x[1], reverse=True)[:top_k]

        top_attended = [
            (tokens[idx] if idx < len(tokens) else "?", weight) for idx, weight in sorted_attn
        ]

        return AttentionSummary(
            top_attended=top_attended,
            self_attention_weight=self_attn,
        )

    @staticmethod
    def parse_layers(layers_str: str | None, moe_layers: tuple[int, ...]) -> list[int]:
        """Parse layers argument into list of layer indices.

        Args:
            layers_str: Comma-separated layer indices, "all", or None for default.
            moe_layers: Available MoE layer indices.

        Returns:
            List of layer indices to analyze.
        """
        if layers_str is None:
            # Default: early, middle, late
            if len(moe_layers) >= 3:
                return [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]]
            return list(moe_layers)

        if layers_str.lower() == "all":
            return list(moe_layers)

        # Parse comma-separated
        return [int(x.strip()) for x in layers_str.split(",")]

    @staticmethod
    def parse_contexts(contexts_str: str | None) -> list[tuple[str, str]]:
        """Parse contexts argument into list of (name, prompt) tuples.

        Args:
            contexts_str: Comma-separated context prompts, or None for default.

        Returns:
            List of (name, prompt) tuples.
        """
        if contexts_str is None:
            return DEFAULT_ATTENTION_CONTEXTS

        contexts = []
        for ctx in contexts_str.split(","):
            ctx = ctx.strip()
            if ctx:
                # Use first word as name, full string as prompt
                name = ctx.split()[0] if ctx.split() else ctx[:10]
                contexts.append((name, ctx))
        return contexts

    @staticmethod
    def get_layer_labels(target_layers: list[int]) -> dict[int, str]:
        """Get human-readable labels for layer indices.

        Args:
            target_layers: List of layer indices.

        Returns:
            Dict mapping layer index to label.
        """
        labels = {
            target_layers[0]: "Early",
            target_layers[-1]: "Late",
        }
        if len(target_layers) >= 3:
            labels[target_layers[len(target_layers) // 2]] = "Middle"
        return labels
