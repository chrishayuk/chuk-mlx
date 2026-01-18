"""
Attention extraction and analysis utilities.

Provides tools for:
- Extracting attention patterns at specific layers
- Analyzing what tokens attend to what
- Tracing attention flow through CoT generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    import mlx.nn as nn


@dataclass
class AttentionPattern:
    """Captured attention pattern at a specific position."""

    layer: int
    position: int
    token: str
    attention_weights: list[float]  # Attention to all previous positions
    top_attended_positions: list[int]
    top_attended_tokens: list[str]
    entropy: float  # How spread out is attention


@dataclass
class AttentionTrace:
    """Full attention trace for a sequence."""

    prompt: str
    tokens: list[str]
    layer: int
    patterns: list[AttentionPattern] = field(default_factory=list)
    invoke_position: int | None = None
    invoke_pattern: AttentionPattern | None = None


class AttentionExtractor:
    """
    Extracts and analyzes attention patterns from transformer models.

    Usage:
        extractor = AttentionExtractor(model, tokenizer)
        trace = extractor.trace_prompt("5 + 3 =", layer=12)
        print(trace.invoke_pattern.top_attended_tokens)
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def trace_prompt(self, prompt: str, layer: int) -> AttentionTrace:
        """
        Extract attention trace for a prompt at a specific layer.

        Args:
            prompt: Input text
            layer: Which layer to extract attention from

        Returns:
            AttentionTrace with patterns for each position
        """
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        token_strs = [
            self.tokenizer.decode([t]).strip() or f"[{t}]" for t in tokens
        ]

        # Find invoke position (= or end)
        invoke_pos = len(tokens) - 1
        for i, t in enumerate(token_strs):
            if "=" in t:
                invoke_pos = i
                break

        trace = AttentionTrace(
            prompt=prompt,
            tokens=token_strs,
            layer=layer,
            invoke_position=invoke_pos,
        )

        # Get attention patterns
        attention = self._get_attention_at_layer(input_ids, layer)

        if attention is not None:
            # Analyze each position
            for pos in range(1, len(tokens)):  # Skip first token
                pattern = self._analyze_position(
                    attention, pos, token_strs
                )
                trace.patterns.append(pattern)

                if pos == invoke_pos:
                    trace.invoke_pattern = pattern

        return trace

    def _get_attention_at_layer(
        self, input_ids: mx.array, layer: int
    ) -> mx.array | None:
        """Extract attention patterns at a specific layer."""
        try:
            output = self.model(input_ids, output_attentions=True)

            if hasattr(output, "attentions") and output.attentions:
                if layer < len(output.attentions):
                    return output.attentions[layer]
        except Exception:
            pass

        return None

    def _analyze_position(
        self,
        attention: mx.array,
        position: int,
        tokens: list[str],
    ) -> AttentionPattern:
        """Analyze attention pattern at a specific position."""
        # Average across heads: [batch, heads, seq, seq] -> [seq]
        attn = attention[0, :, position, :position + 1]
        avg_attn = mx.mean(attn, axis=0)
        mx.eval(avg_attn)

        weights = avg_attn.tolist()

        # Get top attended
        top_k = min(5, position + 1)
        top_indices = mx.argsort(avg_attn)[-top_k:][::-1].tolist()

        # Compute entropy
        entropy = self._compute_entropy(avg_attn)

        return AttentionPattern(
            layer=0,  # Will be set by caller
            position=position,
            token=tokens[position],
            attention_weights=weights,
            top_attended_positions=top_indices,
            top_attended_tokens=[tokens[i] for i in top_indices],
            entropy=entropy,
        )

    def _compute_entropy(self, probs: mx.array) -> float:
        """Compute entropy of attention distribution."""
        # Add small epsilon for numerical stability
        probs_safe = mx.clip(probs, 1e-10, 1.0)
        entropy = -mx.sum(probs_safe * mx.log(probs_safe))
        return float(entropy.item())

    def get_invoke_attention(
        self, prompt: str, layer: int
    ) -> AttentionPattern | None:
        """
        Get attention pattern specifically at the invoke position.

        This is the key measurement: what does the model attend to
        when it sees "=" and needs to invoke a circuit?
        """
        trace = self.trace_prompt(prompt, layer)
        return trace.invoke_pattern

    def compare_patterns(
        self,
        pattern1: AttentionPattern,
        pattern2: AttentionPattern,
    ) -> float:
        """
        Compare two attention patterns for similarity.

        Returns cosine similarity of attention weight vectors.
        """
        # Pad to same length
        max_len = max(len(pattern1.attention_weights), len(pattern2.attention_weights))
        w1 = pattern1.attention_weights + [0.0] * (max_len - len(pattern1.attention_weights))
        w2 = pattern2.attention_weights + [0.0] * (max_len - len(pattern2.attention_weights))

        v1 = mx.array(w1)
        v2 = mx.array(w2)

        dot = mx.sum(v1 * v2)
        norm1 = mx.linalg.norm(v1)
        norm2 = mx.linalg.norm(v2)

        if norm1 > 0 and norm2 > 0:
            similarity = float((dot / (norm1 * norm2)).item())
        else:
            similarity = 0.0

        return similarity
