#!/usr/bin/env python3
"""
Attention Pattern Analysis: Where does the model look when computing?

For arithmetic like "347 * 892 = ", we expect the final position to:
1. Attend to the operands ("347", "892")
2. Attend to the operator ("*")
3. Attend to the equals sign ("=")

This script analyzes attention patterns at the computation layers
to understand HOW the model gathers information for computation.

Usage:
    uv run python examples/introspection/attention_analysis.py \
        --prompt "347 * 892 = " \
        --layers 20 21 22 23 24

    # Compare attention before and during computation
    uv run python examples/introspection/attention_analysis.py \
        --prompt "156 + 287 = " \
        --compare-layers
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class AttentionSnapshot:
    """Attention pattern at a single layer."""

    layer_idx: int
    query_position: int
    query_token: str
    attended_positions: list[tuple[int, str, float]]  # (pos, token, weight)
    entropy: float  # How focused is the attention?
    max_attention: float  # Strongest attention weight


@dataclass
class AttentionAnalysis:
    """Complete attention analysis for a prompt."""

    prompt: str
    tokens: list[str]
    snapshots: list[AttentionSnapshot] = field(default_factory=list)

    def get_attention_shift(self) -> dict[str, Any]:
        """Analyze how attention shifts between layers."""
        if len(self.snapshots) < 2:
            return {}

        shifts = []
        for i in range(len(self.snapshots) - 1):
            s1 = self.snapshots[i]
            s2 = self.snapshots[i + 1]

            # Compare top-3 attended positions
            top3_s1 = set(p for p, _, _ in s1.attended_positions[:3])
            top3_s2 = set(p for p, _, _ in s2.attended_positions[:3])

            overlap = len(top3_s1 & top3_s2)
            shifts.append(
                {
                    "from_layer": s1.layer_idx,
                    "to_layer": s2.layer_idx,
                    "entropy_change": s2.entropy - s1.entropy,
                    "top3_overlap": overlap,
                    "focus_shift": s1.entropy
                    > s2.entropy,  # True if attention becomes more focused
                }
            )

        return {
            "shifts": shifts,
            "overall_focus_trend": "focusing"
            if sum(s["entropy_change"] for s in shifts) < 0
            else "diffusing",
        }


class AttentionAnalyzer:
    """Analyze attention patterns in transformer models."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "AttentionAnalyzer":
        """Load model."""
        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        print(f"  Family: {family_type.value}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {config.num_hidden_layers}")

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self) -> list[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        raise ValueError("Cannot find layers")

    def _get_embed_tokens(self) -> nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        raise ValueError("Cannot find embed_tokens")

    def _get_embedding_scale(self) -> float | None:
        if hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        return None

    def _compute_attention_weights(
        self,
        query: mx.array,
        key: mx.array,
        mask: mx.array,
        scale: float,
    ) -> mx.array:
        """Compute attention weights manually."""
        # query: [batch, heads, seq, head_dim]
        # key: [batch, heads, seq, head_dim]
        scores = (query @ key.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        return weights

    def analyze_attention(
        self,
        prompt: str,
        layers: list[int],
        query_position: int = -1,
        top_k: int = 10,
    ) -> AttentionAnalysis:
        """
        Analyze attention patterns at specified layers.

        Note: This requires the model to expose attention weights.
        If the model doesn't natively return attention, we compute them manually.
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        seq_len = len(tokens)

        if query_position < 0:
            query_position = seq_len + query_position

        layers_list = self._get_layers()
        embed = self._get_embed_tokens()

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        snapshots = []

        for layer_idx, layer in enumerate(layers_list):
            # Try to get attention weights from layer
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            # Extract hidden state
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
                attn_weights = getattr(layer_out, "attention_weights", None)
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
                attn_weights = layer_out[2] if len(layer_out) > 2 else None
            else:
                h = layer_out
                attn_weights = None

            if layer_idx not in layers:
                continue

            # If we have attention weights, analyze them
            if attn_weights is not None:
                # attn_weights: [batch, heads, seq, seq]
                # Average across heads
                avg_attn = mx.mean(attn_weights[0], axis=0)  # [seq, seq]
                query_attn = avg_attn[query_position].tolist()
            else:
                # Try to compute attention manually from the layer's self_attn
                # This is model-specific and may not work for all architectures
                print(
                    f"  Layer {layer_idx}: No attention weights available (model may not expose them)"
                )
                continue

            # Get top-k attended positions
            indexed = list(enumerate(query_attn))
            indexed.sort(key=lambda x: x[1], reverse=True)

            attended = [(pos, tokens[pos], weight) for pos, weight in indexed[:top_k]]

            # Compute entropy
            attn_tensor = mx.array(query_attn)
            attn_clipped = mx.clip(attn_tensor, 1e-10, 1.0)
            entropy = float(-mx.sum(attn_clipped * mx.log(attn_clipped)))

            snapshots.append(
                AttentionSnapshot(
                    layer_idx=layer_idx,
                    query_position=query_position,
                    query_token=tokens[query_position],
                    attended_positions=attended,
                    entropy=entropy,
                    max_attention=attended[0][2] if attended else 0.0,
                )
            )

        return AttentionAnalysis(
            prompt=prompt,
            tokens=tokens,
            snapshots=snapshots,
        )


def print_attention_analysis(analysis: AttentionAnalysis):
    """Print attention analysis."""
    print(f"\n{'=' * 70}")
    print("ATTENTION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Prompt: {repr(analysis.prompt)}")
    print(f"Tokens: {analysis.tokens}")

    for snapshot in analysis.snapshots:
        print(f"\n--- Layer {snapshot.layer_idx} ---")
        print(f"Query: position {snapshot.query_position} ({repr(snapshot.query_token)})")
        print(f"Entropy: {snapshot.entropy:.3f} (lower = more focused)")
        print(f"Max attention: {snapshot.max_attention:.3f}")
        print("\nTop attended positions:")
        for pos, token, weight in snapshot.attended_positions[:7]:
            bar = "#" * int(weight * 40)
            print(f"  {pos:3d} {repr(token):10} {weight:.3f} {bar}")

    # Print shift analysis
    shift = analysis.get_attention_shift()
    if shift:
        print("\n--- Attention Shift Analysis ---")
        print(f"Overall trend: {shift.get('overall_focus_trend', 'unknown')}")
        for s in shift.get("shifts", []):
            focus = "focusing" if s["focus_shift"] else "diffusing"
            print(
                f"  L{s['from_layer']} -> L{s['to_layer']}: {focus}, overlap={s['top3_overlap']}/3"
            )


async def main(model_id: str, prompt: str, layers: list[int], compare: bool = False):
    """Run attention analysis."""
    analyzer = await AttentionAnalyzer.from_pretrained(model_id)

    if compare:
        # Analyze layers before, during, and after computation
        num_layers = analyzer.config.num_hidden_layers
        layers = [
            num_layers // 4,  # Early
            num_layers // 2,  # Middle
            3 * num_layers // 4,  # Late (computation)
            num_layers - 1,  # Final
        ]

    print(f"\nAnalyzing attention at layers: {layers}")

    analysis = analyzer.analyze_attention(prompt, layers)
    print_attention_analysis(analysis)

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--prompt", "-p", default="347 * 892 = ")
    parser.add_argument("--layers", "-l", nargs="+", type=int, default=[18, 20, 22, 24])
    parser.add_argument("--compare-layers", "-c", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.prompt, args.layers, args.compare_layers))
