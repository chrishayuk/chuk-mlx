#!/usr/bin/env python3
"""
Computation Flow Visualization: See how answers emerge layer by layer.

Generates ASCII art showing the probability evolution of tokens across layers,
making it easy to see:
1. When computation happens (answer probability spikes)
2. When serialization occurs (full answer -> first digit)
3. Which layers are "active" for different tasks

Usage:
    uv run python examples/introspection/computation_flow.py \
        --prompt "347 * 892 = " \
        --track " 309524" "3"

    # Compare two prompts
    uv run python examples/introspection/computation_flow.py \
        --compare "156 + 287 = " "156 + 287 ="
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


def auto_detect_tokens(prompt: str) -> list[str]:
    """Auto-detect tokens to track based on arithmetic in prompt."""
    # Pattern: number op number
    patterns = [
        (r"(\d+)\s*\+\s*(\d+)", lambda a, b: a + b),
        (r"(\d+)\s*-\s*(\d+)", lambda a, b: a - b),
        (r"(\d+)\s*\*\s*(\d+)", lambda a, b: a * b),
        (r"(\d+)\s*×\s*(\d+)", lambda a, b: a * b),
        (r"(\d+)\s*/\s*(\d+)", lambda a, b: a // b if b != 0 else 0),
        (r"(\d+)\s*÷\s*(\d+)", lambda a, b: a // b if b != 0 else 0),
    ]

    for pattern, op in patterns:
        match = re.search(pattern, prompt)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            result = op(a, b)
            result_str = str(abs(result))

            # Track: full answer with space, first digit, and space alone
            return [f" {result_str}", result_str[0], " "]

    # Default fallback
    return [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


@dataclass
class TokenTrajectory:
    """Probability trajectory for a single token."""
    token: str
    token_id: int
    probabilities: list[tuple[int, float]]  # (layer, prob)
    ranks: list[tuple[int, int | None]]  # (layer, rank)
    peak_layer: int
    peak_prob: float
    emergence_layer: int | None


@dataclass
class FlowVisualization:
    """Data for visualization."""
    prompt: str
    model_id: str
    num_layers: int
    trajectories: list[TokenTrajectory]
    top_at_each_layer: list[tuple[int, str, float]]  # (layer, token, prob)


class ComputationFlow:
    """Visualize computation flow through layers."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "ComputationFlow":
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

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

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

    def _get_final_norm(self) -> nn.Module | None:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        embed = self._get_embed_tokens()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def _get_embedding_scale(self) -> float | None:
        if hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        return None

    def trace_tokens(
        self,
        prompt: str,
        tokens_to_track: list[str],
        layer_step: int = 1,
    ) -> FlowVisualization:
        """Trace token probabilities through all layers."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get token IDs for tracking
        track_ids = {}
        for tok in tokens_to_track:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                track_ids[tok] = ids[-1]

        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()
        num_layers = len(layers)

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Track probabilities
        trajectories = {tok: {"probs": [], "ranks": []} for tok in tokens_to_track}
        top_at_layer = []

        for layer_idx, layer in enumerate(layers):
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            if layer_idx % layer_step != 0 and layer_idx != num_layers - 1:
                continue

            # Project to logits
            h_normed = final_norm(h) if final_norm else h
            if lm_head:
                head_out = lm_head(h_normed)
                logits = head_out.logits if hasattr(head_out, "logits") else head_out
            else:
                logits = h_normed

            probs = mx.softmax(logits[0, -1, :])
            top_idx = int(mx.argmax(probs))
            top_token = self.tokenizer.decode([top_idx])
            top_prob = float(probs[top_idx])

            top_at_layer.append((layer_idx, top_token, top_prob))

            # Track each token
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()
            for tok in tokens_to_track:
                if tok in track_ids:
                    tid = track_ids[tok]
                    prob = float(probs[tid])
                    rank = sorted_idx.index(tid) + 1 if tid in sorted_idx else None
                    trajectories[tok]["probs"].append((layer_idx, prob))
                    trajectories[tok]["ranks"].append((layer_idx, rank))

        # Build trajectory objects
        result_trajectories = []
        for tok in tokens_to_track:
            if tok not in track_ids:
                continue

            probs = trajectories[tok]["probs"]
            ranks = trajectories[tok]["ranks"]

            if not probs:
                continue

            # Find peak
            peak_layer, peak_prob = max(probs, key=lambda x: x[1])

            # Find emergence (first rank 1)
            emergence = None
            for layer_idx, rank in ranks:
                if rank == 1:
                    emergence = layer_idx
                    break

            result_trajectories.append(TokenTrajectory(
                token=tok,
                token_id=track_ids[tok],
                probabilities=probs,
                ranks=ranks,
                peak_layer=peak_layer,
                peak_prob=peak_prob,
                emergence_layer=emergence,
            ))

        return FlowVisualization(
            prompt=prompt,
            model_id=self.model_id,
            num_layers=num_layers,
            trajectories=result_trajectories,
            top_at_each_layer=top_at_layer,
        )


def render_flow_chart(viz: FlowVisualization, width: int = 60):
    """Render ASCII flow chart."""

    print(f"\n{'='*80}")
    print("COMPUTATION FLOW VISUALIZATION")
    print(f"{'='*80}")
    print(f"Prompt: {repr(viz.prompt)}")
    print(f"Model: {viz.model_id} ({viz.num_layers} layers)")
    print()

    if not viz.trajectories:
        print("No tokens to track!")
        return

    # Header
    layer_indices = [l for l, _, _ in viz.top_at_each_layer]
    max_layers_to_show = 20

    if len(layer_indices) > max_layers_to_show:
        # Show every Nth layer
        step = len(layer_indices) // max_layers_to_show
        layer_indices = layer_indices[::step]
        if viz.num_layers - 1 not in layer_indices:
            layer_indices.append(viz.num_layers - 1)

    # Build layer header
    header = "Token".ljust(15) + "│"
    for l in layer_indices:
        header += f" L{l:02d}"
    print(header)
    print("─" * 15 + "┼" + "─" * (len(layer_indices) * 4 + 1))

    # Plot each trajectory
    for traj in viz.trajectories:
        # Build probability dict for quick lookup
        prob_dict = dict(traj.probabilities)
        rank_dict = dict(traj.ranks)

        row = repr(traj.token)[:14].ljust(15) + "│"

        for l in layer_indices:
            prob = prob_dict.get(l, 0.0)
            rank = rank_dict.get(l)

            if rank == 1:
                # Top-1: show special marker
                row += " ★★★"
            elif prob >= 0.1:
                # High prob: show filled blocks
                blocks = min(3, int(prob * 3))
                row += " " + "█" * blocks + "░" * (3 - blocks)
            elif prob >= 0.01:
                # Medium prob: show partial
                row += " ▓░░"
            elif rank and rank <= 10:
                # Low prob but in top-10
                row += f" r{rank:02d}"
            else:
                # Not visible
                row += "  · "

        row += f"  peak@L{traj.peak_layer}({traj.peak_prob:.1%})"
        if traj.emergence_layer is not None:
            row += f" ★L{traj.emergence_layer}"

        print(row)

    # Show top prediction per layer
    print("─" * 15 + "┼" + "─" * (len(layer_indices) * 4 + 1))
    row = "Top token".ljust(15) + "│"
    for l in layer_indices:
        top = next((t for layer, t, _ in viz.top_at_each_layer if layer == l), "?")
        # Truncate to 3 chars
        top_short = repr(top)[:3]
        row += f" {top_short:>3}"
    print(row)

    # Legend
    print()
    print("Legend: ★★★=rank 1 | ███=high prob | ▓░░=medium | r##=rank | ·=low/absent")

    # Summary
    print()
    print("─" * 80)
    print("SUMMARY:")
    for traj in viz.trajectories:
        status = ""
        if traj.emergence_layer is not None:
            status = f"emerges at L{traj.emergence_layer}"
        else:
            status = "never reaches rank 1"
        print(f"  {repr(traj.token):15} peaks at L{traj.peak_layer} ({traj.peak_prob:.1%}), {status}")


async def main(
    model_id: str,
    prompt: str,
    tokens: list[str],
    layer_step: int = 1,
):
    """Run flow visualization."""
    flow = await ComputationFlow.from_pretrained(model_id)

    viz = flow.trace_tokens(prompt, tokens, layer_step)
    render_flow_chart(viz)

    return viz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--prompt", "-p", default="347 * 892 = ")
    parser.add_argument("--track", "-t", nargs="+", default=None,
                        help="Tokens to track. If not specified, auto-detects for arithmetic.")
    parser.add_argument("--layer-step", "-s", type=int, default=2)
    args = parser.parse_args()

    # Auto-detect tokens if not specified
    tokens = args.track if args.track else auto_detect_tokens(args.prompt)
    print(f"Tracking tokens: {tokens}")

    # Warn about base models
    if "-pt" in args.model or args.model.endswith("-pt"):
        print("\n⚠️  WARNING: You're using a base/pretrained model (not instruction-tuned).")
        print("   Base models may not perform arithmetic correctly.")
        print("   Consider using an instruction-tuned model like 'mlx-community/gemma-3-4b-it-bf16'\n")

    asyncio.run(main(args.model, args.prompt, tokens, args.layer_step))
