#!/usr/bin/env python3
"""
Gemma Layer Roles: What does each layer do?

This script analyzes layer-by-layer specialization in Gemma to understand:
1. Token prediction evolution (logit lens) - when do answers emerge?
2. Attention vs MLP contribution per layer
3. Residual stream changes - what does each layer add?
4. Task-specific layer activation patterns

Comparison with GPT-OSS findings:
- GPT-OSS has distinct "lookup" vs "compute" layers
- GPT-OSS has MoE routing that creates sparse specialization
- Does Gemma have similar layer roles?

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py
    uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py --task arithmetic
    uv run python examples/introspection/experiments/model_specific/gemma_layer_roles.py --task factual
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class LayerContribution:
    """What a single layer contributes."""

    layer_idx: int

    # Logit lens: what does this layer predict?
    top_prediction: str
    top_probability: float
    target_probability: float
    target_rank: int | None

    # Residual contribution
    residual_norm: float  # How much does this layer change the residual?
    attention_norm: float  # Attention contribution norm
    mlp_norm: float  # MLP contribution norm

    # Cosine similarity to final
    similarity_to_final: float


@dataclass
class LayerProfile:
    """Aggregate profile of a layer across many prompts."""

    layer_idx: int

    # Average contributions
    avg_residual_change: float
    avg_attention_contribution: float
    avg_mlp_contribution: float
    attention_mlp_ratio: float

    # Prediction quality
    avg_target_probability: float
    emergence_rate: float  # How often is target in top-1?

    # Role classification
    role: str  # "embedding", "early", "middle", "late", "output"
    specialization: str  # "attention-heavy", "mlp-heavy", "balanced"


class GemmaLayerAnalyzer:
    """Analyze what each layer in Gemma does."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self):
        """Load model."""
        print(f"Loading model: {self.model_id}")

        result = HFLoader.download(self.model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        family_info = get_family_info(family_type)
        self.config = family_info.config_class.from_hf_config(config_data)
        self.model = family_info.model_class(self.config)

        HFLoader.apply_weights_to_model(self.model, model_path, self.config, dtype=DType.BFLOAT16)
        self.tokenizer = HFLoader.load_tokenizer(model_path)

        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        norm = getattr(backbone, "norm", None)

        # LM head
        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        else:
            head = None

        # Embedding scale (critical for Gemma)
        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
            embed_scale = float(self.hidden_size**0.5)

        return layers, embed, norm, head, embed_scale

    def analyze_prompt(
        self,
        prompt: str,
        target_token: str | None = None,
    ) -> list[LayerContribution]:
        """
        Analyze layer-by-layer contributions for a single prompt.

        Returns per-layer analysis including:
        - Logit lens predictions
        - Attention vs MLP contribution
        - Residual stream changes
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        # Get target token ID if specified
        target_id = None
        if target_token:
            target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
            target_id = target_ids[0] if target_ids else None

        # Initial embedding
        h = embed(input_ids)
        h = h * embed_scale

        # Create mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        results = []

        # Get final hidden state for similarity comparison
        h_final = self._get_final_hidden(prompt)

        for layer_idx, layer in enumerate(layers):
            h_before = h

            # === Run attention ===
            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            else:
                h_normed = h

            attn = layer.self_attn
            try:
                attn_out = attn(h_normed, mask=mask)
            except TypeError:
                attn_out = attn(h_normed)

            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]

            # Attention contribution
            h_after_attn = h + attn_out
            attn_contribution = attn_out[0, -1, :]
            attn_norm = float(mx.sqrt(mx.sum(attn_contribution**2)))

            # === Run MLP ===
            if hasattr(layer, "post_attention_layernorm"):
                h_normed_mlp = layer.post_attention_layernorm(h_after_attn)
            else:
                h_normed_mlp = h_after_attn

            mlp = layer.mlp
            mlp_out = mlp(h_normed_mlp)

            # MLP contribution
            h = h_after_attn + mlp_out
            mlp_contribution = mlp_out[0, -1, :]
            mlp_norm = float(mx.sqrt(mx.sum(mlp_contribution**2)))

            # Residual change
            residual_change = h[0, -1, :] - h_before[0, -1, :]
            residual_norm = float(mx.sqrt(mx.sum(residual_change**2)))

            # === Logit lens ===
            h_probe = h
            if norm is not None:
                h_probe = norm(h_probe)

            if head is not None:
                logits = head(h_probe)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                # Tied embeddings
                logits = h_probe @ embed.weight.T

            probs = mx.softmax(logits[0, -1, :])
            top_idx = int(mx.argmax(probs))
            top_token = self.tokenizer.decode([top_idx])
            top_prob = float(probs[top_idx])

            # Target probability and rank
            target_prob = 0.0
            target_rank = None
            if target_id is not None:
                target_prob = float(probs[target_id])
                sorted_idx = mx.argsort(probs)[::-1][:100].tolist()
                if target_id in sorted_idx:
                    target_rank = sorted_idx.index(target_id) + 1

            # Similarity to final
            h_current = np.array(h[0, -1, :].tolist())
            sim = np.dot(h_current, h_final) / (np.linalg.norm(h_current) * np.linalg.norm(h_final))

            results.append(
                LayerContribution(
                    layer_idx=layer_idx,
                    top_prediction=top_token,
                    top_probability=top_prob,
                    target_probability=target_prob,
                    target_rank=target_rank,
                    residual_norm=residual_norm,
                    attention_norm=attn_norm,
                    mlp_norm=mlp_norm,
                    similarity_to_final=float(sim),
                )
            )

        return results

    def _get_final_hidden(self, prompt: str) -> np.ndarray:
        """Get the final hidden state for a prompt."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer in layers:
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

        return np.array(h[0, -1, :].tolist())

    def profile_layers(
        self,
        prompts: list[tuple[str, str]],  # (prompt, target_token)
    ) -> list[LayerProfile]:
        """
        Build aggregate profiles for each layer across multiple prompts.
        """
        print(f"\nProfiling {self.num_layers} layers across {len(prompts)} prompts...")

        # Collect per-layer stats
        layer_stats = defaultdict(
            lambda: {
                "residual_norms": [],
                "attention_norms": [],
                "mlp_norms": [],
                "target_probs": [],
                "is_top1": [],
            }
        )

        for prompt, target in prompts:
            contributions = self.analyze_prompt(prompt, target)

            for c in contributions:
                stats = layer_stats[c.layer_idx]
                stats["residual_norms"].append(c.residual_norm)
                stats["attention_norms"].append(c.attention_norm)
                stats["mlp_norms"].append(c.mlp_norm)
                stats["target_probs"].append(c.target_probability)
                stats["is_top1"].append(c.target_rank == 1 if c.target_rank else False)

        # Build profiles
        profiles = []

        for layer_idx in range(self.num_layers):
            stats = layer_stats[layer_idx]

            avg_residual = np.mean(stats["residual_norms"])
            avg_attention = np.mean(stats["attention_norms"])
            avg_mlp = np.mean(stats["mlp_norms"])

            # Avoid division by zero
            if avg_attention > 0:
                ratio = avg_mlp / avg_attention
            else:
                ratio = float("inf")

            avg_target_prob = np.mean(stats["target_probs"])
            emergence_rate = np.mean(stats["is_top1"])

            # Classify role based on position
            position_ratio = layer_idx / self.num_layers
            if position_ratio < 0.1:
                role = "embedding"
            elif position_ratio < 0.3:
                role = "early"
            elif position_ratio < 0.7:
                role = "middle"
            elif position_ratio < 0.9:
                role = "late"
            else:
                role = "output"

            # Classify specialization
            if ratio < 0.8:
                specialization = "attention-heavy"
            elif ratio > 1.2:
                specialization = "mlp-heavy"
            else:
                specialization = "balanced"

            profiles.append(
                LayerProfile(
                    layer_idx=layer_idx,
                    avg_residual_change=avg_residual,
                    avg_attention_contribution=avg_attention,
                    avg_mlp_contribution=avg_mlp,
                    attention_mlp_ratio=ratio,
                    avg_target_probability=avg_target_prob,
                    emergence_rate=emergence_rate,
                    role=role,
                    specialization=specialization,
                )
            )

        return profiles

    def print_layer_analysis(
        self,
        contributions: list[LayerContribution],
        prompt: str,
        target: str,
    ):
        """Pretty print layer analysis."""
        print(f"\n{'=' * 80}")
        print(f"LAYER ANALYSIS: {repr(prompt)} -> {repr(target)}")
        print(f"{'=' * 80}")

        print(
            f"\n{'Layer':<8} {'Prediction':<15} {'Prob':>8} {'Target':>8} {'Rank':>6} {'Attn':>10} {'MLP':>10} {'Ratio':>8}"
        )
        print("-" * 90)

        for c in contributions:
            pred_display = repr(c.top_prediction)[:12]
            rank_display = str(c.target_rank) if c.target_rank else "-"
            ratio = c.mlp_norm / c.attention_norm if c.attention_norm > 0 else float("inf")

            # Highlight key layers
            marker = ""
            if c.target_rank == 1:
                marker = " <-- EMERGES"
            elif c.target_probability > 0.1 and (
                not contributions[c.layer_idx - 1].target_probability > 0.1
                if c.layer_idx > 0
                else True
            ):
                marker = " <-- RISING"

            print(
                f"L{c.layer_idx:<6} {pred_display:<15} {c.top_probability:>8.3f} {c.target_probability:>8.3f} {rank_display:>6} {c.attention_norm:>10.1f} {c.mlp_norm:>10.1f} {ratio:>8.2f}{marker}"
            )

    def print_profiles(self, profiles: list[LayerProfile]):
        """Pretty print layer profiles."""
        print(f"\n{'=' * 80}")
        print("LAYER PROFILES (aggregated across all prompts)")
        print(f"{'=' * 80}")

        print(
            f"\n{'Layer':<8} {'Role':<12} {'Spec':<16} {'Attn':>10} {'MLP':>10} {'Ratio':>8} {'TargetP':>10} {'Emerge':>8}"
        )
        print("-" * 100)

        for p in profiles:
            print(
                f"L{p.layer_idx:<6} {p.role:<12} {p.specialization:<16} {p.avg_attention_contribution:>10.1f} {p.avg_mlp_contribution:>10.1f} {p.attention_mlp_ratio:>8.2f} {p.avg_target_probability:>10.3f} {p.emergence_rate:>8.1%}"
            )

    def find_phase_transitions(self, profiles: list[LayerProfile]) -> dict:
        """Identify phase transitions in the network."""
        transitions = {
            "attention_to_mlp": None,  # Where MLP starts dominating
            "emergence_layer": None,  # Where answers start emerging
            "stabilization_layer": None,  # Where predictions stabilize
        }

        # Find attention->MLP transition
        for i, p in enumerate(profiles):
            if i > 0 and profiles[i - 1].attention_mlp_ratio < 1.0 and p.attention_mlp_ratio >= 1.0:
                transitions["attention_to_mlp"] = p.layer_idx
                break

        # Find emergence layer (first with >50% emergence rate)
        for p in profiles:
            if p.emergence_rate > 0.5:
                transitions["emergence_layer"] = p.layer_idx
                break

        # Find stabilization (where emergence rate plateaus)
        for i in range(len(profiles) - 3):
            rates = [profiles[i + j].emergence_rate for j in range(4)]
            if all(r > 0.8 for r in rates):
                transitions["stabilization_layer"] = profiles[i].layer_idx
                break

        return transitions

    def run_analysis(self, task: str = "mixed"):
        """Run complete layer analysis."""
        self.load_model()

        # Define test prompts by task
        if task == "arithmetic":
            prompts = [
                ("7 * 8 = ", "5"),  # First digit of 56
                ("3 + 4 = ", "7"),
                ("9 * 9 = ", "8"),  # First digit of 81
                ("15 + 23 = ", "3"),  # First digit of 38
                ("100 - 37 = ", "6"),  # First digit of 63
                ("12 * 12 = ", "1"),  # First digit of 144
                ("25 + 17 = ", "4"),  # First digit of 42
                ("81 / 9 = ", "9"),
            ]
        elif task == "factual":
            prompts = [
                ("The capital of France is", " Paris"),
                ("The speed of light is approximately", " 299"),
                ("Water freezes at", " 0"),
                ("The largest planet is", " Jupiter"),
                ("Shakespeare wrote", " Hamlet"),
                ("Einstein developed the theory of", " relativity"),
                ("DNA stands for", " de"),
                ("The chemical symbol for gold is", " Au"),
            ]
        else:  # mixed
            prompts = [
                ("7 * 8 = ", "5"),
                ("The capital of France is", " Paris"),
                ("9 * 9 = ", "8"),
                ("Water freezes at", " 0"),
                ("3 + 4 = ", "7"),
                ("The largest planet is", " Jupiter"),
                ("12 * 12 = ", "1"),
                ("Einstein developed the theory of", " relativity"),
            ]

        print(f"\nTask: {task}")
        print(f"Prompts: {len(prompts)}")

        # Detailed analysis of first prompt
        print("\n" + "=" * 80)
        print("DETAILED LAYER-BY-LAYER ANALYSIS (first prompt)")
        print("=" * 80)

        first_prompt, first_target = prompts[0]
        contributions = self.analyze_prompt(first_prompt, first_target)
        self.print_layer_analysis(contributions, first_prompt, first_target)

        # Aggregate profiles
        profiles = self.profile_layers(prompts)
        self.print_profiles(profiles)

        # Phase transitions
        transitions = self.find_phase_transitions(profiles)

        print(f"\n{'=' * 80}")
        print("PHASE TRANSITIONS")
        print(f"{'=' * 80}")

        for name, layer in transitions.items():
            if layer is not None:
                print(f"  {name}: Layer {layer}")
            else:
                print(f"  {name}: Not detected")

        # Layer role summary
        print(f"\n{'=' * 80}")
        print("LAYER ROLE SUMMARY")
        print(f"{'=' * 80}")

        # Group by specialization
        attn_heavy = [p for p in profiles if p.specialization == "attention-heavy"]
        mlp_heavy = [p for p in profiles if p.specialization == "mlp-heavy"]
        balanced = [p for p in profiles if p.specialization == "balanced"]

        print(f"\nAttention-heavy layers ({len(attn_heavy)}):")
        print(f"  Layers: {[p.layer_idx for p in attn_heavy]}")

        print(f"\nMLP-heavy layers ({len(mlp_heavy)}):")
        print(f"  Layers: {[p.layer_idx for p in mlp_heavy]}")

        print(f"\nBalanced layers ({len(balanced)}):")
        print(f"  Layers: {[p.layer_idx for p in balanced]}")

        # Interpretation
        print(f"\n{'=' * 80}")
        print("INTERPRETATION")
        print(f"{'=' * 80}")

        print("""
Based on the analysis:

1. EARLY LAYERS (embedding/early):
   - Typically attention-heavy
   - Build contextual representations
   - Low target probability (answer not yet computed)

2. MIDDLE LAYERS:
   - Mix of attention and MLP
   - This is where computation happens
   - Target probability starts rising

3. LATE LAYERS:
   - Often MLP-heavy
   - Refine and "project" answer to output space
   - Target probability stabilizes

4. OUTPUT LAYERS:
   - Final refinement
   - May show slight decrease in some metrics (over-processing)
""")

        # Save results
        results = {
            "task": task,
            "model": self.model_id,
            "num_layers": self.num_layers,
            "profiles": [
                {
                    "layer": p.layer_idx,
                    "role": p.role,
                    "specialization": p.specialization,
                    "attention_mlp_ratio": p.attention_mlp_ratio,
                    "avg_target_prob": p.avg_target_probability,
                    "emergence_rate": p.emergence_rate,
                }
                for p in profiles
            ],
            "transitions": transitions,
        }

        output_path = Path("gemma_discovery_cache/layer_roles.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return profiles, transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--task", "-t", choices=["arithmetic", "factual", "mixed"], default="mixed")
    args = parser.parse_args()

    analyzer = GemmaLayerAnalyzer(model_id=args.model)
    analyzer.run_analysis(task=args.task)


if __name__ == "__main__":
    main()
