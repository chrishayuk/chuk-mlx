#!/usr/bin/env python3
"""
Computation Locator: Find where any model computes any answer.

This is a general-purpose tool to discover:
1. At which layer does the answer first appear? (Emergence)
2. At which layer is the answer most confident? (Peak)
3. At which layer does the answer get "serialized"? (Transition)
4. Which layer's MLP is causal for the answer? (Ablation)
5. Can we transfer answers between prompts? (Patching)

Works with any model family (Gemma, Llama, Qwen, etc.) and any task
(arithmetic, factual recall, code completion, etc.)

Usage:
    # Auto-discover computation layers
    uv run python examples/introspection/computation_locator.py \
        --model "mlx-community/gemma-3-4b-it-bf16" \
        --prompt "347 * 892 = " \
        --answer " 309524"

    # Compare addition vs multiplication
    uv run python examples/introspection/computation_locator.py \
        --model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
        --prompt "156 + 287 = " \
        --answer " 443"

    # Factual recall
    uv run python examples/introspection/computation_locator.py \
        --prompt "The capital of France is" \
        --answer " Paris"

    # Code completion
    uv run python examples/introspection/computation_locator.py \
        --prompt "def hello():\\n    print(" \
        --answer '"'
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


def auto_detect_answer(prompt: str) -> str | None:
    """Auto-detect expected answer for arithmetic prompts."""
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
            return f" {result}"

    return None


class ComputationType(str, Enum):
    """What kind of computation pattern was observed."""

    EARLY_EMERGENCE = "early_emergence"  # Answer appears early, stays
    LATE_EMERGENCE = "late_emergence"  # Answer appears late
    SPIKE_AND_FADE = "spike_and_fade"  # Answer peaks mid-network, fades
    PROGRESSIVE = "progressive"  # Answer probability grows steadily
    SUDDEN = "sudden"  # Answer appears suddenly at one layer
    SERIALIZED = "serialized"  # Full answer → first token pattern


@dataclass
class LayerProbe:
    """Probe result at a single layer."""

    layer_idx: int
    top_token: str
    top_probability: float
    target_token: str
    target_probability: float
    target_rank: int | None
    entropy: float
    is_target_top1: bool


@dataclass
class ComputationProfile:
    """Complete profile of where computation happens."""

    prompt: str
    target_answer: str
    model_id: str
    num_layers: int

    # Per-layer probes
    layer_probes: list[LayerProbe] = field(default_factory=list)

    # Key layers
    emergence_layer: int | None = None  # First layer where answer is top-1
    peak_layer: int | None = None  # Layer with highest answer probability
    peak_probability: float = 0.0
    transition_layer: int | None = None  # Where answer fades (serialization)

    # Computation type
    computation_type: ComputationType | None = None

    # Derived metrics
    computation_window: tuple[int, int] | None = None  # (start, end) of computation
    answer_appears_whole: bool = False  # Does full answer appear before first digit?

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Computation Profile for: {repr(self.prompt)}",
            f"Target answer: {repr(self.target_answer)}",
            f"Model: {self.model_id} ({self.num_layers} layers)",
            "",
        ]

        if self.emergence_layer is not None:
            lines.append(f"✓ Answer emerges at layer {self.emergence_layer}")
        else:
            lines.append("✗ Answer never becomes top-1")

        if self.peak_layer is not None:
            lines.append(
                f"✓ Peak confidence at layer {self.peak_layer} ({self.peak_probability:.1%})"
            )

        if self.transition_layer is not None:
            lines.append(f"✓ Serialization begins at layer {self.transition_layer}")

        if self.computation_type:
            lines.append(f"✓ Computation type: {self.computation_type.value}")

        if self.answer_appears_whole:
            lines.append("✓ Full answer appears before serialization!")

        return "\n".join(lines)


class ComputationLocator:
    """
    Locate where computation happens in any transformer model.

    Model-agnostic: works with any family registered in the model registry.
    Task-agnostic: works with arithmetic, factual recall, code, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any = None,
        model_id: str = "unknown",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "ComputationLocator":
        """Load any supported model."""
        print(f"Loading model: {model_id}")

        # Download
        result = HFLoader.download(model_id)
        model_path = result.model_path

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        # Detect family
        family_type = detect_model_family(config_data)
        if family_type is None:
            model_type = config_data.get("model_type", "unknown")
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"  Detected family: {family_type.value}")

        family_info = get_family_info(family_type)
        config_class = family_info.config_class
        model_class = family_info.model_class

        # Create model
        config = config_class.from_hf_config(config_data)
        model = model_class(config)

        # Load weights
        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        print(f"  Loaded {config.num_hidden_layers} layers")

        # Load tokenizer
        tokenizer = HFLoader.load_tokenizer(model_path)

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self) -> list[nn.Module]:
        """Get transformer layers (model-agnostic)."""
        # Try common patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return list(self.model.transformer.h)
            if hasattr(self.model.transformer, "layers"):
                return list(self.model.transformer.layers)
        raise ValueError("Cannot find layers in model")

    def _get_num_layers(self) -> int:
        """Get number of layers."""
        if self.config and hasattr(self.config, "num_hidden_layers"):
            return self.config.num_hidden_layers
        return len(self._get_layers())

    def _get_embed_tokens(self) -> nn.Module:
        """Get embedding layer."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                return self.model.transformer.wte
        raise ValueError("Cannot find embedding layer")

    def _get_final_norm(self) -> nn.Module | None:
        """Get final layer norm."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "ln_f"):
                return self.model.transformer.ln_f
        return None

    def _get_lm_head(self):
        """Get LM head."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        # Tied embeddings
        embed = self._get_embed_tokens()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def _get_embedding_scale(self) -> float | None:
        """Get embedding scale (for Gemma-style models)."""
        if self.config and hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        return None

    def _forward_all_layers(
        self,
        input_ids: mx.array,
    ) -> dict[int, mx.array]:
        """
        Run forward pass, capturing hidden states at every layer.

        Returns:
            Dict mapping layer_idx -> hidden_state
        """
        layers = self._get_layers()
        embed = self._get_embed_tokens()

        # Embeddings
        h = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale:
            h = h * embed_scale

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        hidden_states = {}

        for layer_idx, layer in enumerate(layers):
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                # SSM layers don't need mask
                layer_out = layer(h)

            # Extract hidden state
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            hidden_states[layer_idx] = h

        return hidden_states

    def _hidden_to_logits(self, hidden: mx.array) -> mx.array:
        """Project hidden state to logits."""
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        h = hidden
        if final_norm:
            h = final_norm(h)

        if lm_head:
            head_out = lm_head(h)
            if hasattr(head_out, "logits"):
                return head_out.logits
            return head_out

        return h

    def _compute_entropy(self, probs: mx.array) -> float:
        """Compute entropy of probability distribution."""
        probs_clipped = mx.clip(probs, 1e-10, 1.0)
        entropy = -mx.sum(probs_clipped * mx.log(probs_clipped))
        return float(entropy)

    def locate_computation(
        self,
        prompt: str,
        target_answer: str,
        layer_step: int = 1,
    ) -> ComputationProfile:
        """
        Locate where the model computes a specific answer.

        Args:
            prompt: Input prompt
            target_answer: Token/answer we're looking for
            layer_step: How many layers to skip (1 = all layers)

        Returns:
            ComputationProfile with full analysis
        """
        # Tokenize
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        # Get target token ID
        target_ids = self.tokenizer.encode(target_answer)
        if not target_ids:
            target_ids = self.tokenizer.encode(target_answer.strip())
        if not target_ids:
            raise ValueError(f"Cannot tokenize target: {repr(target_answer)}")
        target_id = target_ids[-1]  # Use last token if multi-token

        # Get all hidden states
        hidden_states = self._forward_all_layers(input_ids)

        num_layers = self._get_num_layers()
        layers_to_probe = list(range(0, num_layers, layer_step))
        if (num_layers - 1) not in layers_to_probe:
            layers_to_probe.append(num_layers - 1)

        probes = []
        for layer_idx in layers_to_probe:
            if layer_idx not in hidden_states:
                continue

            hidden = hidden_states[layer_idx]
            logits = self._hidden_to_logits(hidden)

            # Get last position
            if logits.ndim == 3:
                pos_logits = logits[0, -1, :]
            else:
                pos_logits = logits[-1, :]

            probs = mx.softmax(pos_logits)

            # Top prediction
            top_idx = int(mx.argmax(probs))
            top_token = self.tokenizer.decode([top_idx])
            top_prob = float(probs[top_idx])

            # Target info
            target_prob = float(probs[target_id])

            # Find target rank
            sorted_idx = mx.argsort(probs)[::-1][:100].tolist()
            target_rank = None
            if target_id in sorted_idx:
                target_rank = sorted_idx.index(target_id) + 1

            # Entropy
            entropy = self._compute_entropy(probs)

            probes.append(
                LayerProbe(
                    layer_idx=layer_idx,
                    top_token=top_token,
                    top_probability=top_prob,
                    target_token=target_answer,
                    target_probability=target_prob,
                    target_rank=target_rank,
                    entropy=entropy,
                    is_target_top1=target_rank == 1,
                )
            )

        # Analyze probes
        profile = ComputationProfile(
            prompt=prompt,
            target_answer=target_answer,
            model_id=self.model_id,
            num_layers=num_layers,
            layer_probes=probes,
        )

        # Find emergence layer (first top-1)
        for probe in probes:
            if probe.is_target_top1:
                profile.emergence_layer = probe.layer_idx
                break

        # Find peak layer
        if probes:
            peak = max(probes, key=lambda p: p.target_probability)
            profile.peak_layer = peak.layer_idx
            profile.peak_probability = peak.target_probability

        # Find transition layer (where answer fades after peak)
        if profile.peak_layer is not None:
            after_peak = [p for p in probes if p.layer_idx > profile.peak_layer]
            for probe in after_peak:
                if not probe.is_target_top1:
                    profile.transition_layer = probe.layer_idx
                    break

        # Classify computation type
        profile.computation_type = self._classify_computation(probes, profile)

        # Check if answer appears whole (for arithmetic serialization detection)
        if profile.peak_layer is not None and profile.transition_layer is not None:
            # Check if what comes after is a "serialization" (first digit)
            first_char = target_answer.strip()[0] if target_answer.strip() else ""
            later_probes = [p for p in probes if p.layer_idx >= profile.transition_layer]
            if later_probes and first_char and first_char in later_probes[-1].top_token:
                profile.answer_appears_whole = True

        # Set computation window
        if profile.emergence_layer is not None and profile.peak_layer is not None:
            profile.computation_window = (profile.emergence_layer, profile.peak_layer)

        return profile

    def _classify_computation(
        self,
        probes: list[LayerProbe],
        profile: ComputationProfile,
    ) -> ComputationType:
        """Classify the computation pattern."""

        if not probes:
            return ComputationType.PROGRESSIVE

        num_layers = profile.num_layers

        # Check for serialization pattern
        if profile.answer_appears_whole or (
            profile.peak_layer is not None
            and profile.transition_layer is not None
            and profile.peak_layer < num_layers * 0.7
        ):
            return ComputationType.SERIALIZED

        # Check for early emergence
        if profile.emergence_layer is not None and profile.emergence_layer < num_layers * 0.3:
            return ComputationType.EARLY_EMERGENCE

        # Check for late emergence
        if profile.emergence_layer is not None and profile.emergence_layer > num_layers * 0.7:
            return ComputationType.LATE_EMERGENCE

        # Check for spike and fade
        if (
            profile.peak_layer is not None
            and profile.peak_probability > 0.5
            and not probes[-1].is_target_top1
        ):
            return ComputationType.SPIKE_AND_FADE

        # Check for sudden
        probs = [p.target_probability for p in probes]
        if len(probs) > 3:
            max_jump = max(probs[i + 1] - probs[i] for i in range(len(probs) - 1))
            if max_jump > 0.3:
                return ComputationType.SUDDEN

        return ComputationType.PROGRESSIVE


def print_probe_table(profile: ComputationProfile):
    """Print a nice table of layer probes."""
    print(
        f"\n{'Layer':<8} {'Top Token':<15} {'Top Prob':<10} {'Target Prob':<12} {'Rank':<8} {'Top1?'}"
    )
    print("-" * 70)

    for probe in profile.layer_probes:
        rank_str = str(probe.target_rank) if probe.target_rank else ">100"
        top1_str = "✓" if probe.is_target_top1 else ""
        bar = "#" * int(probe.target_probability * 30)

        print(
            f"{probe.layer_idx:<8} "
            f"{repr(probe.top_token):<15} "
            f"{probe.top_probability:.4f}    "
            f"{probe.target_probability:.4f} {bar:<12} "
            f"{rank_str:<8} "
            f"{top1_str}"
        )


async def main_analyze(
    model_id: str,
    prompt: str,
    answer: str,
    layer_step: int = 1,
):
    """Run computation location analysis."""

    print(f"\n{'=' * 70}")
    print("COMPUTATION LOCATOR")
    print(f"{'=' * 70}")

    locator = await ComputationLocator.from_pretrained(model_id)

    print(f"\nAnalyzing: {repr(prompt)}")
    print(f"Looking for: {repr(answer)}")

    profile = locator.locate_computation(prompt, answer, layer_step)

    print_probe_table(profile)

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)
    print(profile.summary())

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Locate where computation happens in a transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Any supported model",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="347 * 892 = ",
        help="Input prompt",
    )
    parser.add_argument(
        "--answer",
        "-a",
        default=None,
        help="Target answer/token to locate (auto-detected for arithmetic if not specified)",
    )
    parser.add_argument(
        "--layer-step",
        "-s",
        type=int,
        default=2,
        help="Probe every Nth layer (1=all, 2=every other)",
    )

    args = parser.parse_args()

    # Auto-detect answer for arithmetic
    answer = args.answer if args.answer else auto_detect_answer(args.prompt)
    if answer is None:
        print("Error: Could not auto-detect answer. Please specify --answer")
        exit(1)

    print(f"Tracking answer: {repr(answer)}")

    asyncio.run(
        main_analyze(
            args.model,
            args.prompt,
            answer,
            args.layer_step,
        )
    )


if __name__ == "__main__":
    main()
