#!/usr/bin/env python3
"""
Gate Neuron Finder: Locate the specific neurons that control answer output.

The key insight from ghost hunting:
- "100 - 37 = " → outputs '6' (gate OPEN)
- "100 - 37 ="  → outputs ' ' then fails (gate CLOSED)

This script:
1. Compares activations between open/closed cases
2. Finds neurons with maximum difference
3. Causally verifies by patching single neurons
4. Identifies the minimal set needed to flip the gate

Usage:
    uv run python examples/introspection/gate_neuron_finder.py \
        --open "100 - 37 = " \
        --closed "100 - 37 =" \
        --model "mlx-community/gemma-3-4b-it-bf16"

    # Focus on specific layers
    uv run python examples/introspection/gate_neuron_finder.py \
        --open "156 + 287 = " \
        --closed "156 + 287 =" \
        --layers 18,19,20,21,22,23
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
            return str(abs(result))

    return None


@dataclass
class NeuronCandidate:
    """A candidate gate neuron."""

    layer: int
    neuron: int
    open_value: float
    closed_value: float
    difference: float

    @property
    def direction(self) -> str:
        return "OPEN" if self.difference > 0 else "CLOSE"


@dataclass
class PatchResult:
    """Result of patching a neuron."""

    layer: int
    neuron: int
    original_output: str
    patched_output: str
    flipped: bool
    target_digit: str


class GateNeuronFinder:
    """Find neurons that control the answer gate."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "GateNeuronFinder":
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

        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden dim: {config.hidden_size}")

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self):
        if hasattr(self.model, "model"):
            return list(self.model.model.layers)
        return list(self.model.layers)

    def _get_embed(self):
        if hasattr(self.model, "model"):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    def _get_norm(self):
        if hasattr(self.model, "model"):
            return getattr(self.model.model, "norm", None)
        return getattr(self.model, "norm", None)

    def _get_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return self._get_embed().as_linear

    def _get_scale(self):
        return getattr(self.config, "embedding_scale", None)

    def get_hidden_states(self, prompt: str) -> dict[int, mx.array]:
        """Get hidden states at each layer for the last position."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        hidden_states = {}

        for layer_idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            # Store last position hidden state
            hidden_states[layer_idx] = h[0, -1, :]  # [hidden_dim]

        return hidden_states

    def get_output(self, prompt: str) -> str:
        """Get model output for prompt."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for layer in layers:
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

        h_n = norm(h) if norm else h
        logits = head(h_n)
        if hasattr(logits, "logits"):
            logits = logits.logits

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def get_output_with_patch(
        self,
        prompt: str,
        layer_idx: int,
        neuron_idx: int,
        patch_value: float,
    ) -> str:
        """Get model output with a single neuron patched."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            # Apply patch at specified layer
            if idx == layer_idx:
                # Patch the specific neuron at last position
                h_list = h.tolist()
                h_list[0][-1][neuron_idx] = patch_value
                h = mx.array(h_list)

        h_n = norm(h) if norm else h
        logits = head(h_n)
        if hasattr(logits, "logits"):
            logits = logits.logits

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def find_gate_neurons(
        self,
        open_prompt: str,
        closed_prompt: str,
        layers: list[int] | None = None,
        top_k: int = 20,
    ) -> list[NeuronCandidate]:
        """Find neurons with maximum difference between open/closed cases."""

        print(f"\n{'=' * 70}")
        print("GATE NEURON SEARCH")
        print(f"{'=' * 70}")
        print(f"Open prompt:   {repr(open_prompt)}")
        print(f"Closed prompt: {repr(closed_prompt)}")

        # Get hidden states for both
        print("\nCollecting hidden states...")
        h_open = self.get_hidden_states(open_prompt)
        h_closed = self.get_hidden_states(closed_prompt)

        # Get outputs
        out_open = self.get_output(open_prompt)
        out_closed = self.get_output(closed_prompt)
        print(f"Open output:   {repr(out_open)}")
        print(f"Closed output: {repr(out_closed)}")

        num_layers = len(self._get_layers())
        if layers is None:
            # Focus on layers 15-25 (computation region)
            layers = list(range(max(0, num_layers // 2 - 5), min(num_layers, num_layers // 2 + 10)))

        # Filter out invalid layers (must be < num_layers)
        layers = [l for l in layers if l < num_layers]

        candidates = []

        print(f"\nSearching layers: {layers} (model has {num_layers} layers)")
        print(f"{'=' * 70}")

        for layer in layers:
            h_o = h_open[layer]
            h_c = h_closed[layer]

            # Compute difference
            diff = h_o - h_c  # [hidden_dim]
            diff_list = diff.tolist()

            # Find top positive differences (higher when gate open)
            indexed = [(i, d) for i, d in enumerate(diff_list)]
            indexed.sort(key=lambda x: x[1], reverse=True)

            # Top neurons that are HIGHER when gate is open
            for neuron, d in indexed[: top_k // 2]:
                candidates.append(
                    NeuronCandidate(
                        layer=layer,
                        neuron=neuron,
                        open_value=float(h_o[neuron]),
                        closed_value=float(h_c[neuron]),
                        difference=d,
                    )
                )

            # Top neurons that are LOWER when gate is open
            for neuron, d in indexed[-(top_k // 2) :]:
                candidates.append(
                    NeuronCandidate(
                        layer=layer,
                        neuron=neuron,
                        open_value=float(h_o[neuron]),
                        closed_value=float(h_c[neuron]),
                        difference=d,
                    )
                )

        # Sort by absolute difference
        candidates.sort(key=lambda x: abs(x.difference), reverse=True)

        return candidates

    def verify_candidates(
        self,
        candidates: list[NeuronCandidate],
        closed_prompt: str,
        target_digit: str,
        max_candidates: int = 50,
    ) -> list[PatchResult]:
        """Verify candidates by patching and checking if output flips."""

        print(f"\n{'=' * 70}")
        print("CAUSAL VERIFICATION")
        print(f"{'=' * 70}")
        print(f"Testing {min(len(candidates), max_candidates)} candidates...")
        print(f"Target: flip output to '{target_digit}'")
        print()

        original_output = self.get_output(closed_prompt)

        results = []
        gate_neurons = []

        for i, cand in enumerate(candidates[:max_candidates]):
            # Patch closed prompt with open prompt's value
            patched_output = self.get_output_with_patch(
                closed_prompt,
                cand.layer,
                cand.neuron,
                cand.open_value,
            )

            flipped = target_digit in patched_output

            result = PatchResult(
                layer=cand.layer,
                neuron=cand.neuron,
                original_output=original_output,
                patched_output=patched_output,
                flipped=flipped,
                target_digit=target_digit,
            )
            results.append(result)

            status = "🔥 GATE!" if flipped else "  -"
            print(
                f"{status} L{cand.layer:2d} N{cand.neuron:4d}: "
                f"{cand.closed_value:+.2f} → {cand.open_value:+.2f} "
                f"(Δ={cand.difference:+.2f}) "
                f"output: {repr(patched_output)}"
            )

            if flipped:
                gate_neurons.append((cand, result))

        return results, gate_neurons

    def get_output_with_multi_patch(
        self,
        prompt: str,
        patches: list[tuple[int, int, float]],  # [(layer, neuron, value), ...]
    ) -> str:
        """Get model output with multiple neurons patched."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        # Group patches by layer
        patches_by_layer = {}
        for layer_idx, neuron_idx, value in patches:
            if layer_idx not in patches_by_layer:
                patches_by_layer[layer_idx] = []
            patches_by_layer[layer_idx].append((neuron_idx, value))

        for idx, layer in enumerate(layers):
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            # Apply patches at this layer
            if idx in patches_by_layer:
                h_list = h.tolist()
                for neuron_idx, value in patches_by_layer[idx]:
                    h_list[0][-1][neuron_idx] = value
                h = mx.array(h_list)

        h_n = norm(h) if norm else h
        logits = head(h_n)
        if hasattr(logits, "logits"):
            logits = logits.logits

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def try_multi_neuron_patch(
        self,
        candidates: list[NeuronCandidate],
        closed_prompt: str,
        target_digit: str,
        max_neurons: int = 10,
    ):
        """Try patching multiple neurons together."""

        print(f"\n{'=' * 70}")
        print("MULTI-NEURON PATCHING")
        print(f"{'=' * 70}")

        original_output = self.get_output(closed_prompt)
        print(f"Original output: {repr(original_output)}")
        print(f"Target: '{target_digit}'")
        print()

        # Try incrementally adding more neurons
        for n in range(1, max_neurons + 1):
            # Use top N candidates
            top_n = candidates[:n]

            patches = [(c.layer, c.neuron, c.open_value) for c in top_n]

            output = self.get_output_with_multi_patch(closed_prompt, patches)
            success = target_digit in output

            status = "🔥 SUCCESS!" if success else "  -"
            neurons_str = ", ".join([f"L{c.layer}N{c.neuron}" for c in top_n])
            print(f"{status} Top {n:2d} neurons: output={repr(output):5} [{neurons_str}]")

            if success:
                print(f"\n✅ Found minimal gate: {n} neurons needed")
                print("   Neurons:")
                for c in top_n:
                    print(
                        f"      L{c.layer} N{c.neuron}: {c.closed_value:+.1f} → {c.open_value:+.1f}"
                    )
                return top_n

        print(f"\n❌ Could not flip gate with top {max_neurons} neurons")
        return None

    def try_layer_patch(
        self,
        open_prompt: str,
        closed_prompt: str,
        layers: list[int],
        target_digit: str,
    ):
        """Try patching entire layers."""

        print(f"\n{'=' * 70}")
        print("FULL LAYER PATCHING")
        print(f"{'=' * 70}")

        h_open = self.get_hidden_states(open_prompt)
        original_output = self.get_output(closed_prompt)
        print(f"Original output: {repr(original_output)}")
        print()

        for layer in layers:
            # Patch ALL neurons at this layer
            h_open_layer = h_open[layer].tolist()
            patches = [(layer, i, v) for i, v in enumerate(h_open_layer)]

            output = self.get_output_with_multi_patch(closed_prompt, patches)
            success = target_digit in output

            status = "🔥 GATE LAYER!" if success else "  -"
            print(f"{status} Patch all of L{layer}: output={repr(output)}")

            if success:
                return layer

        return None

    def run_analysis(
        self,
        open_prompt: str,
        closed_prompt: str,
        layers: list[int] | None = None,
    ):
        """Run full gate neuron analysis."""

        # Detect expected answer
        answer = auto_detect_answer(open_prompt)
        if answer:
            target_digit = answer[0]
            print(f"\nExpected answer: {answer}, target first digit: '{target_digit}'")
        else:
            target_digit = None
            print("\nCouldn't auto-detect answer")

        # Find candidates
        candidates = self.find_gate_neurons(open_prompt, closed_prompt, layers)

        # Show top candidates
        print(f"\n{'=' * 70}")
        print("TOP CANDIDATES (by activation difference)")
        print(f"{'=' * 70}")
        for i, c in enumerate(candidates[:30]):
            print(
                f"{i + 1:2d}. L{c.layer:2d} N{c.neuron:4d}: "
                f"open={c.open_value:+.3f} closed={c.closed_value:+.3f} "
                f"Δ={c.difference:+.3f} ({c.direction})"
            )

        # Verify if we have a target
        if target_digit:
            results, gate_neurons = self.verify_candidates(candidates, closed_prompt, target_digit)

            # Summary
            print(f"\n{'=' * 70}")
            print("SINGLE NEURON SUMMARY")
            print(f"{'=' * 70}")
            print(f"Candidates tested: {len(results)}")
            print(f"Gate neurons found: {len(gate_neurons)}")

            if gate_neurons:
                print(f"\n🔥 GATE NEURONS (patching these flips output to '{target_digit}'):")
                for cand, result in gate_neurons:
                    print(f"   Layer {cand.layer}, Neuron {cand.neuron}")
                    print(f"      Closed value: {cand.closed_value:+.3f}")
                    print(f"      Open value:   {cand.open_value:+.3f}")
                    print(f"      Difference:   {cand.difference:+.3f}")
            else:
                print("\n❌ No single neuron flips the gate.")
                print("   Trying multi-neuron and layer-level patching...")

                # Try multi-neuron patching
                self.try_multi_neuron_patch(candidates, closed_prompt, target_digit, max_neurons=20)

                # Try full layer patching
                if layers is None:
                    num_layers = len(self._get_layers())
                    layers = list(
                        range(max(0, num_layers // 2 - 5), min(num_layers, num_layers // 2 + 10))
                    )
                self.try_layer_patch(open_prompt, closed_prompt, layers, target_digit)

        return candidates


async def main(
    model_id: str,
    open_prompt: str,
    closed_prompt: str,
    layers: list[int] | None = None,
):
    """Run gate neuron finder."""
    finder = await GateNeuronFinder.from_pretrained(model_id)
    finder.run_analysis(open_prompt, closed_prompt, layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find gate neurons that control answer output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model ID",
    )
    parser.add_argument(
        "--open",
        "-o",
        default="100 - 37 = ",
        help="Prompt where gate is OPEN (correct output)",
    )
    parser.add_argument(
        "--closed",
        "-c",
        default="100 - 37 =",
        help="Prompt where gate is CLOSED (wrong output)",
    )
    parser.add_argument(
        "--layers",
        "-l",
        default=None,
        help="Comma-separated layers to search (e.g., '18,19,20,21,22,23')",
    )

    args = parser.parse_args()

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    asyncio.run(main(args.model, args.open, args.closed, layers))
