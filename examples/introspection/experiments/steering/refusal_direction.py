#!/usr/bin/env python3
"""
Refusal Direction Finder: Test if L22 contains a "refusal subspace".

Hypothesis: Layer 22 neurons encode a refusal direction that competes
with the computation direction.

Tests:
1. Inject closed L22 → open case: Does it cause refusal?
2. Inject open L22 → closed case: Does it enable output?
3. Compute steering vector: closed_L22 - open_L22
4. Apply steering to new prompts: Does it induce refusal?

Usage:
    uv run python examples/introspection/refusal_direction.py \
        --open "100 - 37 = " \
        --closed "100 - 37 ="
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class RefusalTest:
    """Result of a refusal direction test."""

    test_name: str
    prompt: str
    original_output: str
    modified_output: str
    layer: int
    effect: str  # "induced_refusal", "removed_refusal", "no_change"


class RefusalDirectionFinder:
    """Find and test refusal directions in transformer layers."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "RefusalDirectionFinder":
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

    def get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at a specific layer."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            if idx == layer:
                return h[0, -1, :]  # [hidden_dim]

        return h[0, -1, :]

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

    def run_with_layer_patch(
        self,
        prompt: str,
        patch_layer: int,
        patch_hidden: mx.array,
    ) -> str:
        """Run model with entire layer hidden state patched."""
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

            # Patch at specified layer
            if idx == patch_layer:
                # Replace last position with patch
                h_list = h.tolist()
                h_list[0][-1] = patch_hidden.tolist()
                h = mx.array(h_list)

        h_n = norm(h) if norm else h
        logits = head(h_n)
        if hasattr(logits, "logits"):
            logits = logits.logits

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def run_with_steering(
        self,
        prompt: str,
        steering_layer: int,
        steering_vector: mx.array,
        scale: float = 1.0,
    ) -> str:
        """Run model with steering vector added at a layer."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        emb_scale = self._get_scale()

        h = embed(input_ids)
        if emb_scale:
            h = h * emb_scale

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

            # Add steering at specified layer
            if idx == steering_layer:
                h_list = h.tolist()
                steering_list = steering_vector.tolist()
                for i in range(len(steering_list)):
                    h_list[0][-1][i] += scale * steering_list[i]
                h = mx.array(h_list)

        h_n = norm(h) if norm else h
        logits = head(h_n)
        if hasattr(logits, "logits"):
            logits = logits.logits

        top_idx = int(mx.argmax(logits[0, -1, :]))
        return self.tokenizer.decode([top_idx])

    def run_experiments(
        self,
        open_prompt: str,
        closed_prompt: str,
        test_layers: list[int] | None = None,
    ):
        """Run all refusal direction experiments."""

        print(f"\n{'=' * 70}")
        print("REFUSAL DIRECTION EXPERIMENTS")
        print(f"{'=' * 70}")
        print(f"Open prompt (works):   {repr(open_prompt)}")
        print(f"Closed prompt (fails): {repr(closed_prompt)}")

        # Get baseline outputs
        open_output = self.get_output(open_prompt)
        closed_output = self.get_output(closed_prompt)
        print("\nBaseline outputs:")
        print(f"  Open → {repr(open_output)}")
        print(f"  Closed → {repr(closed_output)}")

        if test_layers is None:
            num_layers = len(self._get_layers())
            # Auto-scale layers based on model depth
            # Test ~60-80% of model depth (where gate typically is)
            if num_layers <= 16:
                test_layers = [2, 4, 6, 8, 10, 12, 14]
            elif num_layers <= 28:
                test_layers = [4, 8, 12, 16, 18, 20, 22]
            else:
                test_layers = [4, 12, 20, 22, 24, 26, 28]

        print(f"\nTesting layers: {test_layers}")

        # Experiment 1: Inject closed hidden state into open case
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 1: Inject CLOSED L[n] → OPEN case")
        print("(Does injecting 'refusal direction' cause working case to fail?)")
        print(f"{'=' * 70}")

        for layer in test_layers:
            h_closed = self.get_hidden_state(closed_prompt, layer)
            patched_output = self.run_with_layer_patch(open_prompt, layer, h_closed)

            effect = ""
            if patched_output == open_output:
                effect = "no change"
            elif patched_output == closed_output:
                effect = "🔥 INDUCED REFUSAL"
            else:
                effect = f"changed to {repr(patched_output)}"

            print(f"  L{layer}: {repr(open_output)} → {repr(patched_output)} ({effect})")

        # Experiment 2: Inject open hidden state into closed case
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 2: Inject OPEN L[n] → CLOSED case")
        print("(Does injecting 'computation direction' make broken case work?)")
        print(f"{'=' * 70}")

        for layer in test_layers:
            h_open = self.get_hidden_state(open_prompt, layer)
            patched_output = self.run_with_layer_patch(closed_prompt, layer, h_open)

            effect = ""
            if patched_output == closed_output:
                effect = "no change"
            elif patched_output == open_output:
                effect = "🔥 REMOVED REFUSAL"
            else:
                effect = f"changed to {repr(patched_output)}"

            print(f"  L{layer}: {repr(closed_output)} → {repr(patched_output)} ({effect})")

        # Experiment 3: Compute and apply steering vector
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 3: Steering vector = CLOSED - OPEN")
        print("(Does adding refusal direction to new prompts cause refusal?)")
        print(f"{'=' * 70}")

        test_prompts = [
            "50 + 25 = ",
            "10 * 10 = ",
            "200 - 50 = ",
        ]

        for layer in [22, 24]:  # Focus on late layers
            h_open = self.get_hidden_state(open_prompt, layer)
            h_closed = self.get_hidden_state(closed_prompt, layer)

            # Refusal direction: closed - open
            refusal_vector = h_closed - h_open

            print(f"\n  Layer {layer}:")
            for prompt in test_prompts:
                original = self.get_output(prompt)

                # Try different scales
                for scale in [0.5, 1.0, 2.0]:
                    steered = self.run_with_steering(prompt, layer, refusal_vector, scale)

                    if steered != original:
                        print(
                            f"    {repr(prompt)}: {repr(original)} → {repr(steered)} (scale={scale})"
                        )

        # Experiment 4: Reverse steering - add computation direction to broken case
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 4: Steering vector = OPEN - CLOSED")
        print("(Does adding computation direction fix broken prompts?)")
        print(f"{'=' * 70}")

        broken_prompts = [
            "50 + 25 =",  # No trailing space
            "10 * 10 =",
            "200 - 50 =",
        ]

        for layer in [22, 24]:
            h_open = self.get_hidden_state(open_prompt, layer)
            h_closed = self.get_hidden_state(closed_prompt, layer)

            # Computation direction: open - closed
            computation_vector = h_open - h_closed

            print(f"\n  Layer {layer}:")
            for prompt in broken_prompts:
                original = self.get_output(prompt)

                for scale in [0.5, 1.0, 2.0]:
                    steered = self.run_with_steering(prompt, layer, computation_vector, scale)

                    if steered != original:
                        print(
                            f"    {repr(prompt)}: {repr(original)} → {repr(steered)} (scale={scale})"
                        )

        # Experiment 5: Early layer steering - enable computation
        num_layers = len(self._get_layers())
        if num_layers <= 16:
            early_layers = [2, 4, 6]
        elif num_layers <= 28:
            early_layers = [2, 4, 8]
        else:
            early_layers = [4, 8, 12]

        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT 5: Early steering ({', '.join(f'L{l}' for l in early_layers)})")
        print("(Can early steering enable correct computation?)")
        print(f"{'=' * 70}")

        for layer in early_layers:
            h_open = self.get_hidden_state(open_prompt, layer)
            h_closed = self.get_hidden_state(closed_prompt, layer)

            computation_vector = h_open - h_closed

            print(f"\n  Layer {layer}:")
            for prompt in broken_prompts:
                original = self.get_output(prompt)

                for scale in [1.0, 2.0, 3.0]:
                    steered = self.run_with_steering(prompt, layer, computation_vector, scale)

                    if steered != original:
                        print(
                            f"    {repr(prompt)}: {repr(original)} → {repr(steered)} (scale={scale})"
                        )


async def main(
    model_id: str,
    open_prompt: str,
    closed_prompt: str,
    layers: list[int] | None = None,
):
    """Run refusal direction experiments."""
    finder = await RefusalDirectionFinder.from_pretrained(model_id)
    finder.run_experiments(open_prompt, closed_prompt, layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and test refusal directions",
    )
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--open", "-o", default="100 - 37 = ")
    parser.add_argument("--closed", "-c", default="100 - 37 =")
    parser.add_argument(
        "--layers", "-l", default=None, help="Comma-separated layers (e.g., '20,21,22,23,24')"
    )

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    asyncio.run(main(args.model, args.open, args.closed, layers))
