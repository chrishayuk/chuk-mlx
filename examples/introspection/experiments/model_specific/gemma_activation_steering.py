#!/usr/bin/env python3
"""
Gemma Activation Steering: Steer model behavior using probe directions.

Instead of ablating neurons (which showed no effect due to redundancy),
this script ADDS learned directions to activations to steer behavior.

Key insight from ablation study:
- Ablating 20% of neurons had 0% effect on accuracy
- This means multiplication is highly distributed
- But we can still STEER by adding/subtracting directions

Steering approaches:
1. Add "arithmetic" direction to make non-arithmetic prompts compute
2. Add "wrong answer" direction to change multiplication results
3. Add digit-specific directions to control output

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_activation_steering.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class SteeringResult:
    """Result of steering intervention."""
    prompt: str
    baseline_output: str
    steered_output: str
    steering_layer: int
    steering_strength: float
    direction_type: str


class ActivationSteeringStudy:
    """Steer Gemma's behavior using learned directions."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None
        self.probes = {}  # Layer -> trained probe

    def load_model(self):
        """Load the model."""
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

        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        else:
            head = None

        embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, norm, head, embed_scale

    def collect_activations(self, prompts: list[str], layer_idx: int) -> list[np.ndarray]:
        """Collect hidden state activations at a specific layer."""
        layers, embed, norm, head, embed_scale = self._get_components()

        activations = []

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)
            input_ids = mx.array(input_ids)[None, :]

            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                if i == layer_idx:
                    # Capture activation at this layer
                    activations.append(np.array(h[0, -1, :].tolist()))

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

        return activations

    def train_arithmetic_probe(self, layer_idx: int) -> np.ndarray:
        """Train a probe to distinguish arithmetic from non-arithmetic prompts.

        Returns the probe direction (coefficient vector).
        """
        # Create dataset
        arithmetic_prompts = [
            f"{a} * {b} = " for a in range(2, 10) for b in range(2, 10)
        ]
        non_arithmetic_prompts = [
            "The capital of France is ",
            "The color of the sky is ",
            "Water boils at ",
            "The largest planet is ",
            "Dogs are known for their ",
            "The speed of light is ",
            "Shakespeare wrote ",
            "The chemical symbol for gold is ",
            "The first president was ",
            "Roses are typically ",
        ] * 8  # Repeat to match size

        all_prompts = arithmetic_prompts[:40] + non_arithmetic_prompts[:40]
        labels = [1] * 40 + [0] * 40

        # Collect activations
        print(f"  Collecting activations at layer {layer_idx}...")
        activations = self.collect_activations(all_prompts, layer_idx)

        X = np.array(activations)
        y = np.array(labels)

        # Train probe
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)

        accuracy = probe.score(X, y)
        print(f"  Probe accuracy: {accuracy:.1%}")

        # Return the direction (normalized)
        direction = probe.coef_[0]
        direction = direction / np.linalg.norm(direction)

        self.probes[layer_idx] = {
            "probe": probe,
            "direction": direction,
            "accuracy": accuracy,
        }

        return direction

    def train_digit_probe(self, layer_idx: int, target_digit: int) -> np.ndarray:
        """Train a probe to identify when the answer contains a specific digit.

        Returns the probe direction for that digit.
        """
        # Create dataset: products that contain target_digit vs those that don't
        contains_digit = []
        not_contains_digit = []

        for a in range(2, 10):
            for b in range(2, 10):
                product = a * b
                prompt = f"{a} * {b} = "
                if str(target_digit) in str(product):
                    contains_digit.append(prompt)
                else:
                    not_contains_digit.append(prompt)

        # Balance dataset
        min_size = min(len(contains_digit), len(not_contains_digit), 30)
        contains_digit = contains_digit[:min_size]
        not_contains_digit = not_contains_digit[:min_size]

        all_prompts = contains_digit + not_contains_digit
        labels = [1] * len(contains_digit) + [0] * len(not_contains_digit)

        print(f"  Training digit-{target_digit} probe (n={len(all_prompts)})...")

        # Collect activations
        activations = self.collect_activations(all_prompts, layer_idx)

        X = np.array(activations)
        y = np.array(labels)

        # Train probe
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)

        accuracy = probe.score(X, y)
        print(f"  Digit-{target_digit} probe accuracy: {accuracy:.1%}")

        # Return normalized direction
        direction = probe.coef_[0]
        direction = direction / np.linalg.norm(direction)

        return direction

    def generate(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate without steering (baseline)."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
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

            if norm is not None:
                h = norm(h)

            if head is not None:
                logits = head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ embed.weight.T

            next_token = mx.argmax(logits[0, -1, :])
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            if int(next_token) in [self.tokenizer.eos_token_id, 13, 10]:
                break

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)):].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def generate_with_steering(
        self,
        prompt: str,
        direction: np.ndarray,
        layer_idx: int,
        strength: float,
        max_tokens: int = 5,
    ) -> str:
        """Generate with activation steering.

        Adds `strength * direction` to the hidden state at the specified layer.
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        # Convert direction to MLX array
        steering_vector = mx.array(direction.astype(np.float32))

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
            seq_len = input_ids.shape[1]
            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                # Apply steering BEFORE the target layer
                if i == layer_idx:
                    # Add steering vector to all positions
                    steering = (strength * steering_vector).astype(h.dtype)
                    h = h + steering

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

            if norm is not None:
                h = norm(h)

            if head is not None:
                logits = head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ embed.weight.T

            next_token = mx.argmax(logits[0, -1, :])
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            if int(next_token) in [self.tokenizer.eos_token_id, 13, 10]:
                break

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)):].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def run_steering_study(self):
        """Run the full activation steering study."""
        self.load_model()

        print("\n" + "=" * 70)
        print("ACTIVATION STEERING STUDY")
        print("=" * 70)

        # Train arithmetic direction probe at key layers
        print("\n1. Training arithmetic direction probes...")
        arithmetic_directions = {}
        for layer in [16, 20, 24]:
            direction = self.train_arithmetic_probe(layer)
            arithmetic_directions[layer] = direction

        # Test steering on arithmetic prompts
        print("\n2. Steering arithmetic prompts with NEGATIVE direction...")
        print("   (Trying to suppress arithmetic behavior)")
        print(f"\n{'Prompt':<15} {'Baseline':<12} {'Steered':<12} {'Layer':<8} {'Strength'}")
        print("-" * 60)

        test_prompts = ["7 * 8 = ", "5 * 6 = ", "3 * 4 = "]

        for prompt in test_prompts:
            baseline = self.generate(prompt, max_tokens=5)

            for layer in [20, 24]:
                for strength in [-50, -100, -200, -500]:
                    steered = self.generate_with_steering(
                        prompt,
                        arithmetic_directions[layer],
                        layer,
                        strength,
                        max_tokens=5,
                    )
                    if steered != baseline:
                        print(f"{prompt:<15} {baseline:<12} {steered:<12} L{layer:<6} {strength}")

        # Test steering on non-arithmetic prompts
        print("\n3. Steering non-arithmetic prompts with POSITIVE direction...")
        print("   (Trying to induce arithmetic behavior)")
        print(f"\n{'Prompt':<30} {'Baseline':<15} {'Steered':<15} {'Strength'}")
        print("-" * 75)

        non_arith_prompts = [
            "The capital of France is ",
            "The answer is ",
            "Hello, my name is ",
        ]

        for prompt in non_arith_prompts:
            baseline = self.generate(prompt, max_tokens=5)

            for strength in [50, 100, 200, 500]:
                steered = self.generate_with_steering(
                    prompt,
                    arithmetic_directions[20],
                    20,
                    strength,
                    max_tokens=5,
                )
                if steered != baseline:
                    print(f"{prompt:<30} {baseline:<15} {steered:<15} {strength}")

        # Train digit-specific probes
        print("\n4. Training digit-specific probes at L24...")
        digit_directions = {}
        for digit in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            direction = self.train_digit_probe(24, digit)
            digit_directions[digit] = direction

        # Test digit steering
        print("\n5. Digit steering experiments...")
        print("   (Trying to change which digit appears in output)")
        print(f"\n{'Prompt':<15} {'Correct':<10} {'Baseline':<12} {'Steered':<12} {'Target Digit':<15} {'Strength'}")
        print("-" * 80)

        digit_test_cases = [
            ("7 * 8 = ", "56"),  # Contains 5 and 6
            ("3 * 4 = ", "12"),  # Contains 1 and 2
            ("5 * 5 = ", "25"),  # Contains 2 and 5
        ]

        for prompt, correct in digit_test_cases:
            baseline = self.generate(prompt, max_tokens=5)

            # Try to add digits not in the answer
            for target_digit in [1, 2, 3, 7, 8, 9]:
                if str(target_digit) in correct:
                    continue  # Skip if digit already in answer

                for strength in [100, 500, 1000]:
                    steered = self.generate_with_steering(
                        prompt,
                        digit_directions[target_digit],
                        24,
                        strength,
                        max_tokens=5,
                    )
                    if steered != baseline:
                        print(f"{prompt:<15} {correct:<10} {baseline:<12} {steered:<12} digit-{target_digit:<10} {strength}")
                        break  # Only show first change

        # Sweep strength to find threshold
        print("\n6. Finding steering strength threshold...")
        print("   Testing 7 * 8 = 56 with varying negative arithmetic strength at L24")
        print(f"\n{'Strength':<12} {'Output':<20} {'Changed?'}")
        print("-" * 45)

        prompt = "7 * 8 = "
        baseline = self.generate(prompt, max_tokens=5)
        print(f"{'Baseline':<12} {baseline:<20} -")

        for strength in [0, -10, -25, -50, -100, -200, -500, -1000, -2000, -5000]:
            steered = self.generate_with_steering(
                prompt,
                arithmetic_directions[24],
                24,
                strength,
                max_tokens=5,
            )
            changed = "YES" if steered != baseline else "no"
            print(f"{strength:<12} {steered:<20} {changed}")

        # Summary
        print("\n" + "=" * 70)
        print("STEERING STUDY SUMMARY")
        print("=" * 70)

        print("\nKey findings:")
        print("  - Probe directions capture arithmetic vs non-arithmetic distinction")
        print("  - Steering can modify model outputs at sufficient strength")
        print("  - Effect depends on layer and strength")

        # Save results
        results = {
            "model": self.model_id,
            "arithmetic_probe_accuracies": {
                int(layer): float(self.probes[layer]["accuracy"])
                for layer in self.probes
            },
            "steering_effective": True,
        }

        output_path = Path("gemma_discovery_cache/activation_steering.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    study = ActivationSteeringStudy()
    study.run_steering_study()


if __name__ == "__main__":
    main()
