#!/usr/bin/env python3
"""
Gemma Full Layer Ablation Study.

Previous findings:
- MLP ablation: 2000 neurons (20%) = 0% drop
- Attention ablation: ALL 8 heads at one layer = 0% drop

This script tests:
1. Ablating ENTIRE layers (skip residual connection)
2. Ablating multiple layers simultaneously
3. Finding the minimum circuit required for multiplication

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_layer_ablation.py
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class LayerAblationStudy:
    """Ablate entire layers to find minimum circuit."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

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

        embed_scale = float(self.hidden_size**0.5)

        return layers, embed, norm, head, embed_scale

    def generate(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate tokens from a prompt."""
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

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)) :].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def generate_with_layer_skip(
        self,
        prompt: str,
        layers_to_skip: set[int],
        max_tokens: int = 5,
    ) -> str:
        """Generate while completely skipping certain layers (no residual)."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
            seq_len = input_ids.shape[1]
            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                if i in layers_to_skip:
                    # COMPLETELY SKIP this layer - just pass through
                    continue

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

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)) :].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def test_multiplication_accuracy(
        self,
        skip_layers: set[int] | None = None,
    ) -> tuple[float, list[dict]]:
        """Test multiplication accuracy with optional layer skipping."""
        test_cases = [
            (2, 3, 6),
            (3, 4, 12),
            (5, 6, 30),
            (7, 8, 56),
            (9, 9, 81),
            (4, 7, 28),
            (6, 8, 48),
            (3, 9, 27),
            (5, 5, 25),
            (8, 9, 72),
        ]

        results = []
        correct = 0

        for a, b, expected in test_cases:
            prompt = f"{a} * {b} = "

            if skip_layers:
                output = self.generate_with_layer_skip(prompt, skip_layers, max_tokens=5)
            else:
                output = self.generate(prompt, max_tokens=5)

            is_correct = str(expected) in output

            if is_correct:
                correct += 1

            results.append(
                {
                    "prompt": prompt,
                    "expected": str(expected),
                    "output": output,
                    "correct": is_correct,
                }
            )

        accuracy = correct / len(test_cases)
        return accuracy, results

    def run_ablation_study(self):
        """Run the layer ablation study."""
        self.load_model()

        print("\n" + "=" * 70)
        print("FULL LAYER ABLATION STUDY")
        print("=" * 70)

        # Baseline
        print("\n1. Computing baseline accuracy...")
        baseline_acc, baseline_results = self.test_multiplication_accuracy()
        print(f"   Baseline accuracy: {baseline_acc:.1%}")

        # Single layer skip
        print("\n2. Single layer ablation (complete skip)...")
        print(f"\n{'Layer':<10} {'Accuracy':<12} {'Drop'}")
        print("-" * 35)

        single_results = []
        for layer in range(self.num_layers):
            acc, _ = self.test_multiplication_accuracy(skip_layers={layer})
            drop = baseline_acc - acc

            if drop > 0:
                print(f"L{layer:<9} {acc:>10.1%} {drop:>+9.1%}")

            single_results.append(
                {
                    "layer": layer,
                    "accuracy": acc,
                    "drop": drop,
                }
            )

        # Impactful layers
        impactful = [r for r in single_results if r["drop"] > 0]
        if not impactful:
            print("   No single layer skip caused accuracy drop!")

        # Multi-layer skip
        print("\n3. Multi-layer ablation tests...")

        # Skip first N layers
        print("\n   Skipping FIRST N layers (early processing):")
        for n in [5, 10, 15, 20, 25]:
            skip = set(range(n))
            acc, _ = self.test_multiplication_accuracy(skip_layers=skip)
            drop = baseline_acc - acc
            print(f"   Skip L0-L{n - 1}: {acc:.1%} (drop: {drop:+.1%})")

        # Skip last N layers
        print("\n   Skipping LAST N layers (late processing):")
        for n in [5, 10, 15, 20, 25]:
            skip = set(range(self.num_layers - n, self.num_layers))
            acc, _ = self.test_multiplication_accuracy(skip_layers=skip)
            drop = baseline_acc - acc
            print(
                f"   Skip L{self.num_layers - n}-L{self.num_layers - 1}: {acc:.1%} (drop: {drop:+.1%})"
            )

        # Skip middle layers
        print("\n   Skipping MIDDLE layers (computation phase):")
        for start in [10, 15]:
            for n in [5, 10]:
                skip = set(range(start, start + n))
                acc, _ = self.test_multiplication_accuracy(skip_layers=skip)
                drop = baseline_acc - acc
                print(f"   Skip L{start}-L{start + n - 1}: {acc:.1%} (drop: {drop:+.1%})")

        # Find minimum layers needed
        print("\n4. Finding MINIMUM layers needed...")

        # Keep only every Nth layer
        for step in [2, 3, 4, 5]:
            keep = set(range(0, self.num_layers, step))
            skip = set(range(self.num_layers)) - keep
            acc, results = self.test_multiplication_accuracy(skip_layers=skip)
            drop = baseline_acc - acc
            kept_pct = len(keep) / self.num_layers * 100
            print(
                f"   Keep every {step}th layer ({len(keep)} layers, {kept_pct:.0f}%): {acc:.1%} (drop: {drop:+.1%})"
            )

        # Detailed test with large skip
        print("\n5. Detailed test: Skip 50% of layers...")
        skip_half = set(range(0, self.num_layers, 2))  # Skip even layers
        acc, results = self.test_multiplication_accuracy(skip_layers=skip_half)
        print(f"   Skipping even layers (L0,L2,L4,...): {acc:.1%}")

        for r in results[:5]:
            status = "✓" if r["correct"] else "✗"
            print(f"   {status} {r['prompt']} -> {r['output'][:20]}...")

        # Summary
        print("\n" + "=" * 70)
        print("LAYER ABLATION SUMMARY")
        print("=" * 70)

        print("\nKey findings:")
        if impactful:
            print(f"  - {len(impactful)} layers individually critical")
            for r in sorted(impactful, key=lambda x: -x["drop"])[:5]:
                print(f"    L{r['layer']}: {r['drop']:+.1%} drop")
        else:
            print("  - NO single layer is individually critical")
            print("  - Computation is distributed across ALL layers")

        print("\nComparison:")
        print("  - Single neuron ablation: 0% drop")
        print("  - 20% MLP neurons ablated: 0% drop")
        print("  - All 8 attention heads at one layer: 0% drop")
        print("  - Single layer completely skipped: Results above")

        # Save
        results = {
            "model": self.model_id,
            "num_layers": self.num_layers,
            "baseline_accuracy": baseline_acc,
            "single_layer_results": single_results,
        }

        output_path = Path("gemma_discovery_cache/layer_ablation.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    study = LayerAblationStudy()
    study.run_ablation_study()


if __name__ == "__main__":
    main()
