#!/usr/bin/env python3
"""
Gemma Neuron Ablation Study: Test if identified neurons are causally important.

This script ablates the key neurons identified in circuit analysis
and measures the impact on multiplication accuracy.

Key neurons to test (from circuit_via_probes.py):
- Neuron 19: ARITHMETIC- (suppressor), active in L20, L24, L28
- Neuron 1698: ARITHMETIC+ (activator), active in L20, L24, L28
- Neuron 2309: ARITHMETIC- (suppressor), active in L20, L24, L28

If ablating these neurons changes multiplication behavior, they are
causally important (not just correlated).

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_neuron_ablation.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class AblationResult:
    """Result of ablating neurons."""

    neurons_ablated: list[int]
    layers_ablated: list[int]
    baseline_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    sample_comparisons: list[dict]  # Before/after for specific problems


class NeuronAblationStudy:
    """Ablate specific neurons and measure impact on multiplication."""

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

        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
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

            # Stop on newline or special tokens
            if int(next_token) in [self.tokenizer.eos_token_id, 13, 10]:  # EOS, \r, \n
                break

        output_ids = input_ids[0, len(self.tokenizer.encode(prompt)) :].tolist()
        return self.tokenizer.decode(output_ids).strip()

    def generate_with_neuron_ablation(
        self,
        prompt: str,
        neurons: list[int],
        layer_idx: int | list[int],
        max_tokens: int = 5,
    ) -> str:
        """
        Generate with specific neurons ablated (zeroed) in a layer's MLP.

        This modifies the MLP intermediate activations during forward pass.

        Args:
            prompt: Input prompt
            neurons: List of neuron indices to ablate
            layer_idx: Single layer index OR list of layer indices to ablate
            max_tokens: Maximum tokens to generate
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        # Support single layer or multiple layers
        if isinstance(layer_idx, int):
            layers_to_ablate = {layer_idx}
        else:
            layers_to_ablate = set(layer_idx)

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        for _ in range(max_tokens):
            seq_len = input_ids.shape[1]
            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                if i in layers_to_ablate:
                    # Manual forward with neuron ablation
                    h = self._forward_with_ablation(layer, h, mask, neurons)
                else:
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

    def _forward_with_ablation(
        self,
        layer: nn.Module,
        h: mx.array,
        mask: mx.array,
        neurons_to_ablate: list[int],
    ) -> mx.array:
        """Forward through a layer with specific MLP neurons zeroed.

        Based on Gemma MLP architecture:
        output = down_proj(gelu_approx(gate_proj(x)) * up_proj(x))

        We ablate neurons in the intermediate representation (after gelu*up).
        """
        # Attention with pre/post normalization (Gemma has 4 norms per block)
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

        # Post-attention norm (Gemma-specific)
        if hasattr(layer, "post_attention_layernorm"):
            attn_out = layer.post_attention_layernorm(attn_out)

        # Residual connection
        h = h + attn_out

        # MLP with ablation
        # Gemma uses pre-feedforward norm
        if hasattr(layer, "pre_feedforward_layernorm"):
            mlp_input = layer.pre_feedforward_layernorm(h)
        else:
            mlp_input = h

        mlp = layer.mlp

        # Get gate and up projections - use gelu_approx like Gemma
        if hasattr(mlp, "gate_proj"):
            gate = mlp.gate_proj(mlp_input)
            up = mlp.up_proj(mlp_input)

            # Apply activation (gelu_approx, matching Gemma's architecture)
            gate_activated = nn.gelu_approx(gate)

            # MLP intermediate
            mlp_intermediate = gate_activated * up

            # ABLATE: Zero out specific neurons
            # Create mask: 1 everywhere except 0 at ablated neurons
            neurons_set = set(neurons_to_ablate)
            ablation_mask = mx.array(
                [0.0 if i in neurons_set else 1.0 for i in range(mlp_intermediate.shape[-1])]
            )
            ablation_mask = ablation_mask.astype(mlp_intermediate.dtype)

            # Verify ablation is happening (debug)
            before_sum = float(mx.sum(mx.abs(mlp_intermediate)).item())
            mlp_intermediate = mlp_intermediate * ablation_mask
            after_sum = float(mx.sum(mx.abs(mlp_intermediate)).item())

            # Debug: uncomment to verify ablation
            # print(f"Before: {before_sum:.2f}, After: {after_sum:.2f}, Ablated: {len(neurons_to_ablate)}")

            # Down projection
            mlp_out = mlp.down_proj(mlp_intermediate)
        else:
            # Simple MLP (no ablation possible without more info)
            mlp_out = mlp(mlp_input)

        # Post-feedforward norm (Gemma-specific)
        if hasattr(layer, "post_feedforward_layernorm"):
            mlp_out = layer.post_feedforward_layernorm(mlp_out)

        # Residual connection
        h = h + mlp_out
        return h

    def test_multiplication_accuracy(
        self,
        ablate_neurons: list[int] | None = None,
        ablate_layer: int | None = None,
    ) -> tuple[float, list[dict]]:
        """
        Test multiplication accuracy, optionally with neurons ablated.

        Returns: (accuracy, detailed_results)
        """
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

            if ablate_neurons and ablate_layer is not None:
                output = self.generate_with_neuron_ablation(
                    prompt, ablate_neurons, ablate_layer, max_tokens=5
                )
            else:
                output = self.generate(prompt, max_tokens=5)

            # Check if correct
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

    def verify_ablation_works(self):
        """Quick test to verify ablation is actually zeroing neurons."""
        print("\nVerifying ablation mechanism...")
        layers, embed, norm, head, embed_scale = self._get_components()

        prompt = "7 * 8 = "
        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        layer_idx = 20
        layer = layers[layer_idx]

        # Run layer normally
        out_normal = layer(h, mask=mask)
        if hasattr(out_normal, "hidden_states"):
            h_normal = out_normal.hidden_states
        else:
            h_normal = out_normal[0] if isinstance(out_normal, tuple) else out_normal

        # Run layer with ablation
        neurons_to_ablate = list(range(1000))  # Ablate first 1000 neurons
        h_ablated = self._forward_with_ablation(layer, h, mask, neurons_to_ablate)

        # Compare outputs
        diff = float(mx.mean(mx.abs(h_normal - h_ablated)).item())
        print(f"  Layer {layer_idx} output difference with 1000 neurons ablated: {diff:.6f}")

        if diff > 0.001:
            print("  ✓ Ablation IS modifying the output")
        else:
            print("  ✗ WARNING: Ablation may not be working correctly!")

        return diff > 0.001

    def run_ablation_study(self):
        """Run the full ablation study."""
        self.load_model()

        # First verify ablation mechanism works
        self.verify_ablation_works()

        # Key neurons from circuit analysis
        key_neurons = {
            19: {"role": "ARITHMETIC-", "layers": [20, 24, 28]},
            1698: {"role": "ARITHMETIC+", "layers": [20, 24, 28]},
            2309: {"role": "ARITHMETIC-", "layers": [20, 24, 28]},
            468: {"role": "ARITHMETIC-", "layers": [20, 24]},
            1305: {"role": "ARITHMETIC-", "layers": [20, 24]},
        }

        print("\n" + "=" * 70)
        print("NEURON ABLATION STUDY")
        print("=" * 70)

        # Baseline accuracy
        print("\n1. Computing baseline accuracy...")
        baseline_acc, baseline_results = self.test_multiplication_accuracy()
        print(f"   Baseline accuracy: {baseline_acc:.1%}")

        print("\n   Sample results:")
        for r in baseline_results[:5]:
            status = "✓" if r["correct"] else "✗"
            print(f"   {status} {r['prompt']} -> {r['output']} (expected: {r['expected']})")

        # Test each neuron individually
        print("\n2. Single neuron ablation (at one layer)...")
        print(f"\n{'Neuron':<10} {'Layer':<8} {'Role':<15} {'Accuracy':<12} {'Drop':<10}")
        print("-" * 60)

        single_results = []

        for neuron, info in key_neurons.items():
            for layer in info["layers"][:1]:  # Test first layer only for speed
                acc, results = self.test_multiplication_accuracy(
                    ablate_neurons=[neuron],
                    ablate_layer=layer,
                )
                drop = baseline_acc - acc

                print(f"{neuron:<10} L{layer:<7} {info['role']:<15} {acc:>10.1%} {drop:>+9.1%}")

                single_results.append(
                    {
                        "neuron": neuron,
                        "layer": layer,
                        "role": info["role"],
                        "accuracy": acc,
                        "drop": drop,
                    }
                )

        # Test top neurons together at single layer
        print("\n3. Combined neurons at single layer...")

        top_neurons = [19, 1698, 2309]
        combined_results = []

        for layer in [20, 24, 28]:
            acc, results = self.test_multiplication_accuracy(
                ablate_neurons=top_neurons,
                ablate_layer=layer,
            )
            drop = baseline_acc - acc

            print(f"   Neurons {top_neurons} @ L{layer}: {acc:.1%} (drop: {drop:+.1%})")

            combined_results.append(
                {
                    "neurons": top_neurons,
                    "layer": layer,
                    "accuracy": acc,
                    "drop": drop,
                }
            )

        # NEW: Test across ALL layers simultaneously
        print("\n4. Ablating neurons across MULTIPLE layers...")

        all_layers = [20, 24, 28]
        all_5_neurons = [19, 1698, 2309, 468, 1305]

        # Top 3 neurons across all 3 layers
        acc, results = self.test_multiplication_accuracy(
            ablate_neurons=top_neurons,
            ablate_layer=all_layers,  # All layers!
        )
        drop = baseline_acc - acc
        print(f"   Top 3 neurons @ L20,L24,L28: {acc:.1%} (drop: {drop:+.1%})")

        # All 5 neurons across all 3 layers
        acc, results = self.test_multiplication_accuracy(
            ablate_neurons=all_5_neurons,
            ablate_layer=all_layers,
        )
        drop = baseline_acc - acc
        print(f"   All 5 neurons @ L20,L24,L28: {acc:.1%} (drop: {drop:+.1%})")

        # NEW: Test ablating MANY neurons (10, 50, 100, 500, 1000)
        print("\n5. Mass ablation tests (to find threshold)...")

        intermediate_size = self.config.intermediate_size  # ~10K neurons

        # Test progressively more neurons
        for num_neurons in [10, 50, 100, 200, 500, 1000, 2000]:
            if num_neurons > intermediate_size:
                break

            # Use first N neuron indices (simulating "top" neurons)
            # Note: These aren't necessarily the most important, but tests threshold
            neurons = list(range(num_neurons))
            acc, results = self.test_multiplication_accuracy(
                ablate_neurons=neurons,
                ablate_layer=all_layers,
            )
            drop = baseline_acc - acc
            pct_ablated = num_neurons / intermediate_size * 100
            print(
                f"   First {num_neurons} neurons ({pct_ablated:.1f}%) @ L20,L24,L28: {acc:.1%} (drop: {drop:+.1%})"
            )

        # NEW: Test ablating RANDOM neurons as control
        print("\n6. Control: Ablating RANDOM neurons...")
        import random

        random.seed(42)

        intermediate_size = self.config.intermediate_size  # ~10K neurons
        random_neurons = random.sample(range(intermediate_size), 5)

        acc, results = self.test_multiplication_accuracy(
            ablate_neurons=random_neurons,
            ablate_layer=all_layers,
        )
        drop = baseline_acc - acc
        print(f"   5 random neurons @ L20,L24,L28: {acc:.1%} (drop: {drop:+.1%})")

        random_neurons_20 = random.sample(range(intermediate_size), 20)
        acc, results = self.test_multiplication_accuracy(
            ablate_neurons=random_neurons_20,
            ablate_layer=all_layers,
        )
        drop = baseline_acc - acc
        print(f"   20 random neurons @ L20,L24,L28: {acc:.1%} (drop: {drop:+.1%})")

        # Detailed comparison for a specific case
        print("\n7. Detailed comparison: 7 * 8 = 56")

        prompt = "7 * 8 = "
        baseline_out = self.generate(prompt, max_tokens=5)
        print(f"   Baseline: {prompt} -> {baseline_out}")

        for neuron in [19, 1698, 2309]:
            ablated_out = self.generate_with_neuron_ablation(
                prompt, [neuron], all_layers, max_tokens=5
            )
            print(f"   Ablate N{neuron} @ all layers: {prompt} -> {ablated_out}")

        ablated_all = self.generate_with_neuron_ablation(
            prompt, all_5_neurons, all_layers, max_tokens=5
        )
        print(f"   Ablate all 5 @ all layers: {prompt} -> {ablated_all}")

        # Summary
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        # Find most impactful neurons
        sorted_results = sorted(single_results, key=lambda x: -abs(x["drop"]))

        print("\nSingle neuron impact (by accuracy drop):")
        for r in sorted_results[:5]:
            impact = (
                "CAUSAL" if abs(r["drop"]) > 0.05 else "MINOR" if abs(r["drop"]) > 0 else "NONE"
            )
            print(f"  Neuron {r['neuron']} @ L{r['layer']}: {r['drop']:+.1%} drop -> {impact}")

        print("\nInterpretation:")
        print(
            "  - If single neurons show NONE: redundancy exists, neurons not individually critical"
        )
        print("  - If combined ablation shows drop: distributed representation")
        print("  - If random ablation shows similar drop: neurons not special")

        # Save results
        results = {
            "model": self.model_id,
            "baseline_accuracy": baseline_acc,
            "single_ablations": single_results,
            "combined_ablations": combined_results,
            "top_neurons": top_neurons,
        }

        output_path = Path("gemma_discovery_cache/neuron_ablation.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    study = NeuronAblationStudy()
    study.run_ablation_study()


if __name__ == "__main__":
    main()
