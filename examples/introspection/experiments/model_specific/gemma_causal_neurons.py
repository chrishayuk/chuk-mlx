#!/usr/bin/env python3
"""
Gemma Causal Neuron Intervention.

Target the specific neurons identified by probes (19, 1698, 2309) and test
if they are causally responsible for arithmetic recognition.

Previous finding: Ablating 20% of neurons = 0% accuracy drop
But those were random/top-weighted neurons. These are the CLASSIFICATION neurons.

Hypothesis: These neurons might affect CLASSIFICATION accuracy, not generation accuracy.

Tests:
1. Ablate neurons 19, 1698, 2309 at L20, L24, L28
2. Measure: Does arithmetic vs language CLASSIFICATION drop?
3. Measure: Does multiplication GENERATION accuracy drop?
4. Compare to GPT-OSS-20B compute neurons

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_causal_neurons.py
"""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class CausalNeuronAnalyzer:
    """Test causal role of identified neurons."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

        # Target neurons from probe analysis
        self.target_neurons = {
            "arithmetic_negative": [19, 2309, 468, 1305],  # Suppress arithmetic
            "arithmetic_positive": [1698],  # Enhance arithmetic
            "all_identified": [19, 1698, 2309, 468, 1305],
        }

        # Layers where these neurons were identified
        self.target_layers = [20, 24, 28]

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
        print(f"  Target neurons: {self.target_neurons['all_identified']}")
        print(f"  Target layers: {self.target_layers}")

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

    def collect_activations_with_ablation(
        self, prompt: str, ablate_neurons: list = None, ablate_layers: list = None
    ) -> dict:
        """Collect activations, optionally ablating specific neurons."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        activations = {}

        for i, layer in enumerate(layers):
            # Get attention output
            attn = layer.self_attn

            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            else:
                h_normed = h

            attn_out = attn(h_normed, mask=mask)

            # Handle tuple output from attention
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]

            if hasattr(layer, "post_attention_layernorm"):
                attn_out = layer.post_attention_layernorm(attn_out)

            h = h + attn_out

            # MLP with potential ablation
            if hasattr(layer, "pre_feedforward_layernorm"):
                mlp_input = layer.pre_feedforward_layernorm(h)
            else:
                mlp_input = h

            mlp = layer.mlp

            # Get intermediate activations
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                gate = mlp.gate_proj(mlp_input)
                up = mlp.up_proj(mlp_input)

                if hasattr(mlp, "act_fn"):
                    intermediate = mlp.act_fn(gate) * up
                else:
                    intermediate = mx.sigmoid(gate) * gate * up  # SiLU approximation

                # ABLATION: Zero out specific neurons at target layers
                if ablate_neurons and ablate_layers and i in ablate_layers:
                    intermediate_list = intermediate.tolist()
                    for batch in range(len(intermediate_list)):
                        for seq in range(len(intermediate_list[batch])):
                            for neuron in ablate_neurons:
                                if neuron < len(intermediate_list[batch][seq]):
                                    intermediate_list[batch][seq][neuron] = 0.0
                    intermediate = mx.array(intermediate_list)

                mlp_out = mlp.down_proj(intermediate)
            else:
                mlp_out = mlp(mlp_input)

            if hasattr(layer, "post_feedforward_layernorm"):
                mlp_out = layer.post_feedforward_layernorm(mlp_out)

            h = h + mlp_out

            # Store activation at target layers
            if i in self.target_layers:
                activations[i] = mx.array(h)

        return activations, h

    def generate_with_ablation(
        self, prompt: str, ablate_neurons: list = None, ablate_layers: list = None
    ) -> str:
        """Generate with specific neurons ablated."""
        layers, embed, norm, head, embed_scale = self._get_components()

        _, h = self.collect_activations_with_ablation(prompt, ablate_neurons, ablate_layers)

        # Continue through remaining layers
        # (activations already processed through all layers)

        if norm is not None:
            h = norm(h)

        if head is not None:
            logits = head(h)
            if hasattr(logits, "logits"):
                logits = logits.logits
        else:
            logits = h @ embed.weight.T

        next_token = mx.argmax(logits[0, -1, :])
        return self.tokenizer.decode([int(next_token)])

    # =========================================================================
    # EXPERIMENT 1: Classification Accuracy with Ablation
    # =========================================================================
    def test_classification_with_ablation(self) -> dict:
        """Test if ablating target neurons affects classification accuracy."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: CLASSIFICATION ACCURACY WITH NEURON ABLATION")
        print("=" * 70)
        print("Do neurons 19, 1698, 2309 causally affect arithmetic classification?")

        # Create dataset
        arithmetic_prompts = []
        language_prompts = []

        for a in range(2, 10):
            for b in range(2, 10):
                arithmetic_prompts.append(f"{a} * {b} = ")

        language_templates = [
            "The cat sat on the",
            "I went to the store to",
            "The weather today is very",
            "She picked up the book and",
            "The dog barked at the",
        ]
        for template in language_templates:
            for _ in range(13):
                language_prompts.append(template)

        np.random.seed(42)
        n_samples = 50
        arithmetic_prompts = list(np.random.choice(arithmetic_prompts, n_samples, replace=False))
        language_prompts = list(np.random.choice(language_prompts, n_samples, replace=False))

        all_prompts = arithmetic_prompts + language_prompts
        labels = [1] * n_samples + [0] * n_samples

        combined = list(zip(all_prompts, labels))
        np.random.shuffle(combined)
        all_prompts, labels = zip(*combined)
        all_prompts = list(all_prompts)
        labels = list(labels)

        print(f"\nDataset: {n_samples} arithmetic + {n_samples} language")

        results = {}

        # Test different ablation conditions
        conditions = [
            ("baseline", None, None),
            ("ablate_19", [19], self.target_layers),
            ("ablate_1698", [1698], self.target_layers),
            ("ablate_2309", [2309], self.target_layers),
            ("ablate_all_negative", self.target_neurons["arithmetic_negative"], self.target_layers),
            ("ablate_all_identified", self.target_neurons["all_identified"], self.target_layers),
            (
                "ablate_random_5",
                list(np.random.randint(0, self.hidden_size, 5)),
                self.target_layers,
            ),
        ]

        for condition_name, neurons, layers in conditions:
            print(f"\nTesting condition: {condition_name}")
            if neurons:
                print(f"  Ablating neurons: {neurons} at layers {layers}")

            # Collect activations
            layer_activations = defaultdict(list)

            for prompt in all_prompts:
                acts, _ = self.collect_activations_with_ablation(prompt, neurons, layers)
                for layer, h in acts.items():
                    layer_activations[layer].append(np.array(h[0, -1, :].tolist()))

            # Train classifier at each target layer
            condition_results = {}

            for layer in self.target_layers:
                X = np.array(layer_activations[layer])
                y = np.array(labels)

                n_test = max(1, len(X) // 5)
                X_train, X_test = X[:-n_test], X[-n_test:]
                y_train, y_test = y[:-n_test], y[-n_test:]

                probe = LogisticRegression(max_iter=2000)
                probe.fit(X_train, y_train)
                accuracy = probe.score(X_test, y_test)

                condition_results[layer] = accuracy

            results[condition_name] = condition_results

        # Print results
        print("\n" + "-" * 70)
        print("CLASSIFICATION ACCURACY BY CONDITION:")
        print("-" * 70)

        header = f"{'Condition':<25}" + "".join([f"L{l:<8}" for l in self.target_layers])
        print(header)
        print("-" * 70)

        baseline = results["baseline"]
        for condition, accs in results.items():
            row = f"{condition:<25}"
            for layer in self.target_layers:
                acc = accs[layer]
                diff = acc - baseline[layer]
                if condition == "baseline":
                    row += f"{acc:>7.1%} "
                else:
                    row += f"{acc:>7.1%} ({diff:+.1%})"
            print(row)

        return results

    # =========================================================================
    # EXPERIMENT 2: Generation Accuracy with Ablation
    # =========================================================================
    def test_generation_with_ablation(self) -> dict:
        """Test if ablating target neurons affects multiplication accuracy."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: GENERATION ACCURACY WITH NEURON ABLATION")
        print("=" * 70)
        print("Do neurons 19, 1698, 2309 causally affect multiplication output?")

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

        conditions = [
            ("baseline", None, None),
            ("ablate_19", [19], self.target_layers),
            ("ablate_1698", [1698], self.target_layers),
            ("ablate_2309", [2309], self.target_layers),
            ("ablate_all_identified", self.target_neurons["all_identified"], self.target_layers),
            (
                "ablate_random_5",
                list(np.random.randint(0, self.hidden_size, 5)),
                self.target_layers,
            ),
        ]

        results = {}

        for condition_name, neurons, layers in conditions:
            correct = 0

            for a, b, expected in test_cases:
                prompt = f"{a} * {b} = "
                output = self.generate_with_ablation(prompt, neurons, layers)

                # Check if first digit matches
                expected_first = str(expected)[0]
                if expected_first in output:
                    correct += 1

            accuracy = correct / len(test_cases)
            results[condition_name] = accuracy

        print(f"\n{'Condition':<30} {'Accuracy':<12} {'Change'}")
        print("-" * 55)

        baseline_acc = results["baseline"]
        for condition, acc in results.items():
            diff = acc - baseline_acc
            if condition == "baseline":
                print(f"{condition:<30} {acc:>10.1%}")
            else:
                print(f"{condition:<30} {acc:>10.1%} {diff:>+10.1%}")

        return results

    # =========================================================================
    # EXPERIMENT 3: Neuron Activation Patterns
    # =========================================================================
    def analyze_neuron_patterns(self) -> dict:
        """Analyze how target neurons activate for arithmetic vs language."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: NEURON ACTIVATION PATTERNS")
        print("=" * 70)
        print("How do neurons 19, 1698, 2309 activate for arithmetic vs language?")

        # Sample prompts
        arithmetic_prompts = ["7 * 8 = ", "3 * 4 = ", "9 * 2 = ", "5 * 6 = ", "8 * 3 = "]
        language_prompts = [
            "The cat sat on the",
            "I went to the",
            "The weather is",
            "She picked up",
            "The dog barked",
        ]

        layers, embed, norm, head, embed_scale = self._get_components()

        print("\nTarget neurons:", self.target_neurons["all_identified"])

        for layer_idx in self.target_layers:
            print(f"\n--- Layer {layer_idx} ---")

            arith_activations = []
            lang_activations = []

            # Collect activations for arithmetic
            for prompt in arithmetic_prompts:
                input_ids = self.tokenizer.encode(prompt)
                input_ids = mx.array(input_ids)[None, :]
                h = embed(input_ids) * embed_scale
                mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
                mask = mask.astype(h.dtype)

                for i, layer in enumerate(layers):
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

                    if i == layer_idx:
                        arith_activations.append(np.array(h[0, -1, :].tolist()))
                        break

            # Collect for language
            for prompt in language_prompts:
                input_ids = self.tokenizer.encode(prompt)
                input_ids = mx.array(input_ids)[None, :]
                h = embed(input_ids) * embed_scale
                mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
                mask = mask.astype(h.dtype)

                for i, layer in enumerate(layers):
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

                    if i == layer_idx:
                        lang_activations.append(np.array(h[0, -1, :].tolist()))
                        break

            arith_activations = np.array(arith_activations)
            lang_activations = np.array(lang_activations)

            print(f"\n{'Neuron':<10} {'Arith Mean':<12} {'Lang Mean':<12} {'Diff':<12} {'Role'}")
            print("-" * 60)

            for neuron in self.target_neurons["all_identified"]:
                arith_mean = np.mean(arith_activations[:, neuron])
                lang_mean = np.mean(lang_activations[:, neuron])
                diff = arith_mean - lang_mean

                if diff > 0.5:
                    role = "ARITHMETIC+"
                elif diff < -0.5:
                    role = "ARITHMETIC-"
                else:
                    role = "Neutral"

                print(
                    f"{neuron:<10} {arith_mean:>10.3f} {lang_mean:>10.3f} {diff:>+10.3f}   {role}"
                )

        return {}

    # =========================================================================
    # EXPERIMENT 4: Compare to GPT-OSS Architecture
    # =========================================================================
    def compare_to_gpt_oss(self) -> dict:
        """Document comparison with GPT-OSS-20B findings."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: COMPARISON TO GPT-OSS-20B")
        print("=" * 70)

        comparison = """
GPT-OSS-20B Compute Neurons (from prior research):
- Located in middle layers (~L12-L19)
- A-encoders: Respond to first operand
- B-encoders: Respond to second operand
- Product neurons: Respond to specific products

Gemma-3-4B Identified Neurons:
- Neuron 19: Active at L20, L24, L28 - ARITHMETIC NEGATIVE
- Neuron 1698: Active at L20, L24, L28 - ARITHMETIC POSITIVE
- Neuron 2309: Active at L20, L24, L28 - ARITHMETIC NEGATIVE

Architectural Comparison:
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-OSS-20B                                  │
│  L0-L3: Encoding                                                │
│  L4-L18: A/B encoders + retrieval                               │
│  L19: Arithmetic Hub (crystallization)                          │
│  L20-L23: Output (L22-23 dispensable)                           │
├─────────────────────────────────────────────────────────────────┤
│                    Gemma-3-4B                                   │
│  L0-L3: Encoding                                                │
│  L4-L16: Retrieval (answer encoded)                             │
│  L17-L22: Computation (L21 critical)                            │
│  L20,L24,L28: Classification neurons active                     │
│  L29-L33: Dispensable                                           │
└─────────────────────────────────────────────────────────────────┘

Key Differences:
1. GPT-OSS has distinct A/B encoder neurons
2. Gemma classification neurons are in LATER layers (L20+)
3. Both have ~15% dispensable late layers
4. Both use lookup tables, not algorithms

Hypothesis:
- GPT-OSS: Operand-specific neurons in middle layers
- Gemma: Task-classification neurons in late layers
- Different implementation, same 6-phase structure
"""
        print(comparison)

        return {"comparison": comparison}

    # =========================================================================
    # RUN ALL
    # =========================================================================
    def run_all_experiments(self) -> dict:
        """Run all causal neuron experiments."""
        results = {}

        results["classification"] = self.test_classification_with_ablation()
        results["generation"] = self.test_generation_with_ablation()
        results["patterns"] = self.analyze_neuron_patterns()
        results["gpt_oss_comparison"] = self.compare_to_gpt_oss()

        # Summary
        print("\n" + "=" * 70)
        print("CAUSAL NEURON ANALYSIS SUMMARY")
        print("=" * 70)

        print("""
KEY FINDINGS:

1. CLASSIFICATION NEURONS (19, 1698, 2309):
   - Located at L20, L24, L28
   - Identified by probe weights
   - Test above shows if they're CAUSALLY important

2. IF classification accuracy drops with ablation:
   → These ARE the arithmetic recognizer neurons
   → Gemma uses late-layer classification

3. IF classification accuracy stays same:
   → Classification is distributed (like generation)
   → No single neurons are critical

4. COMPARISON TO GPT-OSS:
   - GPT-OSS: Operand encoders in middle layers
   - Gemma: Classification neurons in late layers
   - Both: Lookup table structure, 6-phase architecture
""")

        # Save results
        output_path = Path("gemma_discovery_cache/causal_neurons.json")
        output_path.parent.mkdir(exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    analyzer = CausalNeuronAnalyzer()
    analyzer.load_model()
    analyzer.run_all_experiments()


if __name__ == "__main__":
    main()
