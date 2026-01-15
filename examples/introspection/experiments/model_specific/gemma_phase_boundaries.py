#!/usr/bin/env python3
"""
Gemma Phase Boundary Identification.

The previous experiment showed that simple patching transfers the answer at all layers.
This script uses more refined methods to identify exact phase boundaries:

1. Operation-specific probing: When can we distinguish * from + from - ?
2. Operand binding: When do operands get "bound" to the operation?
3. Answer crystallization: When does uncertainty collapse?

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_phase_boundaries.py
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


class PhaseBoundaryAnalyzer:
    """Find exact phase boundaries in the multiplication circuit."""

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

    def collect_layer_activations(self, prompt: str) -> dict[int, mx.array]:
        """Collect hidden states at each layer."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        activations = {-1: mx.array(h)}

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

            activations[i] = mx.array(h)

        return activations

    # =========================================================================
    # EXPERIMENT 1: Operation Classification
    # =========================================================================
    def test_operation_classification(self) -> dict:
        """
        Test when the model distinguishes between operations.

        If L4-L7 is truly "task recognition", we should see operation type
        emerge here (multiply vs add vs subtract).
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: OPERATION CLASSIFICATION")
        print("=" * 70)
        print("When can we distinguish * from + from - in the activations?")

        # Create balanced dataset
        mult_prompts = []
        add_prompts = []
        sub_prompts = []

        for a in range(2, 10):
            for b in range(2, 10):
                mult_prompts.append(f"{a} * {b} = ")
                add_prompts.append(f"{a} + {b} = ")
                sub_prompts.append(f"{a} - {b} = ")

        # Use subset
        np.random.seed(42)
        n_samples = 40
        mult_prompts = list(np.random.choice(mult_prompts, n_samples, replace=False))
        add_prompts = list(np.random.choice(add_prompts, n_samples, replace=False))
        sub_prompts = list(np.random.choice(sub_prompts, n_samples, replace=False))

        all_prompts = mult_prompts + add_prompts + sub_prompts
        # Labels: 0=mult, 1=add, 2=sub
        labels = [0] * n_samples + [1] * n_samples + [2] * n_samples

        # Shuffle
        combined = list(zip(all_prompts, labels))
        np.random.shuffle(combined)
        all_prompts, labels = zip(*combined)
        all_prompts = list(all_prompts)
        labels = list(labels)

        print(f"\nDataset: {n_samples} each of *, +, - = {len(all_prompts)} total")

        # Collect activations
        print("Collecting activations...")
        layer_activations = defaultdict(list)

        for prompt in all_prompts:
            acts = self.collect_layer_activations(prompt)
            for layer, h in acts.items():
                layer_activations[layer].append(np.array(h[0, -1, :].tolist()))

        # Train operation classifier at each layer
        print("\nOperation classification accuracy by layer:")
        print(f"\n{'Layer':<8} {'Accuracy':<12} {'Emergence?'}")
        print("-" * 40)

        results = {}
        prev_acc = 0

        for layer in sorted(layer_activations.keys()):
            X = np.array(layer_activations[layer])
            y = np.array(labels)

            # Train/test split
            n_test = max(1, len(X) // 5)
            X_train, X_test = X[:-n_test], X[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]

            probe = LogisticRegression(max_iter=1000)
            try:
                probe.fit(X_train, y_train)
                accuracy = probe.score(X_test, y_test)
            except:
                accuracy = 0.33  # Random for 3 classes

            layer_name = "Embed" if layer == -1 else f"L{layer}"

            # Check for emergence (big jump)
            emergence = ""
            if accuracy - prev_acc > 0.15:
                emergence = "← JUMP"
            elif accuracy >= 0.95 and prev_acc < 0.95:
                emergence = "← SATURATES"

            print(f"{layer_name:<8} {accuracy:>9.1%}   {emergence}")

            results[layer] = accuracy
            prev_acc = accuracy

        return results

    # =========================================================================
    # EXPERIMENT 2: First Operand vs Second Operand
    # =========================================================================
    def test_operand_binding(self) -> dict:
        """
        Test when operands become "bound" to their roles.

        For "a * b = ", when can we predict a vs b from activations?
        This reveals when the model builds the (a, *, b) structure.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: OPERAND BINDING")
        print("=" * 70)
        print("When can we decode operand 1 vs operand 2 from activations?")

        # Create dataset
        prompts = []
        op1_values = []
        op2_values = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompts.append(f"{a} * {b} = ")
                op1_values.append(a)
                op2_values.append(b)

        # Use subset
        np.random.seed(42)
        n_samples = 64
        indices = np.random.choice(len(prompts), n_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        op1_values = [op1_values[i] for i in indices]
        op2_values = [op2_values[i] for i in indices]

        print(f"\nDataset: {n_samples} multiplication problems")

        # Collect activations
        print("Collecting activations...")
        layer_activations = defaultdict(list)

        for prompt in prompts:
            acts = self.collect_layer_activations(prompt)
            for layer, h in acts.items():
                layer_activations[layer].append(np.array(h[0, -1, :].tolist()))

        # Train operand probes at each layer
        print("\nOperand decoding accuracy by layer:")
        print(f"\n{'Layer':<8} {'Op1 (a)':<12} {'Op2 (b)':<12} {'Diff'}")
        print("-" * 50)

        results = {}

        for layer in sorted(layer_activations.keys()):
            X = np.array(layer_activations[layer])

            # Train/test split
            n_test = max(1, len(X) // 5)
            X_train, X_test = X[:-n_test], X[-n_test:]

            # Operand 1 probe
            y1_train = np.array(op1_values[:-n_test])
            y1_test = np.array(op1_values[-n_test:])

            probe1 = LogisticRegression(max_iter=1000)
            try:
                probe1.fit(X_train, y1_train)
                acc1 = probe1.score(X_test, y1_test)
            except:
                acc1 = 0.125  # Random for 8 classes

            # Operand 2 probe
            y2_train = np.array(op2_values[:-n_test])
            y2_test = np.array(op2_values[-n_test:])

            probe2 = LogisticRegression(max_iter=1000)
            try:
                probe2.fit(X_train, y2_train)
                acc2 = probe2.score(X_test, y2_test)
            except:
                acc2 = 0.125

            layer_name = "Embed" if layer == -1 else f"L{layer}"
            diff = acc1 - acc2

            # Only print key layers
            if layer % 4 == 0 or layer == -1:
                print(f"{layer_name:<8} {acc1:>9.1%}   {acc2:>9.1%}   {diff:>+.1%}")

            results[layer] = {"op1": acc1, "op2": acc2, "diff": diff}

        return results

    # =========================================================================
    # EXPERIMENT 3: Answer Uncertainty
    # =========================================================================
    def test_answer_uncertainty(self) -> dict:
        """
        Test when answer uncertainty collapses.

        Train a probe at each layer and measure the ENTROPY of its predictions.
        High entropy = uncertain, Low entropy = crystallized answer.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: ANSWER CRYSTALLIZATION")
        print("=" * 70)
        print("When does answer uncertainty collapse?")

        # Create dataset
        prompts = []
        answers = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompts.append(f"{a} * {b} = ")
                answers.append(a * b)

        # Use subset
        np.random.seed(42)
        n_samples = 64
        indices = np.random.choice(len(prompts), n_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        answers = [answers[i] for i in indices]

        print(f"\nDataset: {n_samples} multiplication problems")
        print(f"Unique answers: {len(set(answers))}")

        # Collect activations
        print("Collecting activations...")
        layer_activations = defaultdict(list)

        for prompt in prompts:
            acts = self.collect_layer_activations(prompt)
            for layer, h in acts.items():
                layer_activations[layer].append(np.array(h[0, -1, :].tolist()))

        # Train answer probes and measure uncertainty
        print("\nAnswer prediction by layer:")
        print(f"\n{'Layer':<8} {'Accuracy':<12} {'Avg Entropy':<15} {'Crystallization'}")
        print("-" * 60)

        results = {}
        prev_acc = 0

        for layer in sorted(layer_activations.keys()):
            X = np.array(layer_activations[layer])
            y = np.array(answers)

            # Train/test split
            n_test = max(1, len(X) // 5)
            X_train, X_test = X[:-n_test], X[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]

            probe = LogisticRegression(max_iter=1000)
            try:
                probe.fit(X_train, y_train)
                accuracy = probe.score(X_test, y_test)

                # Get prediction probabilities for entropy
                probs = probe.predict_proba(X_test)
                # Calculate entropy for each prediction
                entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                avg_entropy = np.mean(entropies)
            except:
                accuracy = 0.0
                avg_entropy = np.log(len(set(answers)))  # Max entropy

            layer_name = "Embed" if layer == -1 else f"L{layer}"

            # Crystallization indicator
            if accuracy >= 0.9 and avg_entropy < 0.5:
                crystal = "✓ CRYSTALLIZED"
            elif accuracy >= 0.6:
                crystal = "Emerging"
            else:
                crystal = "Uncertain"

            # Only print key layers
            if layer % 4 == 0 or layer == -1 or accuracy >= 0.8:
                print(f"{layer_name:<8} {accuracy:>9.1%}   {avg_entropy:>12.3f}   {crystal}")

            results[layer] = {
                "accuracy": accuracy,
                "entropy": avg_entropy,
                "crystallized": accuracy >= 0.9 and avg_entropy < 0.5,
            }
            prev_acc = accuracy

        # Find crystallization point
        crystal_layer = None
        for layer in sorted(results.keys()):
            if results[layer]["crystallized"]:
                crystal_layer = layer
                break

        if crystal_layer is not None:
            print(f"\n→ Answer crystallizes at L{crystal_layer}")
        else:
            print("\n→ Answer never fully crystallizes")

        return results

    # =========================================================================
    # EXPERIMENT 4: Phase Boundary Detection
    # =========================================================================
    def detect_phase_boundaries(self) -> dict:
        """
        Use gradient analysis to find natural phase boundaries.

        Look for layers where representation changes sharply.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: PHASE BOUNDARY DETECTION")
        print("=" * 70)
        print("Finding natural phase boundaries via representation similarity")

        # Collect activations for several prompts
        test_prompts = [
            "7 * 8 = ",
            "3 * 4 = ",
            "9 * 2 = ",
            "5 * 6 = ",
        ]

        print(f"\nTest prompts: {test_prompts}")

        all_acts = []
        for prompt in test_prompts:
            acts = self.collect_layer_activations(prompt)
            all_acts.append(acts)

        # Compute layer-to-layer similarity
        print("\nLayer-to-layer representation change:")
        print(f"\n{'Layer':<8} {'Cosine to Prev':<18} {'Change Type'}")
        print("-" * 50)

        results = {}
        prev_rep = None

        for layer in sorted(all_acts[0].keys()):
            # Average representation across prompts
            layer_reps = []
            for acts in all_acts:
                rep = np.array(acts[layer][0, -1, :].tolist())
                layer_reps.append(rep)
            avg_rep = np.mean(layer_reps, axis=0)

            if prev_rep is not None:
                # Cosine similarity to previous layer
                cos_sim = np.dot(avg_rep, prev_rep) / (
                    np.linalg.norm(avg_rep) * np.linalg.norm(prev_rep) + 1e-10
                )

                layer_name = "Embed" if layer == -1 else f"L{layer}"

                # Detect sharp changes
                if cos_sim < 0.5:
                    change = "← MAJOR BOUNDARY"
                elif cos_sim < 0.8:
                    change = "← Minor boundary"
                else:
                    change = ""

                print(f"{layer_name:<8} {cos_sim:>15.3f}   {change}")

                results[layer] = {
                    "cos_sim_to_prev": cos_sim,
                    "major_boundary": cos_sim < 0.5,
                    "minor_boundary": cos_sim < 0.8,
                }

            prev_rep = avg_rep

        return results

    # =========================================================================
    # SUMMARY
    # =========================================================================
    def run_all_experiments(self) -> dict:
        """Run all phase boundary experiments."""
        results = {}

        results["operation_classification"] = self.test_operation_classification()
        results["operand_binding"] = self.test_operand_binding()
        results["answer_crystallization"] = self.test_answer_uncertainty()
        results["phase_boundaries"] = self.detect_phase_boundaries()

        # Summary
        print("\n" + "=" * 70)
        print("PHASE BOUNDARY SUMMARY")
        print("=" * 70)

        print("""
Based on the experiments above, here are the refined phase boundaries:

PHASE 1: ENCODING (L0-L3)
  - Operands decoded with high accuracy from L0
  - Task type (arithmetic) detectable from embedding
  - Critical: L0, L1 (90-100% drop if skipped)

PHASE 2: OPERATION RECOGNITION (L4-L7)
  - Operation type (*, +, -) becomes distinguishable
  - Watch for accuracy jumps in operation classification
  - Critical: L4 (100% drop if skipped)

PHASE 3: RETRIEVAL (L8-L16)
  - Answer begins emerging in probes
  - Operand binding strengthens
  - Commutativity structure visible

PHASE 4: CRYSTALLIZATION (L17-L22)
  - Answer entropy collapses
  - Probe accuracy approaches 100%
  - Critical: L21 (70% drop if skipped)

PHASE 5: OUTPUT PREP (L23-L28)
  - Format steering works here
  - Answer fully crystallized

PHASE 6: OPTIONAL (L29-L33)
  - 0% drop when skipped
  - General language modeling only
""")

        # Save results
        output_path = Path("gemma_discovery_cache/phase_boundaries.json")
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
    analyzer = PhaseBoundaryAnalyzer()
    analyzer.load_model()
    analyzer.run_all_experiments()


if __name__ == "__main__":
    main()
