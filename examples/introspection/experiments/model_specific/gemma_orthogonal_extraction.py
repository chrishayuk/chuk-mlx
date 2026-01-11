#!/usr/bin/env python3
"""
Gemma Orthogonal Extraction Experiment.

This is the key experiment to understand Gemma's arithmetic architecture.

GPT-OSS-20B had clear separable structures:
- A-encoders: Directions for first operand
- B-encoders: Directions for second operand
- Orthogonal subspaces for operand roles

Gemma's ablation results suggest either:
(a) No separable circuits — truly distributed/holistic encoding
(b) Circuits exist but are more redundant/overlapping

This experiment extracts:
1. Operand directions (A_d for first operand, B_d for second operand)
2. Tests orthogonality between A and B subspaces
3. Extracts product directions
4. Tests causal steering with operand directions

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_orthogonal_extraction.py
"""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class OrthogonalExtractionAnalyzer:
    """Extract and analyze operand/product directions in Gemma."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

        # Digits we'll analyze
        self.digits = list(range(2, 10))  # 2-9

        # Target layers (where probes showed activity)
        self.target_layers = [8, 16, 20, 21, 24]

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
        head = getattr(self.model, "lm_head", None)
        embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, norm, head, embed_scale

    def collect_layer_activations(self, prompt: str) -> dict:
        """Collect hidden states at target layers."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        activations = {}

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

            if i in self.target_layers:
                activations[i] = np.array(h[0, -1, :].tolist())

        return activations

    def generate_with_steering(self, prompt: str, steering_vector: np.ndarray,
                               layer_idx: int, strength: float) -> str:
        """Generate with a steering vector added at specific layer."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        steering = mx.array(steering_vector).reshape(1, 1, -1) * strength

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

            # Apply steering after target layer
            if i == layer_idx:
                h = h + steering.astype(h.dtype)

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
    # PHASE 1: Extract Operand Directions
    # =========================================================================
    def extract_operand_directions(self) -> dict:
        """Extract A_d (first operand) and B_d (second operand) directions."""
        print("\n" + "=" * 70)
        print("PHASE 1: EXTRACTING OPERAND DIRECTIONS")
        print("=" * 70)

        # A_d: mean activation when d is first operand
        # B_d: mean activation when d is second operand

        a_activations = {d: defaultdict(list) for d in self.digits}
        b_activations = {d: defaultdict(list) for d in self.digits}

        print("\nCollecting activations for all a*b combinations...")

        for a in self.digits:
            for b in self.digits:
                prompt = f"{a} * {b} = "
                acts = self.collect_layer_activations(prompt)

                for layer, act in acts.items():
                    a_activations[a][layer].append(act)
                    b_activations[b][layer].append(act)

        # Compute mean directions
        a_directions = {}
        b_directions = {}

        for d in self.digits:
            a_directions[d] = {}
            b_directions[d] = {}

            for layer in self.target_layers:
                a_directions[d][layer] = np.mean(a_activations[d][layer], axis=0)
                b_directions[d][layer] = np.mean(b_activations[d][layer], axis=0)

        print(f"\nExtracted directions for digits {self.digits}")
        print(f"At layers: {self.target_layers}")

        return {'a_directions': a_directions, 'b_directions': b_directions}

    # =========================================================================
    # PHASE 2: Orthogonality Tests
    # =========================================================================
    def test_orthogonality(self, directions: dict) -> dict:
        """Test orthogonality between A and B subspaces."""
        print("\n" + "=" * 70)
        print("PHASE 2: ORTHOGONALITY TESTS")
        print("=" * 70)

        a_dirs = directions['a_directions']
        b_dirs = directions['b_directions']

        results = {}

        for layer in self.target_layers:
            print(f"\n--- Layer {layer} ---")

            # Get direction vectors for this layer
            a_vecs = np.array([a_dirs[d][layer] for d in self.digits])
            b_vecs = np.array([b_dirs[d][layer] for d in self.digits])

            # Normalize
            a_norms = np.linalg.norm(a_vecs, axis=1, keepdims=True)
            b_norms = np.linalg.norm(b_vecs, axis=1, keepdims=True)
            a_vecs_norm = a_vecs / (a_norms + 1e-10)
            b_vecs_norm = b_vecs / (b_norms + 1e-10)

            # A vs A similarity (should be LOW if distinct operand encodings)
            a_vs_a = a_vecs_norm @ a_vecs_norm.T
            a_vs_a_offdiag = a_vs_a[np.triu_indices(len(self.digits), k=1)]
            a_vs_a_mean = np.mean(a_vs_a_offdiag)

            # B vs B similarity
            b_vs_b = b_vecs_norm @ b_vecs_norm.T
            b_vs_b_offdiag = b_vs_b[np.triu_indices(len(self.digits), k=1)]
            b_vs_b_mean = np.mean(b_vs_b_offdiag)

            # A vs B similarity (should be ~0 if orthogonal subspaces)
            a_vs_b = a_vecs_norm @ b_vecs_norm.T
            a_vs_b_mean = np.mean(a_vs_b)

            # Same digit, different role: A_i vs B_i
            same_digit = np.array([np.dot(a_vecs_norm[i], b_vecs_norm[i])
                                   for i in range(len(self.digits))])
            same_digit_mean = np.mean(same_digit)

            print(f"\n{'Comparison':<30} {'Mean Similarity':<15} {'Interpretation'}")
            print("-" * 65)

            # Interpret results
            def interpret(val, low_thresh=0.5, high_thresh=0.8):
                if val < low_thresh:
                    return "DISTINCT"
                elif val < high_thresh:
                    return "Moderate overlap"
                else:
                    return "HIGH OVERLAP"

            print(f"{'A_i vs A_j (diff digits)':<30} {a_vs_a_mean:>12.3f}   {interpret(a_vs_a_mean)}")
            print(f"{'B_i vs B_j (diff digits)':<30} {b_vs_b_mean:>12.3f}   {interpret(b_vs_b_mean)}")
            print(f"{'A_i vs B_j (cross-role)':<30} {a_vs_b_mean:>12.3f}   {'ORTHOGONAL' if abs(a_vs_b_mean) < 0.3 else 'NOT orthogonal'}")
            print(f"{'A_i vs B_i (same digit)':<30} {same_digit_mean:>12.3f}   {'Role > Digit' if same_digit_mean < 0.5 else 'Digit > Role'}")

            results[layer] = {
                'a_vs_a_mean': float(a_vs_a_mean),
                'b_vs_b_mean': float(b_vs_b_mean),
                'a_vs_b_mean': float(a_vs_b_mean),
                'same_digit_mean': float(same_digit_mean),
                'a_vs_a_matrix': a_vs_a.tolist(),
                'b_vs_b_matrix': b_vs_b.tolist(),
            }

        return results

    # =========================================================================
    # PHASE 3: Product Directions
    # =========================================================================
    def extract_product_directions(self) -> dict:
        """Extract directions for each product value."""
        print("\n" + "=" * 70)
        print("PHASE 3: PRODUCT DIRECTIONS")
        print("=" * 70)

        # Group prompts by product
        product_prompts = defaultdict(list)
        for a in self.digits:
            for b in self.digits:
                product = a * b
                product_prompts[product].append((a, b))

        print(f"\nProducts with multiple factorizations:")
        for p in sorted(product_prompts.keys()):
            if len(product_prompts[p]) > 1:
                print(f"  {p}: {product_prompts[p]}")

        # Collect activations by product
        product_activations = {p: defaultdict(list) for p in product_prompts.keys()}

        for product, pairs in product_prompts.items():
            for a, b in pairs:
                prompt = f"{a} * {b} = "
                acts = self.collect_layer_activations(prompt)
                for layer, act in acts.items():
                    product_activations[product][layer].append(act)

        # Compute mean product directions
        product_directions = {}
        for product in product_prompts.keys():
            product_directions[product] = {}
            for layer in self.target_layers:
                product_directions[product][layer] = np.mean(
                    product_activations[product][layer], axis=0
                )

        # Test: Same-product pairs vs same-operand pairs
        print("\n--- Same-Product Clustering Test ---")

        for layer in self.target_layers:
            # Same-product similarity (e.g., 3*4 vs 2*6 for product=12)
            same_product_sims = []
            for product, pairs in product_prompts.items():
                if len(pairs) > 1:
                    # Compare all pairs with same product
                    for i, (a1, b1) in enumerate(pairs):
                        for a2, b2 in pairs[i+1:]:
                            prompt1 = f"{a1} * {b1} = "
                            prompt2 = f"{a2} * {b2} = "
                            act1 = self.collect_layer_activations(prompt1)[layer]
                            act2 = self.collect_layer_activations(prompt2)[layer]
                            sim = np.dot(act1, act2) / (np.linalg.norm(act1) * np.linalg.norm(act2) + 1e-10)
                            same_product_sims.append(sim)

            # Same-operand similarity (e.g., 3*4 vs 3*5 - same first operand)
            same_operand_sims = []
            for a in self.digits[:4]:  # Subset for speed
                for b1 in self.digits[:4]:
                    for b2 in self.digits[:4]:
                        if b1 != b2:
                            prompt1 = f"{a} * {b1} = "
                            prompt2 = f"{a} * {b2} = "
                            act1 = self.collect_layer_activations(prompt1)[layer]
                            act2 = self.collect_layer_activations(prompt2)[layer]
                            sim = np.dot(act1, act2) / (np.linalg.norm(act1) * np.linalg.norm(act2) + 1e-10)
                            same_operand_sims.append(sim)

            same_product_mean = np.mean(same_product_sims) if same_product_sims else 0
            same_operand_mean = np.mean(same_operand_sims) if same_operand_sims else 0

            print(f"\nLayer {layer}:")
            print(f"  Same-product similarity: {same_product_mean:.3f}")
            print(f"  Same-operand similarity: {same_operand_mean:.3f}")

            if same_product_mean > same_operand_mean + 0.05:
                print(f"  → PRODUCT-INDEXED lookup (1D)")
            elif same_operand_mean > same_product_mean + 0.05:
                print(f"  → OPERAND-INDEXED lookup (2D)")
            else:
                print(f"  → Mixed/distributed encoding")

        return {'product_directions': product_directions, 'products': list(product_prompts.keys())}

    # =========================================================================
    # PHASE 4: Causal Steering with Operand Directions
    # =========================================================================
    def test_operand_steering(self, directions: dict) -> dict:
        """Test if operand directions are causally effective."""
        print("\n" + "=" * 70)
        print("PHASE 4: CAUSAL STEERING WITH OPERAND DIRECTIONS")
        print("=" * 70)

        a_dirs = directions['a_directions']
        b_dirs = directions['b_directions']

        results = {}

        # Test prompt: 5 * 6 = 30
        base_prompt = "5 * 6 = "
        base_answer = "30"

        print(f"\nBase prompt: '{base_prompt}' (expected: {base_answer})")

        # Get baseline
        baseline_output = self.generate_with_steering(base_prompt, np.zeros(self.hidden_size), 20, 0)
        print(f"Baseline output: {baseline_output}")

        # Test steering with different operand directions
        test_cases = [
            # Steer A direction: Change first operand
            ('A_7 (first operand → 7)', 7, 'a', [35, 42]),  # 7*6=42, but might get 7*5=35
            ('A_3 (first operand → 3)', 3, 'a', [18]),      # 3*6=18
            ('A_9 (first operand → 9)', 9, 'a', [54]),      # 9*6=54

            # Steer B direction: Change second operand
            ('B_8 (second operand → 8)', 8, 'b', [40]),     # 5*8=40
            ('B_3 (second operand → 3)', 3, 'b', [15]),     # 5*3=15
            ('B_9 (second operand → 9)', 9, 'b', [45]),     # 5*9=45
        ]

        for layer in [20, 24]:
            print(f"\n--- Steering at Layer {layer} ---")
            print(f"{'Direction':<30} {'Strength':<10} {'Output':<10} {'Expected':<15} {'Result'}")
            print("-" * 80)

            layer_results = []

            for desc, digit, role, expected_products in test_cases:
                if role == 'a':
                    direction = a_dirs[digit][layer] - a_dirs[5][layer]  # Difference from baseline
                else:
                    direction = b_dirs[digit][layer] - b_dirs[6][layer]

                # Normalize direction
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                for strength in [50, 100, 200]:
                    output = self.generate_with_steering(base_prompt, direction, layer, strength)

                    # Check if output matches expected
                    success = any(str(p)[0] in output for p in expected_products)
                    result = "✓ STEERED" if success else f"(got {output})"

                    print(f"{desc:<30} {strength:<10} {output:<10} {expected_products} {result}")

                    layer_results.append({
                        'direction': desc,
                        'strength': strength,
                        'output': output,
                        'expected': expected_products,
                        'success': success,
                    })

            results[layer] = layer_results

        return results

    # =========================================================================
    # SUMMARY
    # =========================================================================
    def run_all_phases(self) -> dict:
        """Run all phases of orthogonal extraction."""
        results = {}

        # Phase 1: Extract directions
        directions = self.extract_operand_directions()
        results['directions'] = {
            'a_directions': {str(d): {str(l): dirs[l].tolist()
                                       for l in self.target_layers}
                             for d, dirs in directions['a_directions'].items()},
            'b_directions': {str(d): {str(l): dirs[l].tolist()
                                       for l in self.target_layers}
                             for d, dirs in directions['b_directions'].items()},
        }

        # Phase 2: Orthogonality
        orthogonality = self.test_orthogonality(directions)
        results['orthogonality'] = orthogonality

        # Phase 3: Product directions
        products = self.extract_product_directions()
        results['products'] = {
            'directions': {str(p): {str(l): dirs[l].tolist()
                                    for l in self.target_layers}
                           for p, dirs in products['product_directions'].items()},
        }

        # Phase 4: Steering
        steering = self.test_operand_steering(directions)
        results['steering'] = steering

        # Summary
        print("\n" + "=" * 70)
        print("ORTHOGONAL EXTRACTION SUMMARY")
        print("=" * 70)

        print("""
KEY FINDINGS:

1. OPERAND DIRECTION SEPARABILITY:
   - A_i vs A_j: Are different first-operand directions distinct?
   - B_i vs B_j: Are different second-operand directions distinct?
   - Check orthogonality results above

2. A vs B ORTHOGONALITY:
   - GPT-OSS: A and B subspaces were orthogonal
   - Gemma: Check if A_i vs B_j similarity is near 0

3. LOOKUP STRUCTURE:
   - Product-indexed (1D): Same-product pairs cluster tighter
   - Operand-indexed (2D): Same-operand pairs cluster tighter

4. CAUSAL STEERING:
   - If operand steering works: Separable circuits exist
   - If operand steering fails: Holistic encoding
""")

        # Interpret overall pattern
        layer = 20
        orth = orthogonality[layer]

        print("\n" + "-" * 70)
        print("ARCHITECTURAL INTERPRETATION:")
        print("-" * 70)

        if orth['a_vs_a_mean'] > 0.8 and orth['b_vs_b_mean'] > 0.8:
            print("→ HIGH OVERLAP in operand directions")
            print("→ Gemma uses HOLISTIC encoding (not GPT-OSS style)")
        elif orth['a_vs_b_mean'] < 0.3:
            print("→ A and B subspaces are ORTHOGONAL")
            print("→ Gemma has separable operand encoders (like GPT-OSS)")
        else:
            print("→ MIXED structure - partially separable")

        if orth['same_digit_mean'] > 0.7:
            print("→ DIGIT dominates over ROLE")
            print("→ 7-as-first looks like 7-as-second")
        else:
            print("→ ROLE dominates over DIGIT")
            print("→ 7-as-first is distinct from 7-as-second (like GPT-OSS)")

        # Save results
        output_path = Path("gemma_discovery_cache/orthogonal_extraction.json")
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

        # Save only summary (full directions are too large)
        summary = {
            'orthogonality': convert_numpy(orthogonality),
            'steering': convert_numpy(steering),
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    analyzer = OrthogonalExtractionAnalyzer()
    analyzer.load_model()
    analyzer.run_all_phases()


if __name__ == "__main__":
    main()
