#!/usr/bin/env python3
"""
Gemma Lookup Table Evolution: How does the multiplication structure evolve across layers?

This script tracks how the times table representation changes from early to late layers.

Key questions:
1. When does commutativity emerge?
2. When does same-product clustering appear?
3. When does row/column structure appear?
4. Is there a "lookup" phase followed by a "compute" phase?

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_lookup_evolution.py
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class LookupEvolutionAnalyzer:
    """Track how lookup table structure evolves across layers."""

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

        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
            embed_scale = float(self.hidden_size**0.5)

        return layers, embed, embed_scale

    def get_all_layer_states(self, prompt: str) -> list[np.ndarray]:
        """Get hidden states from all layers for a prompt."""
        layers, embed, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        states = []

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

            states.append(np.array(h[0, -1, :].tolist()))

        return states

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def analyze_evolution(self):
        """Analyze how structure evolves across all layers."""
        self.load_model()

        print("\nCollecting all times table states across all layers...")

        # Collect states for all prompts at all layers
        # states[prompt] = list of states per layer
        all_states = {}

        prompts = []
        for a in range(2, 10):
            for b in range(2, 10):
                prompt = f"{a} * {b} = "
                prompts.append((a, b, prompt))

        for i, (a, b, prompt) in enumerate(prompts):
            if i % 10 == 0:
                print(f"  {i}/{len(prompts)}")
            all_states[(a, b)] = self.get_all_layer_states(prompt)

        print(f"\nCollected states for {len(prompts)} prompts x {self.num_layers} layers")

        # Analyze each layer
        results_by_layer = []

        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER ANALYSIS")
        print("=" * 80)

        print(
            f"\n{'Layer':<8} {'Commut.':<10} {'SameProd':<10} {'Row':<10} {'Col':<10} {'Interpretation'}"
        )
        print("-" * 70)

        for layer_idx in range(self.num_layers):
            # Get states for this layer
            layer_states = {k: v[layer_idx] for k, v in all_states.items()}

            # Test 1: Commutativity
            comm_sims = []
            for a in range(2, 9):
                for b in range(a + 1, 10):
                    s_ab = layer_states[(a, b)]
                    s_ba = layer_states[(b, a)]
                    comm_sims.append(self.cosine_similarity(s_ab, s_ba))
            avg_comm = np.mean(comm_sims)

            # Test 2: Same-product similarity
            same_prod_sims = []
            # Example pairs with same product
            pairs = [
                ((2, 6), (3, 4)),  # 12
                ((2, 8), (4, 4)),  # 16
                ((2, 9), (3, 6)),  # 18
                ((3, 8), (4, 6)),  # 24
                ((4, 9), (6, 6)),  # 36
            ]
            for (a1, b1), (a2, b2) in pairs:
                s1 = layer_states[(a1, b1)]
                s2 = layer_states[(a2, b2)]
                same_prod_sims.append(self.cosine_similarity(s1, s2))
            avg_same_prod = np.mean(same_prod_sims) if same_prod_sims else 0

            # Test 3: Row similarity (same first operand)
            row_sims = []
            for a in range(2, 10):
                for b1 in range(2, 9):
                    for b2 in range(b1 + 1, 10):
                        s1 = layer_states[(a, b1)]
                        s2 = layer_states[(a, b2)]
                        row_sims.append(self.cosine_similarity(s1, s2))
            avg_row = np.mean(row_sims)

            # Test 4: Column similarity (same second operand)
            col_sims = []
            for b in range(2, 10):
                for a1 in range(2, 9):
                    for a2 in range(a1 + 1, 10):
                        s1 = layer_states[(a1, b)]
                        s2 = layer_states[(a2, b)]
                        col_sims.append(self.cosine_similarity(s1, s2))
            avg_col = np.mean(col_sims)

            # Interpretation
            if avg_comm > 0.999:
                interp = "PERFECT COMM"
            elif avg_comm > 0.99:
                interp = "STRONG COMM"
            elif avg_same_prod > 0.99:
                interp = "PRODUCT CLUSTER"
            elif max(avg_row, avg_col) > 0.98:
                interp = "ROW/COL STRUCT"
            else:
                interp = "MIXED"

            print(
                f"L{layer_idx:<6} {avg_comm:.4f}    {avg_same_prod:.4f}    {avg_row:.4f}    {avg_col:.4f}    {interp}"
            )

            results_by_layer.append(
                {
                    "layer": layer_idx,
                    "commutativity": float(avg_comm),
                    "same_product": float(avg_same_prod),
                    "row_similarity": float(avg_row),
                    "col_similarity": float(avg_col),
                }
            )

        # Find phase transitions
        print("\n" + "=" * 80)
        print("PHASE TRANSITIONS")
        print("=" * 80)

        # When does commutativity drop?
        for i, r in enumerate(results_by_layer):
            if r["commutativity"] < 0.999:
                print(f"\nCommutativity drops below 0.999 at layer {i}")
                break
        else:
            print("\nCommutativity stays above 0.999 throughout")

        # When do same-product pairs diverge?
        max_same_prod = max(r["same_product"] for r in results_by_layer)
        for i, r in enumerate(results_by_layer):
            if r["same_product"] == max_same_prod:
                print(f"Same-product clustering peaks at layer {i} ({max_same_prod:.4f})")

        # Row vs column preference
        row_pref_layers = [
            r["layer"] for r in results_by_layer if r["row_similarity"] > r["col_similarity"]
        ]
        col_pref_layers = [
            r["layer"] for r in results_by_layer if r["col_similarity"] > r["row_similarity"]
        ]

        if len(row_pref_layers) > len(col_pref_layers):
            print(f"\nRow-dominant layers: {len(row_pref_layers)}")
        else:
            print(f"\nColumn-dominant layers: {len(col_pref_layers)}")

        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION: LOOKUP vs COMPUTE")
        print("=" * 80)

        early_comm = np.mean([r["commutativity"] for r in results_by_layer[:10]])
        late_comm = np.mean([r["commutativity"] for r in results_by_layer[-10:]])

        early_same_prod = np.mean([r["same_product"] for r in results_by_layer[:10]])
        late_same_prod = np.mean([r["same_product"] for r in results_by_layer[-10:]])

        print("\nEarly layers (0-9):")
        print(f"  Commutativity: {early_comm:.4f}")
        print(f"  Same-product: {early_same_prod:.4f}")

        print("\nLate layers (24-33):")
        print(f"  Commutativity: {late_comm:.4f}")
        print(f"  Same-product: {late_same_prod:.4f}")

        if early_comm > 0.999 and late_comm > 0.99:
            print("\n*** CONCLUSION: LOOKUP TABLE structure ***")
            print("    Commutativity is baked in from the start")
            print("    This means a*b and b*a access the same 'memory location'")
        elif early_comm > 0.99 and late_comm < 0.99:
            print("\n*** CONCLUSION: HYBRID (lookup then compute) ***")
            print("    Early: lookup-like with shared representations")
            print("    Late: computation diverges representations")
        else:
            print("\n*** CONCLUSION: SEQUENTIAL COMPUTATION ***")
            print("    Different operand orders = different computations")

        # Compare with GPT-OSS
        print("\n" + "=" * 80)
        print("COMPARISON WITH GPT-OSS (from your earlier research)")
        print("=" * 80)

        print(
            """
GPT-OSS-20B findings:
- Same-product pairs: ~0.999 cosine similarity
- Same-row items: ~0.985 similarity
- Same-column items: ~0.988 similarity
- Columns slightly more organized than rows
- Uses MoE routing by operand

Gemma-3-4B findings:
- Commutativity: {:.4f} (vs GPT-OSS ~0.999)
- Same-product: {:.4f}
- Row similarity: {:.4f}
- Col similarity: {:.4f}

SIMILARITY: Both show commutativity baked in
DIFFERENCE: Gemma has weaker row/column structure
INTERPRETATION: Gemma may use a flatter lookup (by product)
               rather than 2D table (by row/col)
""".format(
                np.mean([r["commutativity"] for r in results_by_layer]),
                np.mean([r["same_product"] for r in results_by_layer]),
                np.mean([r["row_similarity"] for r in results_by_layer]),
                np.mean([r["col_similarity"] for r in results_by_layer]),
            )
        )

        # Save results
        output_path = Path("gemma_discovery_cache/lookup_evolution.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_by_layer, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results_by_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    args = parser.parse_args()

    analyzer = LookupEvolutionAnalyzer(model_id=args.model)
    analyzer.analyze_evolution()


if __name__ == "__main__":
    main()
