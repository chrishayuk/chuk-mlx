#!/usr/bin/env python3
"""
Gemma Lookup Table Analysis: Is Gemma using lookup tables for multiplication?

This script investigates whether Gemma stores multiplication as lookup tables
(like GPT-OSS) vs computing it algorithmically.

Key tests:
1. Same-product similarity: Do 2*6 and 3*4 have similar activations? (Both = 12)
2. Row/column structure: Are same-row items (7*2, 7*3, 7*4) clustered?
3. Commutativity: Are a*b and b*a nearly identical?
4. Neighborhood analysis: What "wrong" answers activate alongside correct ones?
5. Computation vs retrieval: Does the pattern suggest lookup or calculation?

Comparison with GPT-OSS findings:
- GPT-OSS showed clear row/column structure
- GPT-OSS had same-product pairs with high similarity
- GPT-OSS experts specialized by operand value

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_lookup_table_analysis.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class MultiplicationEntry:
    """Entry in the times table analysis."""
    a: int
    b: int
    product: int
    prompt: str
    hidden_state: np.ndarray  # at the analysis layer


class LookupTableAnalyzer:
    """Analyze if Gemma uses lookup tables for multiplication."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None
        self.entries: list[MultiplicationEntry] = []

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
            embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, embed_scale

    def get_hidden_state(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get hidden state at a specific layer for a prompt."""
        layers, embed, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
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
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    def collect_times_table(self, layer_idx: int = 24):
        """Collect hidden states for all times table entries."""
        print(f"\nCollecting times table at layer {layer_idx}...")

        self.entries = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompt = f"{a} * {b} = "
                product = a * b

                h = self.get_hidden_state(prompt, layer_idx)

                entry = MultiplicationEntry(
                    a=a, b=b, product=product,
                    prompt=prompt, hidden_state=h
                )
                self.entries.append(entry)

        print(f"  Collected {len(self.entries)} entries")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_commutativity(self) -> dict:
        """
        Test 1: Commutativity

        If it's a lookup table, a*b and b*a should have nearly identical activations.
        If it's computing, they might differ (different input sequence).
        """
        print("\n" + "=" * 70)
        print("TEST 1: COMMUTATIVITY (a*b vs b*a)")
        print("=" * 70)

        similarities = []

        print(f"\n{'Pair':<15} {'Similarity':<12} {'Interpretation'}")
        print("-" * 50)

        for a in range(2, 9):
            for b in range(a + 1, 10):  # Only upper triangle
                # Find a*b and b*a
                entry_ab = next(e for e in self.entries if e.a == a and e.b == b)
                entry_ba = next(e for e in self.entries if e.a == b and e.b == a)

                sim = self.cosine_similarity(entry_ab.hidden_state, entry_ba.hidden_state)
                similarities.append(sim)

                interp = "IDENTICAL" if sim > 0.999 else "SIMILAR" if sim > 0.99 else "DIFFERENT"
                print(f"{a}*{b} vs {b}*{a}  {sim:.6f}     {interp}")

        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)

        print(f"\nSummary:")
        print(f"  Average similarity: {avg_sim:.6f}")
        print(f"  Min similarity: {min_sim:.6f}")
        print(f"  Max similarity: {max_sim:.6f}")

        if avg_sim > 0.999:
            print(f"\n  CONCLUSION: Commutativity is BAKED IN (avg > 0.999)")
            print(f"  This suggests LOOKUP TABLE structure")
        elif avg_sim > 0.99:
            print(f"\n  CONCLUSION: Strong commutativity (avg > 0.99)")
            print(f"  This suggests computation with shared intermediate states")
        else:
            print(f"\n  CONCLUSION: Weak commutativity (avg < 0.99)")
            print(f"  This suggests SEQUENTIAL COMPUTATION")

        return {
            "avg_similarity": avg_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
            "all_similarities": similarities,
        }

    def test_same_product(self) -> dict:
        """
        Test 2: Same-product pairs

        If it's a lookup table indexed by product, then different operand
        pairs with the same product (e.g., 2*6 and 3*4, both = 12) should
        have similar activations.
        """
        print("\n" + "=" * 70)
        print("TEST 2: SAME-PRODUCT PAIRS")
        print("=" * 70)

        # Group entries by product
        by_product = defaultdict(list)
        for e in self.entries:
            by_product[e.product].append(e)

        # Find products with multiple distinct pairs
        multi_pairs = {p: entries for p, entries in by_product.items()
                       if len(set((e.a, e.b) for e in entries)) > 1}

        print(f"\nProducts with multiple operand pairs: {sorted(multi_pairs.keys())}")

        results = []

        print(f"\n{'Product':<10} {'Pairs':<25} {'Similarity':<12}")
        print("-" * 50)

        for product in sorted(multi_pairs.keys()):
            entries = multi_pairs[product]

            # Get unique pairs (considering commutativity)
            unique_pairs = []
            seen = set()
            for e in entries:
                key = tuple(sorted([e.a, e.b]))
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append(e)

            if len(unique_pairs) < 2:
                continue

            # Compare all pairs of unique entries
            for i in range(len(unique_pairs)):
                for j in range(i + 1, len(unique_pairs)):
                    e1, e2 = unique_pairs[i], unique_pairs[j]
                    sim = self.cosine_similarity(e1.hidden_state, e2.hidden_state)

                    pair_str = f"{e1.a}*{e1.b} vs {e2.a}*{e2.b}"
                    print(f"{product:<10} {pair_str:<25} {sim:.6f}")

                    results.append({
                        "product": product,
                        "pair1": (e1.a, e1.b),
                        "pair2": (e2.a, e2.b),
                        "similarity": sim,
                    })

        if results:
            avg_sim = np.mean([r["similarity"] for r in results])
            print(f"\nAverage same-product similarity: {avg_sim:.6f}")

            if avg_sim > 0.95:
                print("\n  CONCLUSION: Same-product pairs are CLUSTERED")
                print("  This suggests PRODUCT-INDEXED lookup")
            elif avg_sim > 0.8:
                print("\n  CONCLUSION: Moderate same-product clustering")
                print("  This suggests PARTIAL product association")
            else:
                print("\n  CONCLUSION: Same-product pairs are DISTINCT")
                print("  This suggests OPERAND-BASED storage, not product-based")

        return {"results": results}

    def test_row_column_structure(self) -> dict:
        """
        Test 3: Row/Column structure

        If stored as a 2D table:
        - Same row (a*2, a*3, a*4...) should cluster
        - Same column (2*b, 3*b, 4*b...) should cluster
        """
        print("\n" + "=" * 70)
        print("TEST 3: ROW/COLUMN STRUCTURE")
        print("=" * 70)

        row_sims = []  # same first operand
        col_sims = []  # same second operand
        random_sims = []  # different row and column

        for i, e1 in enumerate(self.entries):
            for j, e2 in enumerate(self.entries):
                if i >= j:
                    continue

                sim = self.cosine_similarity(e1.hidden_state, e2.hidden_state)

                if e1.a == e2.a and e1.b != e2.b:
                    row_sims.append(sim)
                elif e1.b == e2.b and e1.a != e2.a:
                    col_sims.append(sim)
                elif e1.a != e2.a and e1.b != e2.b:
                    random_sims.append(sim)

        avg_row = np.mean(row_sims) if row_sims else 0
        avg_col = np.mean(col_sims) if col_sims else 0
        avg_random = np.mean(random_sims) if random_sims else 0

        print(f"\n{'Category':<20} {'Count':<10} {'Avg Similarity':<15}")
        print("-" * 50)
        print(f"{'Same row (a*_)':<20} {len(row_sims):<10} {avg_row:.6f}")
        print(f"{'Same column (_*b)':<20} {len(col_sims):<10} {avg_col:.6f}")
        print(f"{'Different both':<20} {len(random_sims):<10} {avg_random:.6f}")

        row_vs_random = avg_row - avg_random
        col_vs_random = avg_col - avg_random

        print(f"\nRow advantage over random: {row_vs_random:+.6f}")
        print(f"Column advantage over random: {col_vs_random:+.6f}")

        if avg_row > avg_col:
            print(f"\n  Structure: ROW-DOMINANT (first operand matters more)")
        elif avg_col > avg_row:
            print(f"\n  Structure: COLUMN-DOMINANT (second operand matters more)")
        else:
            print(f"\n  Structure: BALANCED (both operands matter equally)")

        if max(row_vs_random, col_vs_random) > 0.05:
            print(f"  CONCLUSION: Clear 2D TABLE structure detected")
        else:
            print(f"  CONCLUSION: Weak or no 2D structure")

        return {
            "avg_row_similarity": avg_row,
            "avg_col_similarity": avg_col,
            "avg_random_similarity": avg_random,
            "row_advantage": row_vs_random,
            "col_advantage": col_vs_random,
        }

    def test_product_proximity(self) -> dict:
        """
        Test 4: Product proximity

        Are products with similar values (e.g., 35, 36, 40, 42) clustered?
        This would suggest numerical/ordinal organization.
        """
        print("\n" + "=" * 70)
        print("TEST 4: PRODUCT PROXIMITY")
        print("=" * 70)

        # Compute similarities and product distances
        data = []

        for i, e1 in enumerate(self.entries):
            for j, e2 in enumerate(self.entries):
                if i >= j:
                    continue

                sim = self.cosine_similarity(e1.hidden_state, e2.hidden_state)
                product_diff = abs(e1.product - e2.product)

                data.append({
                    "pair": (f"{e1.a}*{e1.b}", f"{e2.a}*{e2.b}"),
                    "products": (e1.product, e2.product),
                    "product_diff": product_diff,
                    "similarity": sim,
                })

        # Bin by product difference
        bins = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]

        print(f"\n{'Product Diff':<15} {'Count':<10} {'Avg Similarity':<15}")
        print("-" * 45)

        correlations = []
        for low, high in bins:
            matches = [d for d in data if low <= d["product_diff"] <= high]
            if matches:
                avg_sim = np.mean([d["similarity"] for d in matches])
                print(f"{low}-{high:<13} {len(matches):<10} {avg_sim:.6f}")
                correlations.append(((low + high) / 2, avg_sim))

        # Compute correlation between product distance and similarity
        if len(data) > 10:
            diffs = np.array([d["product_diff"] for d in data])
            sims = np.array([d["similarity"] for d in data])
            correlation = np.corrcoef(diffs, sims)[0, 1]

            print(f"\nCorrelation (product_diff vs similarity): {correlation:.4f}")

            if correlation < -0.3:
                print("  CONCLUSION: Strong NUMERICAL ordering (similar products cluster)")
            elif correlation < -0.1:
                print("  CONCLUSION: Weak numerical ordering")
            else:
                print("  CONCLUSION: No numerical ordering (not organized by product value)")

        return {"data": data}

    def test_clustering(self) -> dict:
        """
        Test 5: Natural clustering

        What clusters naturally emerge from the activation space?
        Do they correspond to rows, columns, products, or something else?
        """
        print("\n" + "=" * 70)
        print("TEST 5: NATURAL CLUSTERING (K-Means)")
        print("=" * 70)

        X = np.array([e.hidden_state for e in self.entries])

        # Try different numbers of clusters
        for n_clusters in [4, 8, 9]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            print(f"\n{n_clusters} clusters:")

            for cluster_id in range(n_clusters):
                members = [self.entries[i] for i, l in enumerate(labels) if l == cluster_id]

                if not members:
                    continue

                # Analyze cluster composition
                first_ops = [e.a for e in members]
                second_ops = [e.b for e in members]
                products = [e.product for e in members]

                # Check if cluster is row-based (same first operand)
                if len(set(first_ops)) == 1:
                    cluster_type = f"ROW {first_ops[0]}*_"
                # Check if column-based
                elif len(set(second_ops)) == 1:
                    cluster_type = f"COL _*{second_ops[0]}"
                # Check if product-range based
                elif max(products) - min(products) < 20:
                    cluster_type = f"PRODUCTS {min(products)}-{max(products)}"
                else:
                    cluster_type = "MIXED"

                member_str = ", ".join(f"{e.a}*{e.b}" for e in members[:5])
                if len(members) > 5:
                    member_str += f"... ({len(members)} total)"

                print(f"  Cluster {cluster_id}: {cluster_type}")
                print(f"    Members: {member_str}")

        return {}

    def run_all_tests(self, layer_idx: int = 24):
        """Run all lookup table tests."""
        self.load_model()
        self.collect_times_table(layer_idx)

        results = {}

        results["commutativity"] = self.test_commutativity()
        results["same_product"] = self.test_same_product()
        results["row_column"] = self.test_row_column_structure()
        results["product_proximity"] = self.test_product_proximity()
        results["clustering"] = self.test_clustering()

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY: Is Gemma using a Lookup Table?")
        print("=" * 70)

        comm_avg = results["commutativity"]["avg_similarity"]
        row_adv = results["row_column"]["row_advantage"]
        col_adv = results["row_column"]["col_advantage"]

        evidence_for_lookup = []
        evidence_against_lookup = []

        if comm_avg > 0.999:
            evidence_for_lookup.append(f"Perfect commutativity ({comm_avg:.4f})")
        elif comm_avg > 0.99:
            evidence_for_lookup.append(f"Strong commutativity ({comm_avg:.4f})")
        else:
            evidence_against_lookup.append(f"Imperfect commutativity ({comm_avg:.4f})")

        if max(row_adv, col_adv) > 0.05:
            evidence_for_lookup.append(f"Clear row/column structure (adv: {max(row_adv, col_adv):.4f})")
        else:
            evidence_against_lookup.append(f"Weak row/column structure")

        print("\nEvidence FOR lookup table:")
        for e in evidence_for_lookup:
            print(f"  + {e}")

        print("\nEvidence AGAINST lookup table:")
        for e in evidence_against_lookup:
            print(f"  - {e}")

        if len(evidence_for_lookup) > len(evidence_against_lookup):
            print("\n*** CONCLUSION: Gemma likely uses LOOKUP TABLE for multiplication ***")
        else:
            print("\n*** CONCLUSION: Gemma may use COMPUTATION rather than pure lookup ***")

        # Save results
        output_path = Path("gemma_discovery_cache/lookup_table_analysis.json")
        output_path.parent.mkdir(exist_ok=True)

        # Convert numpy types to Python types for JSON
        json_results = {
            "layer": layer_idx,
            "commutativity": {
                "avg_similarity": float(results["commutativity"]["avg_similarity"]),
                "min_similarity": float(results["commutativity"]["min_similarity"]),
                "max_similarity": float(results["commutativity"]["max_similarity"]),
            },
            "row_column": {
                "avg_row_similarity": float(results["row_column"]["avg_row_similarity"]),
                "avg_col_similarity": float(results["row_column"]["avg_col_similarity"]),
                "avg_random_similarity": float(results["row_column"]["avg_random_similarity"]),
            },
            "conclusion": "LOOKUP" if len(evidence_for_lookup) > len(evidence_against_lookup) else "COMPUTE",
        }

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--layer", "-l", type=int, default=24,
                        help="Layer to analyze (default: 24, computation layer)")
    args = parser.parse_args()

    analyzer = LookupTableAnalyzer(model_id=args.model)
    analyzer.run_all_tests(layer_idx=args.layer)


if __name__ == "__main__":
    main()
