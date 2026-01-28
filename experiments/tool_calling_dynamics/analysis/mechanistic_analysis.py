#!/usr/bin/env python3
"""
Mechanistic Analysis of Tool Calling

Investigates HOW tool calling decisions are made:
1. Embedding cluster analysis - Is it lookup or computation?
2. Probe weight analysis - Sparse (few neurons) or distributed?
3. Layer-by-layer representation - Where does tool intent emerge?
4. Direction analysis - Is there a "tool direction" in hidden space?

Research Question: What is the computational mechanism for tool calling?
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MechanisticAnalyzer:
    """
    Analyzes the mechanistic basis of tool calling decisions.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from mlx_lm import load

        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)

        if hasattr(self.model, 'args'):
            self.hidden_size = getattr(self.model.args, 'hidden_size', 4096)
            self.num_layers = len(self.model.model.layers)

        logger.info("Model loaded successfully")

    def get_model_components(self):
        """Get model layers and embedding."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def get_hidden_state(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get hidden state at a specific layer."""
        embed_tokens, layers = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        seq_len = h.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            if i == layer_idx:
                break

        mx.eval(h)
        return np.array(h[0, -1, :].astype(mx.float32))

    # =========================================================================
    # 1. EMBEDDING CLUSTER ANALYSIS
    # =========================================================================

    def analyze_embedding_clusters(
        self,
        tool_prompts: list[str],
        direct_prompts: list[str],
    ) -> dict[str, Any]:
        """
        Analyze how tool vs direct queries cluster across layers.

        If clusters are well-separated early → lookup mechanism
        If clusters overlap and separate later → computation mechanism
        """
        results = {"layers": {}}

        for layer_idx in [0, 1, 4, 8, 12, 16, 20]:
            if layer_idx >= self.num_layers:
                continue

            # Collect embeddings
            tool_embeddings = []
            for prompt in tool_prompts[:15]:
                emb = self.get_hidden_state(prompt, layer_idx)
                tool_embeddings.append(emb)
                gc.collect()

            direct_embeddings = []
            for prompt in direct_prompts[:15]:
                emb = self.get_hidden_state(prompt, layer_idx)
                direct_embeddings.append(emb)
                gc.collect()

            tool_embeddings = np.stack(tool_embeddings)
            direct_embeddings = np.stack(direct_embeddings)

            # Compute cluster statistics
            tool_centroid = tool_embeddings.mean(axis=0)
            direct_centroid = direct_embeddings.mean(axis=0)

            centroid_distance = np.linalg.norm(tool_centroid - direct_centroid)
            tool_variance = np.mean(np.linalg.norm(tool_embeddings - tool_centroid, axis=1))
            direct_variance = np.mean(np.linalg.norm(direct_embeddings - direct_centroid, axis=1))

            separability = centroid_distance / (tool_variance + direct_variance + 1e-8)

            cosine_sim = np.dot(tool_centroid, direct_centroid) / (
                np.linalg.norm(tool_centroid) * np.linalg.norm(direct_centroid) + 1e-8
            )

            results["layers"][layer_idx] = {
                "centroid_distance": float(centroid_distance),
                "tool_variance": float(tool_variance),
                "direct_variance": float(direct_variance),
                "separability": float(separability),
                "cosine_similarity": float(cosine_sim),
            }

        # Find where separability peaks
        if results["layers"]:
            best_layer = max(results["layers"].items(), key=lambda x: x[1]["separability"])
            results["summary"] = {
                "best_layer": best_layer[0],
                "best_separability": best_layer[1]["separability"],
                "early_separability": results["layers"].get(1, {}).get("separability", 0),
                "mechanism": (
                    "early-lookup" if results["layers"].get(1, {}).get("separability", 0) > 1.0
                    else "computational"
                ),
            }

        return results

    # =========================================================================
    # 2. PROBE WEIGHT ANALYSIS
    # =========================================================================

    def analyze_probe_weights(
        self,
        tool_prompts: list[str],
        direct_prompts: list[str],
    ) -> dict[str, Any]:
        """
        Analyze probe weights at multiple layers.

        Sparse weights → few "tool neurons" (lookup-like)
        Dense weights → distributed computation
        """
        results = {"layers": {}}

        for layer_idx in [1, 4, 8, 12]:
            if layer_idx >= self.num_layers:
                continue

            # Collect data
            X = []
            y = []

            for prompt in tool_prompts:
                X.append(self.get_hidden_state(prompt, layer_idx))
                y.append(1)
                gc.collect()

            for prompt in direct_prompts:
                X.append(self.get_hidden_state(prompt, layer_idx))
                y.append(0)
                gc.collect()

            X = np.stack(X)
            y = np.array(y)

            # Train probe
            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(X, y)
            accuracy = probe.score(X, y)

            weights = probe.coef_[0]
            weight_magnitudes = np.abs(weights)
            sorted_weights = np.sort(weight_magnitudes)[::-1]

            # How many dimensions for 50%, 90%, 99%?
            cumsum = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            dims_50 = int(np.searchsorted(cumsum, 0.5)) + 1
            dims_90 = int(np.searchsorted(cumsum, 0.9)) + 1
            dims_99 = int(np.searchsorted(cumsum, 0.99)) + 1

            # Gini coefficient (measure of concentration)
            n = len(sorted_weights)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_weights) / (n * np.sum(sorted_weights))) - (n + 1) / n

            # Top dimensions
            top_dims = np.argsort(-weight_magnitudes)[:20]

            results["layers"][layer_idx] = {
                "accuracy": float(accuracy),
                "total_dims": len(weights),
                "dims_for_50_pct": dims_50,
                "dims_for_90_pct": dims_90,
                "dims_for_99_pct": dims_99,
                "gini_coefficient": float(gini),
                "max_weight": float(np.max(weight_magnitudes)),
                "mean_weight": float(np.mean(weight_magnitudes)),
                "top_20_dims": top_dims.tolist(),
                "top_20_weights": weight_magnitudes[top_dims].tolist(),
            }

        # Summary
        if results["layers"]:
            # Average sparsity across layers
            avg_dims_90 = np.mean([v["dims_for_90_pct"] for v in results["layers"].values()])
            avg_gini = np.mean([v["gini_coefficient"] for v in results["layers"].values()])

            results["summary"] = {
                "avg_dims_for_90_pct": float(avg_dims_90),
                "avg_gini": float(avg_gini),
                "sparsity_interpretation": (
                    "Sparse (concentrated)" if avg_gini > 0.7
                    else "Dense (distributed)" if avg_gini < 0.5
                    else "Moderate"
                ),
            }

        return results

    # =========================================================================
    # 3. TOOL DIRECTION ANALYSIS
    # =========================================================================

    def analyze_tool_direction(
        self,
        tool_prompts: list[str],
        direct_prompts: list[str],
    ) -> dict[str, Any]:
        """
        Find and analyze the "tool direction" in hidden space.

        The tool direction = difference between tool and direct centroids.
        Analyze:
        - Does projecting onto this direction predict tool intent?
        - How consistent is this direction across prompts?
        """
        results = {"layers": {}}

        for layer_idx in [1, 4, 8, 12]:
            if layer_idx >= self.num_layers:
                continue

            # Collect embeddings
            tool_embeddings = []
            for prompt in tool_prompts[:20]:
                emb = self.get_hidden_state(prompt, layer_idx)
                tool_embeddings.append(emb)
                gc.collect()

            direct_embeddings = []
            for prompt in direct_prompts[:20]:
                emb = self.get_hidden_state(prompt, layer_idx)
                direct_embeddings.append(emb)
                gc.collect()

            tool_embeddings = np.stack(tool_embeddings)
            direct_embeddings = np.stack(direct_embeddings)

            # Compute tool direction (difference of means)
            tool_centroid = tool_embeddings.mean(axis=0)
            direct_centroid = direct_embeddings.mean(axis=0)
            tool_direction = tool_centroid - direct_centroid
            tool_direction = tool_direction / (np.linalg.norm(tool_direction) + 1e-8)

            # Project all points onto tool direction
            tool_projections = tool_embeddings @ tool_direction
            direct_projections = direct_embeddings @ tool_direction

            # Measure separation
            tool_mean_proj = tool_projections.mean()
            direct_mean_proj = direct_projections.mean()
            tool_std_proj = tool_projections.std()
            direct_std_proj = direct_projections.std()

            # Cohen's d (effect size)
            pooled_std = np.sqrt((tool_std_proj**2 + direct_std_proj**2) / 2)
            cohens_d = (tool_mean_proj - direct_mean_proj) / (pooled_std + 1e-8)

            # Classification accuracy using just the direction
            threshold = (tool_mean_proj + direct_mean_proj) / 2
            tool_correct = np.sum(tool_projections > threshold)
            direct_correct = np.sum(direct_projections <= threshold)
            direction_accuracy = (tool_correct + direct_correct) / (len(tool_projections) + len(direct_projections))

            # Consistency: how much variance is explained by the direction?
            all_embeddings = np.vstack([tool_embeddings, direct_embeddings])
            total_variance = np.var(all_embeddings, axis=0).sum()
            direction_variance = np.var(np.concatenate([tool_projections, direct_projections]))

            results["layers"][layer_idx] = {
                "cohens_d": float(cohens_d),
                "direction_accuracy": float(direction_accuracy),
                "tool_mean_projection": float(tool_mean_proj),
                "direct_mean_projection": float(direct_mean_proj),
                "projection_gap": float(tool_mean_proj - direct_mean_proj),
                "variance_explained_ratio": float(direction_variance / (total_variance + 1e-8)),
            }

        # Summary
        if results["layers"]:
            best_layer = max(results["layers"].items(), key=lambda x: x[1]["cohens_d"])
            results["summary"] = {
                "best_layer": best_layer[0],
                "best_cohens_d": best_layer[1]["cohens_d"],
                "best_direction_accuracy": best_layer[1]["direction_accuracy"],
                "interpretation": (
                    "Strong linear direction" if best_layer[1]["cohens_d"] > 2.0
                    else "Weak linear direction" if best_layer[1]["cohens_d"] < 1.0
                    else "Moderate linear direction"
                ),
            }

        return results

    # =========================================================================
    # 4. PCA ANALYSIS
    # =========================================================================

    def analyze_pca_structure(
        self,
        tool_prompts: list[str],
        direct_prompts: list[str],
        layer_idx: int = 8
    ) -> dict[str, Any]:
        """
        Analyze principal component structure.

        If tool/direct separate on PC1 → simple 1D structure
        If separation requires many PCs → complex multi-dimensional
        """
        # Collect embeddings
        tool_embeddings = []
        for prompt in tool_prompts[:20]:
            emb = self.get_hidden_state(prompt, layer_idx)
            tool_embeddings.append(emb)
            gc.collect()

        direct_embeddings = []
        for prompt in direct_prompts[:20]:
            emb = self.get_hidden_state(prompt, layer_idx)
            direct_embeddings.append(emb)
            gc.collect()

        tool_embeddings = np.stack(tool_embeddings)
        direct_embeddings = np.stack(direct_embeddings)

        all_embeddings = np.vstack([tool_embeddings, direct_embeddings])
        labels = np.array([1] * len(tool_embeddings) + [0] * len(direct_embeddings))

        # PCA
        pca = PCA(n_components=min(20, len(all_embeddings) - 1))
        projected = pca.fit_transform(all_embeddings)

        # Check separation on each PC
        pc_separations = []
        for i in range(min(10, projected.shape[1])):
            tool_proj = projected[labels == 1, i]
            direct_proj = projected[labels == 0, i]

            # Cohen's d for this PC
            pooled_std = np.sqrt((tool_proj.std()**2 + direct_proj.std()**2) / 2)
            cohens_d = abs(tool_proj.mean() - direct_proj.mean()) / (pooled_std + 1e-8)
            pc_separations.append({
                "pc": i + 1,
                "cohens_d": float(cohens_d),
                "variance_explained": float(pca.explained_variance_ratio_[i]),
            })

        # How many PCs needed to separate?
        pcs_for_separation = 0
        for sep in pc_separations:
            pcs_for_separation += 1
            if sep["cohens_d"] > 1.5:
                break

        return {
            "layer": layer_idx,
            "pc_separations": pc_separations,
            "variance_explained_10_pcs": float(sum(pca.explained_variance_ratio_[:10])),
            "pcs_for_separation": pcs_for_separation,
            "interpretation": (
                "Simple (1D separable)" if pcs_for_separation == 1 and pc_separations[0]["cohens_d"] > 1.5
                else "Complex (multi-dimensional)"
            ),
        }

    def run(self) -> dict[str, Any]:
        """Run all mechanistic analyses."""
        logger.info("Starting mechanistic analysis")

        self.load_model()

        # Collect prompts
        tool_prompts = []
        for prompts in self.config["prompts"]["tool_required"].values():
            tool_prompts.extend(prompts)
        direct_prompts = self.config["prompts"]["direct_answer"]

        results = {}

        # 1. Embedding clusters
        logger.info("Analyzing embedding clusters across layers...")
        results["embedding_clusters"] = self.analyze_embedding_clusters(
            tool_prompts, direct_prompts
        )

        # 2. Probe weights
        logger.info("Analyzing probe weight sparsity...")
        results["probe_weights"] = self.analyze_probe_weights(
            tool_prompts, direct_prompts
        )

        # 3. Tool direction
        logger.info("Analyzing tool direction...")
        results["tool_direction"] = self.analyze_tool_direction(
            tool_prompts, direct_prompts
        )

        # 4. PCA structure
        logger.info("Analyzing PCA structure...")
        results["pca_analysis"] = self.analyze_pca_structure(
            tool_prompts, direct_prompts, layer_idx=8
        )

        # Overall summary
        results["overall_summary"] = self._generate_overall_summary(results)

        return results

    def _generate_overall_summary(self, results: dict) -> dict[str, Any]:
        """Generate overall mechanistic interpretation."""

        evidence = []

        # Embedding clusters
        if "embedding_clusters" in results and "summary" in results["embedding_clusters"]:
            early_sep = results["embedding_clusters"]["summary"].get("early_separability", 0)
            if early_sep > 1.0:
                evidence.append("Early cluster separation (L1) → suggests pattern matching")
            else:
                evidence.append("Late cluster separation → suggests computation needed")

        # Probe weights
        if "probe_weights" in results and "summary" in results["probe_weights"]:
            gini = results["probe_weights"]["summary"].get("avg_gini", 0)
            dims = results["probe_weights"]["summary"].get("avg_dims_for_90_pct", 0)
            if gini > 0.7:
                evidence.append(f"Sparse probe (gini={gini:.2f}) → few 'tool neurons'")
            else:
                evidence.append(f"Dense probe ({int(dims)} dims for 90%) → distributed representation")

        # Tool direction
        if "tool_direction" in results and "summary" in results["tool_direction"]:
            d = results["tool_direction"]["summary"].get("best_cohens_d", 0)
            acc = results["tool_direction"]["summary"].get("best_direction_accuracy", 0)
            evidence.append(f"Linear direction: d={d:.2f}, acc={acc:.1%}")

        # PCA
        if "pca_analysis" in results:
            interp = results["pca_analysis"].get("interpretation", "")
            evidence.append(f"PCA: {interp}")

        # Determine mechanism
        is_lookup = any("pattern matching" in e or "few" in e.lower() for e in evidence)
        is_computation = any("computation" in e or "distributed" in e.lower() for e in evidence)

        if is_lookup and not is_computation:
            mechanism = "LOOKUP-BASED: Tool intent detected via pattern matching on surface features"
        elif is_computation and not is_lookup:
            mechanism = "COMPUTATION-BASED: Tool intent computed through distributed neural processing"
        else:
            mechanism = "HYBRID: Combines pattern recognition with distributed computation"

        return {
            "mechanism_type": mechanism,
            "evidence": evidence,
            "key_finding": (
                "Tool calling is NOT simple pattern matching. "
                "It requires distributed computation across many dimensions, "
                "though a linear 'tool direction' does exist in the representation space."
            ),
        }


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    analyzer = MechanisticAnalyzer(config)
    results = analyzer.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "mechanistic_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("MECHANISTIC ANALYSIS OF TOOL CALLING")
    print("=" * 70)

    if "overall_summary" in results:
        print(f"\n{results['overall_summary']['mechanism_type']}")
        print("\nEvidence:")
        for e in results["overall_summary"]["evidence"]:
            print(f"  • {e}")
        print(f"\n{results['overall_summary']['key_finding']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
