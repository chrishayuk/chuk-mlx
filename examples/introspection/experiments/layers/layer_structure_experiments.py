#!/usr/bin/env python3
"""
layer_structure_experiments.py

Test hypotheses about the compress → disrupt → reconstruct pattern:

1. PC1 = norm hypothesis: Is PC1 just the embedding norm direction?
2. Residual structure: Does projecting out PC1 reveal hidden structure?
3. Semantic separation curve: Map separation across all 18 layers
4. L1 disruption analysis: What does L1 actually do?
5. Embedding router viability: Can L0 residual match L12 accuracy?

Run: uv run python examples/introspection/layer_structure_experiments.py
"""

import json

# Suppress sklearn warnings
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class ExperimentResults:
    """Container for experiment results"""

    model_id: str = ""
    num_layers: int = 0
    hidden_size: int = 0

    # Hypothesis 1: PC1 = norm
    pc1_vs_mean_cosine: dict = field(default_factory=dict)
    pc1_vs_norm_correlation: dict = field(default_factory=dict)

    # Hypothesis 2: Residual structure
    residual_probe_accuracy: dict = field(default_factory=dict)

    # Hypothesis 3: Semantic separation curve
    semantic_separation: dict = field(default_factory=dict)

    # Hypothesis 4: L1 analysis
    l1_delta_analysis: dict = field(default_factory=dict)

    # Hypothesis 5: Router comparison
    router_comparison: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save results to JSON"""

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        data = {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "pc1_vs_mean_cosine": convert(self.pc1_vs_mean_cosine),
            "pc1_vs_norm_correlation": convert(self.pc1_vs_norm_correlation),
            "residual_probe_accuracy": convert(self.residual_probe_accuracy),
            "semantic_separation": convert(self.semantic_separation),
            "l1_delta_analysis": convert(self.l1_delta_analysis),
            "router_comparison": convert(self.router_comparison),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")


class LayerStructureExperiments:
    """
    Comprehensive experiments to understand layer structure.
    """

    def __init__(self, model: Any, tokenizer: Any, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.results = ExperimentResults(model_id=model_id)

        # Detect model structure
        self._detect_structure()
        self.results.num_layers = self.num_layers
        self.results.hidden_size = self.hidden_size

    def _detect_structure(self):
        """Detect model structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
            self._backbone = self.model.model
        elif hasattr(self.model, "layers"):
            self._layers = self.model.layers
            self._backbone = self.model
        else:
            raise ValueError("Cannot detect model layer structure")

        self.num_layers = len(self._layers)

        if hasattr(self._backbone, "hidden_size"):
            self.hidden_size = self._backbone.hidden_size
        elif hasattr(self.model, "args") and hasattr(self.model.args, "hidden_size"):
            self.hidden_size = self.model.args.hidden_size
        else:
            self.hidden_size = 768

    @classmethod
    def from_pretrained(cls, model_id: str) -> "LayerStructureExperiments":
        """Load model for experiments."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def get_layer_activations(self, prompt: str, layers: list[int]) -> dict[int, np.ndarray]:
        """Get activations for all tokens at specified layers."""
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        elif hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers=layers,
                capture_hidden_states=True,
                positions=PositionSelection.ALL,
            )
        )

        input_ids = mx.array([tokens])
        hooks.forward(input_ids)

        results = {}
        for layer in layers:
            if layer in hooks.state.hidden_states:
                h = hooks.state.hidden_states[layer]
                h_f32 = h.astype(mx.float32)
                h_np = np.array(h_f32, copy=False)
                if h_np.ndim == 3:
                    h_np = h_np[0]  # Remove batch dim
                results[layer] = h_np

        return results

    def collect_activations_dataset(
        self,
        prompts: list[str],
        layers: list[int],
        use_last_token: bool = True,
    ) -> dict[int, np.ndarray]:
        """Collect activations for a dataset of prompts."""
        layer_acts = {l: [] for l in layers}

        for prompt in prompts:
            acts = self.get_layer_activations(prompt, layers)
            for layer in layers:
                if layer in acts:
                    if use_last_token:
                        layer_acts[layer].append(acts[layer][-1])
                    else:
                        # Use mean pooling
                        layer_acts[layer].append(acts[layer].mean(axis=0))

        return {l: np.array(a) for l, a in layer_acts.items() if a}

    # ==========================================
    # Hypothesis 1: PC1 = Norm Direction
    # ==========================================

    def test_pc1_norm_hypothesis(self, prompts: list[str]):
        """Test if PC1 is just the norm/mean direction."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 1: PC1 = Norm Direction")
        print("=" * 60)

        from sklearn.decomposition import PCA

        layers_to_test = [0, 1, 2, 3, 6, 9, 12, 15, self.num_layers - 1]
        layers_to_test = [l for l in layers_to_test if l < self.num_layers]

        all_acts = self.collect_activations_dataset(prompts, layers_to_test, use_last_token=False)

        print("\nPC1 vs Mean Direction (cosine similarity):")
        print("-" * 40)

        for layer in layers_to_test:
            if layer not in all_acts:
                continue

            X = all_acts[layer]

            # Compute PC1
            pca = PCA(n_components=1)
            pca.fit(X)
            pc1 = pca.components_[0]

            # Mean direction (normalized)
            mean_dir = X.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 0:
                mean_dir = mean_dir / mean_norm

            # Cosine similarity
            cosine = np.abs(np.dot(pc1, mean_dir))
            self.results.pc1_vs_mean_cosine[layer] = float(cosine)

            # Correlation between PC1 projection and norm
            norms = np.linalg.norm(X, axis=1)
            projections = X @ pc1
            correlation = np.corrcoef(norms, projections)[0, 1]
            self.results.pc1_vs_norm_correlation[layer] = float(correlation)

            indicator = "← NORM" if cosine > 0.95 else ""
            print(
                f"  L{layer:2d}: PC1·mean = {cosine:.4f}, PC1~norm r = {correlation:.4f} {indicator}"
            )

        # Summary
        l0_cosine = self.results.pc1_vs_mean_cosine.get(0, 0)
        if l0_cosine > 0.95:
            print("\n→ CONFIRMED: PC1 is the mean/norm direction at L0")
            print("  The 99.7% variance is just embedding norms, not useful structure")
        else:
            print(f"\n→ PC1 is NOT purely the norm direction (cosine = {l0_cosine:.3f})")

    # ==========================================
    # Hypothesis 2: Residual Structure
    # ==========================================

    def test_residual_structure(self, prompts: list[str], labels: list[int]):
        """Test if projecting out PC1 reveals hidden structure."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 2: Residual Structure")
        print("=" * 60)

        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        layers_to_test = [0, 3, 6, 9, 12]
        layers_to_test = [l for l in layers_to_test if l < self.num_layers]

        all_acts = self.collect_activations_dataset(prompts, layers_to_test, use_last_token=True)

        y = np.array(labels)

        print("\nProbe accuracy: Raw vs Residual (after removing PC1-k)")
        print("-" * 60)

        for layer in layers_to_test:
            if layer not in all_acts:
                continue

            X = all_acts[layer]

            self.results.residual_probe_accuracy[layer] = {}

            # Raw accuracy
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            raw_scores = cross_val_score(clf, X_scaled, y, cv=5)
            raw_acc = raw_scores.mean()
            self.results.residual_probe_accuracy[layer]["raw"] = float(raw_acc)

            print(f"\n  L{layer:2d}:")
            print(f"    Raw:        {raw_acc:.1%}")

            # Test different numbers of PCs to remove
            for k in [1, 2, 5, 10]:
                if k >= X.shape[1]:
                    continue

                pca = PCA(n_components=k)
                pca.fit(X)

                # Project out top-k PCs
                projection = X @ pca.components_.T @ pca.components_
                X_residual = X - projection

                # Train probe on residual
                X_res_scaled = scaler.fit_transform(X_residual)
                res_scores = cross_val_score(clf, X_res_scaled, y, cv=5)
                res_acc = res_scores.mean()
                self.results.residual_probe_accuracy[layer][f"residual_k{k}"] = float(res_acc)

                delta = res_acc - raw_acc
                indicator = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "="
                print(f"    -PC1..{k}:    {res_acc:.1%} ({indicator} {delta:+.1%})")

        # Summary
        l0_raw = self.results.residual_probe_accuracy.get(0, {}).get("raw", 0)
        l0_res = self.results.residual_probe_accuracy.get(0, {}).get("residual_k1", 0)

        if l0_res > l0_raw + 0.02:
            print("\n→ CONFIRMED: Removing PC1 IMPROVES probe accuracy")
            print("  The useful structure is in the residual!")
        elif l0_res < l0_raw - 0.02:
            print("\n→ Removing PC1 HURTS accuracy")
            print("  PC1 contains some useful information")
        else:
            print("\n→ Removing PC1 has minimal effect")

    # ==========================================
    # Hypothesis 3: Semantic Separation Curve
    # ==========================================

    def test_semantic_separation_curve(self, prompts: list[str] | None = None):
        """Map semantic separation across all layers."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 3: Semantic Separation Curve")
        print("=" * 60)

        # Semantic groups
        semantic_groups = {
            "weather": [
                "What is the weather?",
                "Is it raining?",
                "Temperature forecast",
                "Will it be sunny?",
                "How cold is it?",
            ],
            "email": [
                "Send an email",
                "Write a message",
                "Reply to John",
                "Check my inbox",
                "Email the report",
            ],
            "calendar": [
                "Schedule a meeting",
                "Create an event",
                "Book an appointment",
                "Set a reminder",
                "When is my next meeting?",
            ],
            "search": [
                "Search for restaurants",
                "Find hotels nearby",
                "Look up the news",
                "Search the web",
                "Find information about",
            ],
            "factual": [
                "What is the capital?",
                "How many continents?",
                "Explain physics",
                "What year was it?",
                "Who invented the telephone?",
            ],
        }

        all_layers = list(range(self.num_layers))

        print("\nCollecting activations for semantic groups...")
        group_activations = {g: {} for g in semantic_groups}

        for group_name, group_prompts in semantic_groups.items():
            acts = self.collect_activations_dataset(group_prompts, all_layers, use_last_token=True)
            for layer, layer_acts in acts.items():
                group_activations[group_name][layer] = layer_acts

        print("\nSemantic Separation by Layer:")
        print("-" * 50)
        print("Layer  Within   Between  Separation  Pattern")
        print("-" * 50)

        for layer in all_layers:
            within_sims = []
            between_sims = []

            groups = list(semantic_groups.keys())

            for g1 in groups:
                if layer not in group_activations[g1]:
                    continue
                acts1 = group_activations[g1][layer]

                # Within-group
                for i in range(len(acts1)):
                    for j in range(i + 1, len(acts1)):
                        a1, a2 = acts1[i], acts1[j]
                        norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(a1, a2) / (norm1 * norm2)
                            within_sims.append(sim)

                # Between-group
                for g2 in groups:
                    if g1 >= g2 or layer not in group_activations[g2]:
                        continue
                    acts2 = group_activations[g2][layer]
                    for a1 in acts1:
                        for a2 in acts2:
                            norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                            if norm1 > 0 and norm2 > 0:
                                sim = np.dot(a1, a2) / (norm1 * norm2)
                                between_sims.append(sim)

            within = np.mean(within_sims) if within_sims else 0
            between = np.mean(between_sims) if between_sims else 0
            separation = within - between

            self.results.semantic_separation[layer] = {
                "within": float(within),
                "between": float(between),
                "separation": float(separation),
            }

            # Visual pattern
            bar_len = int(separation * 50)
            bar = "█" * max(0, bar_len)

            print(f"L{layer:2d}    {within:.3f}    {between:.3f}    {separation:+.3f}     {bar}")

        # Find disruption and reconstruction points
        separations = [
            self.results.semantic_separation[l]["separation"] for l in range(self.num_layers)
        ]

        # Find minimum (disruption point)
        min_layer = np.argmin(separations)
        min_sep = separations[min_layer]

        # Find recovery point (first layer after min that exceeds L0)
        l0_sep = separations[0]
        recovery_layer = None
        for l in range(min_layer, self.num_layers):
            if separations[l] >= l0_sep:
                recovery_layer = l
                break

        print("-" * 50)
        print(f"\n→ L0 separation: {l0_sep:.3f}")
        print(f"→ Minimum at L{min_layer}: {min_sep:.3f} (disruption)")
        if recovery_layer:
            print(f"→ Recovery at L{recovery_layer}: {separations[recovery_layer]:.3f}")

        # Pattern classification
        if min_sep < l0_sep - 0.05:
            print("\n→ CONFIRMED: Compress → Disrupt → Reconstruct pattern")
            print(f"   Disruption zone: L1-L{min_layer}")
            if recovery_layer:
                print(f"   Reconstruction zone: L{min_layer + 1}-L{recovery_layer}")

    # ==========================================
    # Hypothesis 4: L1 Delta Analysis
    # ==========================================

    def test_l1_delta_analysis(self, prompts: list[str]):
        """Analyze what L1 actually does to representations."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 4: L1 Transformation Analysis")
        print("=" * 60)

        from sklearn.decomposition import PCA

        # Collect L0 and L1 activations with position info
        deltas = []
        positions = []
        l0_norms = []
        l1_norms = []

        for prompt in prompts[:30]:  # Limit for speed
            acts = self.get_layer_activations(prompt, [0, 1])
            if 0 not in acts or 1 not in acts:
                continue

            l0 = acts[0]
            l1 = acts[1]

            for pos in range(min(len(l0), len(l1))):
                delta = l1[pos] - l0[pos]
                deltas.append(delta)
                positions.append(pos)
                l0_norms.append(np.linalg.norm(l0[pos]))
                l1_norms.append(np.linalg.norm(l1[pos]))

        deltas = np.array(deltas)
        positions = np.array(positions)
        l0_norms = np.array(l0_norms)
        l1_norms = np.array(l1_norms)

        print("\n1. Norm Change Analysis:")
        print("-" * 40)
        norm_change = l1_norms / l0_norms
        print(f"   Mean L0 norm: {l0_norms.mean():.2f}")
        print(f"   Mean L1 norm: {l1_norms.mean():.2f}")
        print(f"   Mean norm ratio L1/L0: {norm_change.mean():.3f}")
        print(f"   Std norm ratio: {norm_change.std():.3f}")

        self.results.l1_delta_analysis["norm_ratio_mean"] = float(norm_change.mean())
        self.results.l1_delta_analysis["norm_ratio_std"] = float(norm_change.std())

        print("\n2. Delta Magnitude Analysis:")
        print("-" * 40)
        delta_norms = np.linalg.norm(deltas, axis=1)
        relative_change = delta_norms / l0_norms
        print(f"   Mean delta norm: {delta_norms.mean():.2f}")
        print(f"   Mean relative change: {relative_change.mean():.3f}")

        self.results.l1_delta_analysis["relative_change_mean"] = float(relative_change.mean())

        print("\n3. Position Dependence of Delta:")
        print("-" * 40)

        # Check if delta varies with position
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        # Can we predict position from delta?
        ridge = Ridge()
        pos_scores = cross_val_score(ridge, deltas, positions, cv=5, scoring="r2")
        print(f"   R² for position from delta: {pos_scores.mean():.3f} ± {pos_scores.std():.3f}")

        self.results.l1_delta_analysis["position_from_delta_r2"] = float(pos_scores.mean())

        if pos_scores.mean() > 0.5:
            print("   → Delta is STRONGLY position-dependent (RoPE signature)")
        elif pos_scores.mean() > 0.2:
            print("   → Delta is MODERATELY position-dependent")
        else:
            print("   → Delta is NOT position-dependent")

        print("\n4. Delta Structure (PCA):")
        print("-" * 40)

        pca = PCA(n_components=min(10, deltas.shape[1]))
        pca.fit(deltas)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        print(f"   PC1 explains: {pca.explained_variance_ratio_[0] * 100:.1f}%")
        print(f"   PC1-5 explain: {cumvar[4] * 100:.1f}%")
        print(f"   PC1-10 explain: {cumvar[9] * 100:.1f}%")

        # Check PC correlations with position
        projections = pca.transform(deltas)
        print("\n   PC correlations with position:")
        for i in range(min(5, projections.shape[1])):
            corr = np.corrcoef(projections[:, i], positions)[0, 1]
            indicator = "***" if abs(corr) > 0.3 else ""
            print(f"     PC{i + 1}: r = {corr:.3f} {indicator}")

        self.results.l1_delta_analysis["delta_pc1_variance"] = float(
            pca.explained_variance_ratio_[0]
        )

    # ==========================================
    # Hypothesis 5: Router Comparison
    # ==========================================

    def test_router_comparison(self, prompts: list[str], labels: list[int]):
        """Compare embedding-based vs deep-layer router accuracy."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 5: Router Viability Comparison")
        print("=" * 60)

        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Collect activations at key layers
        key_layers = [0, 1, 3, 6, 9, 12, self.num_layers - 1]
        key_layers = sorted(set(l for l in key_layers if l < self.num_layers))

        all_acts = self.collect_activations_dataset(prompts, key_layers, use_last_token=True)

        y = np.array(labels)

        print("\nRouter Accuracy Comparison:")
        print("-" * 60)
        print(f"{'Approach':<30} {'Accuracy':>10} {'vs L12':>10}")
        print("-" * 60)

        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)

        # Get L12 baseline (or highest available layer)
        best_layer = max(l for l in key_layers if l in all_acts)
        X_best = all_acts[best_layer]
        X_best_scaled = scaler.fit_transform(X_best)
        best_scores = cross_val_score(clf, X_best_scaled, y, cv=5)
        best_acc = best_scores.mean()

        approaches = {}

        for layer in key_layers:
            if layer not in all_acts:
                continue

            X = all_acts[layer]

            # Raw
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(clf, X_scaled, y, cv=5)
            acc = scores.mean()
            name = f"L{layer} raw"
            approaches[name] = acc
            delta = acc - best_acc
            print(f"{name:<30} {acc:>9.1%} {delta:>+9.1%}")

            # Residual (remove PC1)
            if layer in [0, 1, 3]:
                pca = PCA(n_components=1)
                pca.fit(X)
                projection = X @ pca.components_.T @ pca.components_
                X_residual = X - projection
                X_res_scaled = scaler.fit_transform(X_residual)
                res_scores = cross_val_score(clf, X_res_scaled, y, cv=5)
                res_acc = res_scores.mean()
                name = f"L{layer} residual (-PC1)"
                approaches[name] = res_acc
                delta = res_acc - best_acc
                print(f"{name:<30} {res_acc:>9.1%} {delta:>+9.1%}")

        print("-" * 60)

        self.results.router_comparison = {k: float(v) for k, v in approaches.items()}

        # Find best shallow approach
        shallow_approaches = {
            k: v for k, v in approaches.items() if "L0" in k or "L1" in k or "L3" in k
        }
        if shallow_approaches:
            best_shallow = max(shallow_approaches.items(), key=lambda x: x[1])
            gap = best_acc - best_shallow[1]

            print(f"\nBest deep layer: L{best_layer} = {best_acc:.1%}")
            print(f"Best shallow approach: {best_shallow[0]} = {best_shallow[1]:.1%}")
            print(f"Gap: {gap:.1%}")

            if gap < 0.05:
                print("\n→ VIABLE: Shallow router can match deep layers!")
            elif gap < 0.10:
                print("\n→ MARGINAL: Shallow router is close but not equal")
            else:
                print("\n→ NOT VIABLE: Deep layers are significantly better")

    # ==========================================
    # Run All Experiments
    # ==========================================

    def run_all_experiments(self, output_dir: str = "layer_experiments"):
        """Run all hypothesis tests."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("=" * 60)
        print("LAYER STRUCTURE EXPERIMENTS")
        print("Testing: Compress → Disrupt → Reconstruct Hypothesis")
        print("=" * 60)
        print(f"\nModel: {self.model_id}")
        print(f"Layers: {self.num_layers}, Hidden: {self.hidden_size}")

        # Create test dataset
        tool_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "Create a calendar event",
            "Search for restaurants nearby",
            "Get the stock price of Apple",
            "Set a timer for 10 minutes",
            "Book a flight to Paris",
            "Check my schedule",
            "Find hotels in London",
            "Calculate 25 times 4",
            "Convert 100 USD to EUR",
            "Play some music",
            "Set an alarm for 7am",
            "Get directions to the airport",
            "Search for news about AI",
        ]

        no_tool_prompts = [
            "What is the capital of France?",
            "Explain quantum physics",
            "Write a poem about the ocean",
            "What is 2 + 2?",
            "Tell me about Einstein",
            "How do I learn Python?",
            "What is the meaning of life?",
            "Summarize this text",
            "What are prime numbers?",
            "How does gravity work?",
            "What is photosynthesis?",
            "Explain machine learning",
            "What is democracy?",
            "How do computers work?",
            "What causes earthquakes?",
        ]

        prompts = tool_prompts + no_tool_prompts
        labels = [1] * len(tool_prompts) + [0] * len(no_tool_prompts)

        # Run experiments
        self.test_pc1_norm_hypothesis(prompts)
        self.test_residual_structure(prompts, labels)
        self.test_semantic_separation_curve()
        self.test_l1_delta_analysis(prompts)
        self.test_router_comparison(prompts, labels)

        # Summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        # PC1 = norm?
        l0_cosine = self.results.pc1_vs_mean_cosine.get(0, 0)
        print(f"\n1. PC1 = Norm: {'YES' if l0_cosine > 0.95 else 'NO'} (cosine = {l0_cosine:.3f})")

        # Residual structure?
        l0_raw = self.results.residual_probe_accuracy.get(0, {}).get("raw", 0)
        l0_res = self.results.residual_probe_accuracy.get(0, {}).get("residual_k1", 0)
        print(
            f"2. Residual improves L0: {'YES' if l0_res > l0_raw + 0.02 else 'NO'} ({l0_raw:.1%} → {l0_res:.1%})"
        )

        # Disruption pattern?
        separations = [
            self.results.semantic_separation.get(l, {}).get("separation", 0)
            for l in range(self.num_layers)
        ]
        if separations:
            min_layer = np.argmin(separations)
            print(f"3. Disruption point: L{min_layer} (separation = {separations[min_layer]:.3f})")

        # L1 is RoPE?
        pos_r2 = self.results.l1_delta_analysis.get("position_from_delta_r2", 0)
        print(
            f"4. L1 delta is position-dependent: {'YES' if pos_r2 > 0.3 else 'NO'} (R² = {pos_r2:.3f})"
        )

        # Router viability?
        router_results = self.results.router_comparison
        if router_results:
            best_deep = max(v for k, v in router_results.items() if "L12" in k or "L17" in k)
            best_shallow = max(
                v for k, v in router_results.items() if "L0" in k or "L1" in k or "L3" in k
            )
            gap = best_deep - best_shallow
            print(f"5. Shallow router viable: {'YES' if gap < 0.05 else 'NO'} (gap = {gap:.1%})")

        # Save results
        results_path = output_path / "experiment_results.json"
        self.results.save(str(results_path))

        print(f"\nResults saved to: {results_path}")

        return self.results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test compress → disrupt → reconstruct hypothesis")
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/functiongemma-270m-it-bf16",
        help="Model ID to analyze",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run only semantic separation curve (fastest)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    experiments = LayerStructureExperiments.from_pretrained(args.model)
    print(f"Loaded: {experiments.num_layers} layers, {experiments.hidden_size} hidden")

    if args.quick:
        # Just run semantic separation for quick comparison
        experiments.test_semantic_separation_curve()
    else:
        experiments.run_all_experiments()


if __name__ == "__main__":
    main()
