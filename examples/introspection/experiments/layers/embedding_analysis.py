#!/usr/bin/env python3
"""
embedding_analysis.py

Analyze the embedding matrix directly, not L0 activations.
This proves what information exists BEFORE any computation.

The key distinction:
    Token IDs → Embedding Matrix → Raw Embedding → (LayerNorm?) → L0 Activation
                      ↑
                This is what we analyze

Run: uv run python examples/introspection/embedding_analysis.py
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
class EmbeddingAnalysisResults:
    """Container for embedding analysis results"""

    model_id: str = ""
    vocab_size: int = 0
    hidden_dim: int = 0

    # Structure analysis
    pc1_variance: float = 0
    dims_for_90_variance: int = 0
    dims_for_95_variance: int = 0
    pc1_norm_correlation: float = 0

    # Category analysis
    within_category_similarity: float = 0
    between_category_similarity: float = 0
    category_separation: float = 0

    # Tool clustering
    tool_category_accuracy: float = 0

    # Feature probes
    probe_results: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save results to JSON"""
        data = {
            "model_id": self.model_id,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "pc1_variance": self.pc1_variance,
            "dims_for_90_variance": self.dims_for_90_variance,
            "dims_for_95_variance": self.dims_for_95_variance,
            "pc1_norm_correlation": self.pc1_norm_correlation,
            "within_category_similarity": self.within_category_similarity,
            "between_category_similarity": self.between_category_similarity,
            "category_separation": self.category_separation,
            "tool_category_accuracy": self.tool_category_accuracy,
            "probe_results": self.probe_results,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")


class EmbeddingAnalyzer:
    """
    Analyze raw embedding matrix to prove what information
    exists BEFORE any transformer computation.
    """

    def __init__(self, model: Any, tokenizer: Any, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.results = EmbeddingAnalysisResults(model_id=model_id)

        # Extract the raw embedding matrix
        self._extract_embeddings()

    def _extract_embeddings(self):
        """Extract the embedding matrix from the model."""
        # Try different model structures
        embed_layer = None

        if hasattr(self.model, "model"):
            backbone = self.model.model
            if hasattr(backbone, "embed_tokens"):
                embed_layer = backbone.embed_tokens
            elif hasattr(backbone, "wte"):
                embed_layer = backbone.wte
            elif hasattr(backbone, "embeddings"):
                embed_layer = backbone.embeddings

        if embed_layer is None and hasattr(self.model, "embed_tokens"):
            embed_layer = self.model.embed_tokens

        if embed_layer is None:
            raise ValueError("Cannot find embedding layer in model")

        # Get weight matrix
        if hasattr(embed_layer, "weight"):
            weight = embed_layer.weight
        else:
            raise ValueError("Embedding layer has no weight attribute")

        # Convert to numpy (handle bfloat16)
        weight_f32 = weight.astype(mx.float32)
        self.embeddings = np.array(weight_f32, copy=False)

        self.vocab_size = self.embeddings.shape[0]
        self.hidden_dim = self.embeddings.shape[1]

        self.results.vocab_size = self.vocab_size
        self.results.hidden_dim = self.hidden_dim

        print(f"Embedding matrix: {self.vocab_size} tokens × {self.hidden_dim} dims")

    @classmethod
    def from_pretrained(cls, model_id: str) -> "EmbeddingAnalyzer":
        """Load model for embedding analysis."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def _get_token_ids(self, word: str) -> list[int]:
        """Get token IDs for a word, trying different variants."""
        variants = [word, word.lower(), word.capitalize(), f" {word}", f"▁{word}"]

        for variant in variants:
            try:
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()
                elif hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                if tokens:
                    return tokens
            except:
                pass
        return []

    # ==========================================
    # 1. Embedding Space Structure
    # ==========================================

    def analyze_embedding_structure(self, n_samples: int = 5000):
        """What's the structure of the embedding space?"""
        print("\n" + "=" * 60)
        print("EMBEDDING SPACE STRUCTURE")
        print("=" * 60)

        from sklearn.decomposition import PCA

        # Sample tokens (full vocab might be too large)
        np.random.seed(42)
        sample_idx = np.random.choice(
            self.vocab_size, min(n_samples, self.vocab_size), replace=False
        )
        X = self.embeddings[sample_idx]

        # PCA
        n_components = min(50, self.hidden_dim, len(X) - 1)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        cumvar = np.cumsum(pca.explained_variance_ratio_)

        self.results.pc1_variance = float(pca.explained_variance_ratio_[0])

        print("\nVariance explained:")
        print(f"  PC1:    {pca.explained_variance_ratio_[0] * 100:.1f}%")
        if len(cumvar) > 4:
            print(f"  PC1-5:  {cumvar[4] * 100:.1f}%")
        if len(cumvar) > 9:
            print(f"  PC1-10: {cumvar[9] * 100:.1f}%")
        if len(cumvar) > 19:
            print(f"  PC1-20: {cumvar[19] * 100:.1f}%")

        n_90 = np.searchsorted(cumvar, 0.9) + 1
        n_95 = np.searchsorted(cumvar, 0.95) + 1

        self.results.dims_for_90_variance = int(n_90)
        self.results.dims_for_95_variance = int(n_95)

        print(f"\n  Dims for 90% variance: {n_90}")
        print(f"  Dims for 95% variance: {n_95}")

        # Check if PC1 is just norm
        pc1 = pca.components_[0]
        norms = np.linalg.norm(X, axis=1)
        projections = X @ pc1
        correlation = np.corrcoef(norms, projections)[0, 1]

        self.results.pc1_norm_correlation = float(correlation)

        print(f"\n  PC1 correlation with norm: {correlation:.3f}")
        if abs(correlation) > 0.9:
            print("  → PC1 is primarily token norm (not semantic)")
        elif abs(correlation) > 0.5:
            print("  → PC1 partially correlates with norm")
        else:
            print("  → PC1 encodes semantic structure, not norm")

        # Check mean direction
        mean_dir = X.mean(axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        pc1_mean_cosine = np.abs(np.dot(pc1, mean_dir))
        print(f"  PC1 alignment with mean: {pc1_mean_cosine:.3f}")

        return pca

    # ==========================================
    # 2. Token Category Analysis
    # ==========================================

    def analyze_token_categories(self):
        """Do token categories cluster in embedding space?"""
        print("\n" + "=" * 60)
        print("TOKEN CATEGORY CLUSTERING IN RAW EMBEDDINGS")
        print("=" * 60)

        # Keywords for each category
        keywords = {
            "question": ["what", "where", "when", "how", "why", "who", "which"],
            "action": ["send", "get", "create", "delete", "find", "search", "set", "make", "book"],
            "location": ["tokyo", "london", "paris", "york", "airport", "hotel", "restaurant"],
            "time": ["today", "tomorrow", "yesterday", "morning", "evening", "monday"],
            "number": ["one", "two", "three", "four", "five", "six", "seven"],
            "pronoun": ["i", "you", "he", "she", "it", "we", "they", "my", "your"],
        }

        # Get embeddings for each category
        category_embeddings = {}

        for category, words in keywords.items():
            embeddings_list = []
            found_words = []

            for word in words:
                token_ids = self._get_token_ids(word)
                if token_ids:
                    # Use first token
                    tid = token_ids[0]
                    if tid < self.vocab_size:
                        embeddings_list.append(self.embeddings[tid])
                        found_words.append(word)

            if embeddings_list:
                category_embeddings[category] = {
                    "embeddings": np.array(embeddings_list),
                    "centroid": np.mean(embeddings_list, axis=0),
                    "words": found_words,
                    "n_tokens": len(embeddings_list),
                }

        # Print found tokens
        print("\nTokens found per category:")
        for category, data in category_embeddings.items():
            print(
                f"  {category:<12}: {data['n_tokens']:2d} tokens ({', '.join(data['words'][:5])}...)"
            )

        # Compute category similarity matrix
        cats = list(category_embeddings.keys())

        print("\nCategory Centroid Similarity:")
        print(" " * 14, end="")
        for c in cats:
            print(f"{c[:8]:>9}", end="")
        print()

        for c1 in cats:
            print(f"{c1:<12}", end="  ")
            v1 = category_embeddings[c1]["centroid"]
            n1 = np.linalg.norm(v1)

            for c2 in cats:
                v2 = category_embeddings[c2]["centroid"]
                n2 = np.linalg.norm(v2)

                if n1 > 0 and n2 > 0:
                    sim = np.dot(v1, v2) / (n1 * n2)
                else:
                    sim = 0

                if c1 == c2:
                    print(f"  {sim:.2f}*", end="")
                elif sim > 0.8:
                    print(f"  {sim:.2f}!", end="")
                else:
                    print(f"  {sim:.2f} ", end="")
            print()

        # Within vs between category similarity
        within_sims = []
        between_sims = []

        for c1 in cats:
            embs1 = category_embeddings[c1]["embeddings"]

            # Within-category
            for i in range(len(embs1)):
                for j in range(i + 1, len(embs1)):
                    n1 = np.linalg.norm(embs1[i])
                    n2 = np.linalg.norm(embs1[j])
                    if n1 > 0 and n2 > 0:
                        sim = np.dot(embs1[i], embs1[j]) / (n1 * n2)
                        within_sims.append(sim)

            # Between-category
            for c2 in cats:
                if c1 != c2:
                    embs2 = category_embeddings[c2]["embeddings"]
                    for e1 in embs1:
                        for e2 in embs2:
                            n1 = np.linalg.norm(e1)
                            n2 = np.linalg.norm(e2)
                            if n1 > 0 and n2 > 0:
                                sim = np.dot(e1, e2) / (n1 * n2)
                                between_sims.append(sim)

        within_mean = np.mean(within_sims) if within_sims else 0
        between_mean = np.mean(between_sims) if between_sims else 0
        separation = within_mean - between_mean

        self.results.within_category_similarity = float(within_mean)
        self.results.between_category_similarity = float(between_mean)
        self.results.category_separation = float(separation)

        print(f"\nWithin-category similarity:  {within_mean:.3f}")
        print(f"Between-category similarity: {between_mean:.3f}")
        print(f"Separation: {separation:.3f}")

        if separation > 0.1:
            print("→ Semantic categories ARE clustered in raw embeddings")
        elif separation > 0.05:
            print("→ Weak category clustering in embeddings")
        else:
            print("→ Categories NOT well separated in embeddings")

        return category_embeddings

    # ==========================================
    # 3. Nearest Neighbor Analysis
    # ==========================================

    def find_nearest_neighbors(self, word: str, k: int = 10):
        """What tokens are closest in embedding space?"""
        token_ids = self._get_token_ids(word)
        if not token_ids:
            print(f"Word '{word}' not found in vocabulary")
            return []

        target_emb = self.embeddings[token_ids[0]]
        target_norm = np.linalg.norm(target_emb)

        if target_norm == 0:
            print(f"Word '{word}' has zero-norm embedding")
            return []

        # Compute similarities to all tokens
        norms = np.linalg.norm(self.embeddings, axis=1)
        valid_mask = norms > 0

        sims = np.zeros(self.vocab_size)
        sims[valid_mask] = np.dot(self.embeddings[valid_mask], target_emb) / (
            norms[valid_mask] * target_norm
        )

        # Top k (excluding self)
        sims[token_ids[0]] = -1  # Exclude self
        top_k = np.argsort(sims)[-k:][::-1]

        print(f"\nNearest neighbors to '{word}':")
        results = []
        for tid in top_k:
            try:
                token_str = self.tokenizer.decode([tid])
                print(f"  {sims[tid]:.3f}: '{token_str}'")
                results.append((tid, sims[tid], token_str))
            except:
                pass

        return results

    # ==========================================
    # 4. Feature Probes on Embeddings
    # ==========================================

    def probe_embeddings(self, feature_name: str, positive_words: list, negative_words: list):
        """Can we linearly separate a feature in raw embedding space?"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        pos_embeddings = []
        neg_embeddings = []

        for word in positive_words:
            token_ids = self._get_token_ids(word)
            for tid in token_ids[:1]:  # First token only
                if tid < self.vocab_size:
                    pos_embeddings.append(self.embeddings[tid])

        for word in negative_words:
            token_ids = self._get_token_ids(word)
            for tid in token_ids[:1]:
                if tid < self.vocab_size:
                    neg_embeddings.append(self.embeddings[tid])

        if len(pos_embeddings) < 3 or len(neg_embeddings) < 3:
            print(f"\n{feature_name}: Not enough tokens found")
            return None

        X = np.vstack(pos_embeddings + neg_embeddings)
        y = np.array([1] * len(pos_embeddings) + [0] * len(neg_embeddings))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validated accuracy
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_folds = min(5, len(y) // 2)
        if cv_folds < 2:
            print(f"\n{feature_name}: Not enough samples for CV")
            return None

        scores = cross_val_score(clf, X_scaled, y, cv=cv_folds)

        # Fit and get direction
        clf.fit(X_scaled, y)
        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)

        # Separation (in scaled space)
        pos_proj = np.mean([np.dot(scaler.transform([e])[0], direction) for e in pos_embeddings])
        neg_proj = np.mean([np.dot(scaler.transform([e])[0], direction) for e in neg_embeddings])
        separation = pos_proj - neg_proj

        result = {
            "accuracy": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "separation": float(separation),
            "n_positive": len(pos_embeddings),
            "n_negative": len(neg_embeddings),
        }

        self.results.probe_results[feature_name] = result

        print(f"\n{feature_name}:")
        print(f"  Tokens: {len(pos_embeddings)} pos, {len(neg_embeddings)} neg")
        print(f"  Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"  Separation: {separation:.3f}")

        if scores.mean() > 0.8:
            print("  → Feature IS linearly separable in embeddings")
        elif scores.mean() > 0.6:
            print("  → Feature is WEAKLY separable")
        else:
            print("  → Feature NOT separable in embeddings")

        return result

    # ==========================================
    # 5. Tool-Relevant Token Clustering
    # ==========================================

    def analyze_tool_tokens(self):
        """Are tool-relevant tokens clustered in embedding space?"""
        print("\n" + "=" * 60)
        print("TOOL-RELEVANT TOKEN CLUSTERING")
        print("=" * 60)

        tool_categories = {
            "weather": [
                "weather",
                "temperature",
                "rain",
                "sunny",
                "forecast",
                "cloudy",
                "wind",
                "cold",
                "hot",
            ],
            "email": ["email", "send", "message", "inbox", "compose", "reply", "forward", "mail"],
            "calendar": [
                "calendar",
                "schedule",
                "meeting",
                "event",
                "appointment",
                "reminder",
                "book",
            ],
            "search": ["search", "find", "look", "query", "browse", "discover", "locate"],
            "general": ["the", "a", "is", "are", "and", "or", "but", "for", "to", "of"],
        }

        category_embeddings = {}

        for category, words in tool_categories.items():
            embs = []
            found_words = []

            for word in words:
                token_ids = self._get_token_ids(word)
                if token_ids and token_ids[0] < self.vocab_size:
                    embs.append(self.embeddings[token_ids[0]])
                    found_words.append(word)

            if embs:
                category_embeddings[category] = {"embeddings": np.array(embs), "words": found_words}
                print(f"  {category:<12}: {len(embs)} tokens")

        # Classification test
        if len(category_embeddings) < 2:
            print("\nNot enough categories with tokens")
            return

        all_embs = []
        all_labels = []

        for category, data in category_embeddings.items():
            for emb in data["embeddings"]:
                all_embs.append(emb)
                all_labels.append(category)

        X = np.vstack(all_embs)

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        le = LabelEncoder()
        y = le.fit_transform(all_labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv_folds = min(3, len(set(y)))
        scores = cross_val_score(clf, X_scaled, y, cv=cv_folds)

        self.results.tool_category_accuracy = float(scores.mean())

        chance = 1 / len(category_embeddings)
        print("\nTool category classification:")
        print(f"  Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"  Chance baseline: {chance:.2f}")

        if scores.mean() > chance + 0.2:
            print("→ Tool categories ARE separable in raw embeddings")
        elif scores.mean() > chance + 0.1:
            print("→ Tool categories are WEAKLY separable")
        else:
            print("→ Tool categories NOT clearly separable")

        # PCA visualization
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)

        print("\nCategory centroids in PCA space:")
        for category in tool_categories.keys():
            if category not in [le.inverse_transform([i])[0] for i in range(len(le.classes_))]:
                continue
            mask = [l == category for l in all_labels]
            if sum(mask) > 0:
                points = X_2d[mask]
                centroid = points.mean(axis=0)
                print(f"  {category:<12}: ({centroid[0]:+.2f}, {centroid[1]:+.2f})")

    # ==========================================
    # 6. Semantic Analogy Test
    # ==========================================

    def test_analogies(self):
        """Do embeddings capture semantic relationships?"""
        print("\n" + "=" * 60)
        print("SEMANTIC ANALOGY TEST")
        print("=" * 60)

        analogies = [
            ("king", "queen", "man", "woman"),
            ("tokyo", "japan", "paris", "france"),
            ("send", "sent", "get", "got"),
            ("big", "bigger", "small", "smaller"),
            ("hot", "cold", "up", "down"),
        ]

        correct = 0
        total = 0

        for a, b, c, expected_d in analogies:
            try:
                # Get embeddings
                ids_a = self._get_token_ids(a)
                ids_b = self._get_token_ids(b)
                ids_c = self._get_token_ids(c)
                ids_d = self._get_token_ids(expected_d)

                if not (ids_a and ids_b and ids_c and ids_d):
                    print(f"\n{a} : {b} :: {c} : ? → Missing tokens")
                    continue

                emb_a = self.embeddings[ids_a[0]]
                emb_b = self.embeddings[ids_b[0]]
                emb_c = self.embeddings[ids_c[0]]

                # Compute: B - A + C ≈ D
                emb_d_predicted = emb_b - emb_a + emb_c
                pred_norm = np.linalg.norm(emb_d_predicted)

                if pred_norm == 0:
                    continue

                # Find nearest token
                norms = np.linalg.norm(self.embeddings, axis=1)
                valid_mask = norms > 0

                sims = np.zeros(self.vocab_size)
                sims[valid_mask] = np.dot(self.embeddings[valid_mask], emb_d_predicted) / (
                    norms[valid_mask] * pred_norm
                )

                # Exclude a, b, c from results
                for ids in [ids_a, ids_b, ids_c]:
                    if ids:
                        sims[ids[0]] = -1

                top_5_ids = np.argsort(sims)[-5:][::-1]
                top_words = []
                for tid in top_5_ids:
                    try:
                        top_words.append(self.tokenizer.decode([tid]).strip().lower())
                    except:
                        pass

                total += 1
                is_correct = expected_d.lower() in top_words
                if is_correct:
                    correct += 1

                print(f"\n{a} : {b} :: {c} : ?")
                print(f"  Expected: {expected_d}")
                print(f"  Top 5: {top_words[:5]}")
                print(f"  {'✓ CORRECT' if is_correct else '✗ Wrong'}")

            except Exception as e:
                print(f"\n{a} : {b} :: {c} : ? → Error: {e}")

        if total > 0:
            print(f"\nAnalogy accuracy: {correct}/{total} = {correct / total:.0%}")

    # ==========================================
    # 7. Compare Embeddings vs L0
    # ==========================================

    def compare_embedding_vs_l0(self, prompts: list[str]):
        """Compare raw embeddings to L0 activations."""
        print("\n" + "=" * 60)
        print("EMBEDDINGS vs L0 ACTIVATIONS")
        print("=" * 60)

        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        # Get both raw embeddings and L0 activations for same tokens
        embedding_acts = []
        l0_acts = []

        for prompt in prompts[:10]:
            # Get tokens
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            elif hasattr(tokens, "tolist"):
                tokens = tokens.tolist()

            # Raw embeddings
            for tid in tokens:
                if tid < self.vocab_size:
                    embedding_acts.append(self.embeddings[tid])

            # L0 activations
            hooks = ModelHooks(self.model)
            hooks.configure(
                CaptureConfig(
                    layers=[0],
                    capture_hidden_states=True,
                    positions=PositionSelection.ALL,
                )
            )

            input_ids = mx.array([tokens])
            hooks.forward(input_ids)

            if 0 in hooks.state.hidden_states:
                h = hooks.state.hidden_states[0]
                h_f32 = h.astype(mx.float32)
                h_np = np.array(h_f32, copy=False)
                if h_np.ndim == 3:
                    h_np = h_np[0]
                for i in range(min(len(tokens), h_np.shape[0])):
                    l0_acts.append(h_np[i])

        if not embedding_acts or not l0_acts:
            print("Could not collect activations")
            return

        # Truncate to same length
        n = min(len(embedding_acts), len(l0_acts))
        embedding_acts = np.array(embedding_acts[:n])
        l0_acts = np.array(l0_acts[:n])

        print(f"\nComparing {n} token activations:")

        # 1. Norm comparison
        emb_norms = np.linalg.norm(embedding_acts, axis=1)
        l0_norms = np.linalg.norm(l0_acts, axis=1)

        print("\n1. Norm statistics:")
        print(f"   Embedding norms: mean={emb_norms.mean():.2f}, std={emb_norms.std():.2f}")
        print(f"   L0 norms:        mean={l0_norms.mean():.2f}, std={l0_norms.std():.2f}")
        print(f"   Ratio L0/Emb:    {(l0_norms / emb_norms).mean():.2f}")

        # 2. Cosine similarity
        cosines = []
        for i in range(n):
            n1 = np.linalg.norm(embedding_acts[i])
            n2 = np.linalg.norm(l0_acts[i])
            if n1 > 0 and n2 > 0:
                cos = np.dot(embedding_acts[i], l0_acts[i]) / (n1 * n2)
                cosines.append(cos)

        print("\n2. Cosine similarity (Emb vs L0):")
        print(f"   Mean: {np.mean(cosines):.3f}")
        print(f"   Std:  {np.std(cosines):.3f}")
        print(f"   Min:  {np.min(cosines):.3f}")
        print(f"   Max:  {np.max(cosines):.3f}")

        if np.mean(cosines) > 0.95:
            print("   → L0 ≈ scaled embeddings (LayerNorm only)")
        elif np.mean(cosines) > 0.8:
            print("   → L0 is similar to embeddings but transformed")
        else:
            print("   → L0 is significantly different from embeddings")

        # 3. Relative change
        deltas = l0_acts - embedding_acts
        delta_norms = np.linalg.norm(deltas, axis=1)
        relative_change = delta_norms / emb_norms

        print("\n3. Relative change (Emb → L0):")
        print(f"   Mean: {relative_change.mean():.3f}")
        print("   This represents the effect of any pre-layer normalization")

    # ==========================================
    # Full Analysis
    # ==========================================

    def run_full_analysis(self, output_dir: str = "embedding_analysis"):
        """Run all embedding analyses"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("=" * 60)
        print("RAW EMBEDDING ANALYSIS")
        print("Proving what's in embeddings BEFORE computation")
        print("=" * 60)
        print(f"\nModel: {self.model_id}")
        print(f"Vocabulary: {self.vocab_size} tokens")
        print(f"Dimensions: {self.hidden_dim}")

        # 1. Structure
        self.analyze_embedding_structure()

        # 2. Categories
        self.analyze_token_categories()

        # 3. Tool tokens
        self.analyze_tool_tokens()

        # 4. Nearest neighbors
        print("\n" + "=" * 60)
        print("NEAREST NEIGHBOR EXAMPLES")
        print("=" * 60)

        test_words = ["weather", "send", "tokyo", "tomorrow", "what", "search"]
        for word in test_words:
            self.find_nearest_neighbors(word, k=5)

        # 5. Feature probes
        print("\n" + "=" * 60)
        print("FEATURE PROBES ON RAW EMBEDDINGS")
        print("=" * 60)

        probes = {
            "question_vs_statement": (
                ["what", "where", "when", "how", "why", "who", "which"],
                ["the", "a", "is", "send", "get", "make", "and", "or"],
            ),
            "action_vs_state": (
                ["send", "get", "create", "delete", "find", "make", "set", "book"],
                ["is", "are", "was", "has", "have", "been", "were", "had"],
            ),
            "location_vs_common": (
                ["tokyo", "london", "paris", "airport", "hotel", "restaurant"],
                ["the", "a", "is", "and", "but", "or", "for", "to"],
            ),
            "tool_indicator": (
                ["weather", "email", "calendar", "search", "timer", "alarm"],
                ["think", "explain", "understand", "believe", "know", "feel"],
            ),
        }

        for feature_name, (pos, neg) in probes.items():
            self.probe_embeddings(feature_name, pos, neg)

        # 6. Analogies
        self.test_analogies()

        # 7. Embedding vs L0 comparison
        test_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "The capital of France is Paris",
        ]
        self.compare_embedding_vs_l0(test_prompts)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\n1. Embedding Structure:")
        print(f"   PC1 explains {self.results.pc1_variance * 100:.1f}% variance")
        print(f"   PC1 correlation with norm: {self.results.pc1_norm_correlation:.3f}")
        print(f"   Dims for 90% variance: {self.results.dims_for_90_variance}")

        print("\n2. Category Clustering:")
        print(f"   Within-category:  {self.results.within_category_similarity:.3f}")
        print(f"   Between-category: {self.results.between_category_similarity:.3f}")
        print(f"   Separation:       {self.results.category_separation:.3f}")

        print(f"\n3. Tool Category Accuracy: {self.results.tool_category_accuracy:.1%}")

        print("\n4. Feature Probe Accuracy:")
        for name, result in self.results.probe_results.items():
            print(f"   {name}: {result['accuracy']:.1%}")

        # Verdict
        print("\n" + "=" * 60)
        print("VERDICT: What's in Raw Embeddings?")
        print("=" * 60)

        has_clustering = self.results.category_separation > 0.05
        has_tool_sep = self.results.tool_category_accuracy > 0.4
        has_features = any(r["accuracy"] > 0.7 for r in self.results.probe_results.values())

        if has_clustering and has_tool_sep and has_features:
            print("""
✓ Semantic clustering IS present in raw embeddings
✓ Tool-relevant categories ARE separable
✓ Linguistic features ARE linearly decodable

CONCLUSION: The embedding matrix contains rich semantic structure
            BEFORE any transformer computation.
            L0 probe accuracy reflects embedding content.
""")
        else:
            print("""
The embedding matrix shows:
  Clustering: {"YES" if has_clustering else "NO"}
  Tool separation: {"YES" if has_tool_sep else "NO"}
  Feature probes: {"YES" if has_features else "NO"}

Further investigation needed.
""")

        # Save results
        results_path = output_path / "embedding_results.json"
        self.results.save(str(results_path))

        return self.results


def main():
    MODEL_ID = "mlx-community/functiongemma-270m-it-bf16"

    print("Loading model...")
    analyzer = EmbeddingAnalyzer.from_pretrained(MODEL_ID)

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
