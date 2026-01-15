#!/usr/bin/env python3
"""
early_layer_analysis.py

What's actually encoded in L0-L3?
Since probes show high accuracy at L0, the embeddings are richer than expected.
This script unpacks what's already there before computation begins.

Run: uv run python examples/introspection/early_layer_analysis.py
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class TokenAnalysis:
    """Analysis of a single token's representation"""

    token_id: int
    token_str: str
    position: int
    layer: int
    activation: np.ndarray
    token_type: str  # word, punct, number, special, subword


class EarlyLayerAnalyzer:
    """
    Analyze what's encoded in L0-L3

    Focus on:
    1. Token-level properties (not sequence-level)
    2. Position encoding
    3. Token type clustering
    4. Semantic similarity at embedding level
    5. What the residual stream contains before any computation
    """

    def __init__(self, model: Any, tokenizer: Any, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.results = {}

        # Detect model structure
        self._detect_structure()

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

        # Get hidden size
        if hasattr(self._backbone, "hidden_size"):
            self.hidden_size = self._backbone.hidden_size
        elif hasattr(self.model, "args") and hasattr(self.model.args, "hidden_size"):
            self.hidden_size = self.model.args.hidden_size
        else:
            self.hidden_size = 768  # Fallback

    @classmethod
    def from_pretrained(cls, model_id: str) -> "EarlyLayerAnalyzer":
        """Load a model for analysis."""
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    # ==========================================
    # Token Type Classification
    # ==========================================

    def classify_token(self, token_str: str) -> str:
        """Classify a token into types"""

        # Clean up token representation
        clean = token_str.replace("▁", "").replace("Ġ", "").strip()

        if not clean:
            return "whitespace"

        # Special tokens
        if token_str.startswith("<") and token_str.endswith(">"):
            return "special"
        if token_str in ["<bos>", "<eos>", "<pad>", "<unk>", "<s>", "</s>"]:
            return "special"

        # Punctuation
        if all(c in ".,!?;:()[]{}\"'-–—…" for c in clean):
            return "punctuation"

        # Numbers
        if clean.isdigit() or re.match(r"^[\d.,]+$", clean):
            return "number"

        # Subword (starts with ## or lowercase continuation)
        if token_str.startswith("##") or (
            not token_str.startswith("▁")
            and not token_str.startswith("Ġ")
            and len(clean) > 0
            and not clean[0].isupper()
            and len(clean) < 4
        ):
            return "subword"

        # Regular word
        if clean.isalpha():
            return "word"

        # Mixed
        return "mixed"

    # ==========================================
    # Activation Collection
    # ==========================================

    def get_all_token_activations(
        self, prompt: str, layers: list[int] | None = None
    ) -> dict[int, list[TokenAnalysis]]:
        """Get activations for every token at specified layers"""

        if layers is None:
            layers = [0, 1, 2, 3]

        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        elif hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # Setup hooks
        hooks = ModelHooks(self.model)
        hooks.configure(
            CaptureConfig(
                layers=layers,
                capture_hidden_states=True,
                positions=PositionSelection.ALL,
            )
        )

        # Forward pass
        input_ids = mx.array([tokens])
        hooks.forward(input_ids)

        results = {layer: [] for layer in layers}

        for layer in layers:
            if layer not in hooks.state.hidden_states:
                continue

            h = hooks.state.hidden_states[layer]
            # Cast to float32 for numpy conversion
            h_f32 = h.astype(mx.float32)
            h_np = np.array(h_f32, copy=False)

            # h_np shape is [batch, seq, hidden]
            if h_np.ndim == 3:
                h_np = h_np[0]  # Remove batch dim

            for pos, (token_id, token_str) in enumerate(zip(tokens, token_strs)):
                if pos < h_np.shape[0]:
                    activation = h_np[pos]
                    token_type = self.classify_token(token_str)

                    results[layer].append(
                        TokenAnalysis(
                            token_id=token_id,
                            token_str=token_str,
                            position=pos,
                            layer=layer,
                            activation=activation,
                            token_type=token_type,
                        )
                    )

        return results

    # ==========================================
    # Position Encoding Analysis
    # ==========================================

    def analyze_position_encoding(self, prompts: list[str], layer: int = 0):
        """How is position encoded in activations?"""

        print(f"\n{'=' * 60}")
        print(f"POSITION ENCODING ANALYSIS - Layer {layer}")
        print("=" * 60)

        position_activations = defaultdict(list)

        for prompt in prompts:
            token_data = self.get_all_token_activations(prompt, [layer])
            for ta in token_data[layer]:
                if ta.position < 20:  # First 20 positions
                    position_activations[ta.position].append(ta.activation)

        # Average activation per position
        position_means = {}
        for pos, acts in position_activations.items():
            position_means[pos] = np.mean(acts, axis=0)

        # Position similarity matrix
        positions = sorted(position_means.keys())
        n_pos = len(positions)
        sim_matrix = np.zeros((n_pos, n_pos))

        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                v1 = position_means[p1]
                v2 = position_means[p2]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                else:
                    sim = 0
                sim_matrix[i, j] = sim

        print("\nPosition Similarity Matrix (cosine):")
        print("      ", end="")
        for p in positions[:10]:
            print(f"  P{p:<3}", end="")
        print()

        for i, p1 in enumerate(positions[:10]):
            print(f"P{p1:<4}", end=" ")
            for j, p2 in enumerate(positions[:10]):
                if i < len(sim_matrix) and j < len(sim_matrix[i]):
                    sim = sim_matrix[i, j]
                    if sim > 0.95:
                        print(f" {sim:.2f}*", end="")
                    else:
                        print(f" {sim:.2f} ", end="")
            print()

        # Check if position is linearly decodable
        print("\nPosition as linear direction:")

        # Collect position-activation pairs
        all_acts = []
        all_pos = []
        for pos, acts in position_activations.items():
            for act in acts:
                all_acts.append(act)
                all_pos.append(pos)

        if len(all_acts) < 10:
            print("  Not enough samples for regression")
            return {"r2_score": 0}

        X = np.array(all_acts)
        y = np.array(all_pos)

        # Linear regression to predict position
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        reg = Ridge()
        scores = cross_val_score(reg, X, y, cv=min(5, len(X) // 2), scoring="r2")
        print(f"  R² for position prediction: {scores.mean():.3f} ± {scores.std():.3f}")

        # Fit and get direction
        reg.fit(X, y)
        position_direction = reg.coef_ / np.linalg.norm(reg.coef_)

        print(f"  Position direction magnitude: {np.linalg.norm(reg.coef_):.3f}")

        if scores.mean() > 0.8:
            print("  → Position is STRONGLY encoded as a linear direction")
        elif scores.mean() > 0.5:
            print("  → Position is MODERATELY encoded")
        else:
            print("  → Position encoding is WEAK or nonlinear")

        return {
            "similarity_matrix": sim_matrix,
            "position_direction": position_direction,
            "r2_score": scores.mean(),
        }

    # ==========================================
    # Token Type Clustering
    # ==========================================

    def analyze_token_types(self, prompts: list[str], layer: int = 0):
        """How do different token types cluster?"""

        print(f"\n{'=' * 60}")
        print(f"TOKEN TYPE CLUSTERING - Layer {layer}")
        print("=" * 60)

        type_activations = defaultdict(list)
        type_tokens = defaultdict(list)

        for prompt in prompts:
            token_data = self.get_all_token_activations(prompt, [layer])
            for ta in token_data[layer]:
                type_activations[ta.token_type].append(ta.activation)
                type_tokens[ta.token_type].append(ta.token_str)

        print("\nToken counts by type:")
        for token_type in sorted(type_activations.keys()):
            count = len(type_activations[token_type])
            examples = list(set(type_tokens[token_type]))[:5]
            print(f"  {token_type:<12}: {count:4d} tokens  (e.g., {examples})")

        # Compute type centroids
        type_centroids = {}
        for token_type, acts in type_activations.items():
            type_centroids[token_type] = np.mean(acts, axis=0)

        # Type similarity matrix
        types = sorted(type_centroids.keys())
        print("\nToken Type Similarity (cosine):")
        print("             ", end="")
        for t in types:
            print(f" {t[:6]:<7}", end="")
        print()

        for t1 in types:
            print(f"{t1:<12}", end=" ")
            for t2 in types:
                v1 = type_centroids[t1]
                v2 = type_centroids[t2]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                else:
                    sim = 0
                if t1 == t2:
                    print(f" {sim:.2f}*", end=" ")
                elif sim > 0.9:
                    print(f" {sim:.2f}!", end=" ")
                else:
                    print(f" {sim:.2f} ", end=" ")
            print()

        # Probe: can we predict token type from activation?
        print("\nToken type linear separability:")

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        all_acts = []
        all_types = []
        for token_type, acts in type_activations.items():
            for act in acts:
                all_acts.append(act)
                all_types.append(token_type)

        if len(all_acts) < 10:
            print("  Not enough samples")
            return {"classification_accuracy": 0}

        X = np.array(all_acts)
        le = LabelEncoder()
        y = le.fit_transform(all_types)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, len(set(y))))
        print(f"  Classification accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Chance baseline: {1 / len(types):.3f}")

        if scores.mean() > 0.8:
            print("  → Token types ARE clearly separable")
        elif scores.mean() > 0.5:
            print("  → Token types are MODERATELY separable")
        else:
            print("  → Token types are NOT well separable")

        return {"type_centroids": type_centroids, "classification_accuracy": scores.mean()}

    # ==========================================
    # Semantic Content at L0
    # ==========================================

    def analyze_semantic_similarity(self, layer: int = 0):
        """Is semantic similarity present in early layers?"""

        print(f"\n{'=' * 60}")
        print(f"SEMANTIC SIMILARITY - Layer {layer}")
        print("=" * 60)

        # Semantic groups to test
        semantic_groups = {
            "weather": ["weather", "rain", "temperature", "sunny", "forecast", "cloudy"],
            "communication": ["email", "send", "message", "write", "reply", "contact"],
            "time": ["tomorrow", "today", "yesterday", "morning", "evening", "schedule"],
            "locations": ["Tokyo", "London", "Paris", "airport", "restaurant", "hotel"],
            "actions": ["get", "find", "search", "create", "delete", "update"],
            "questions": ["what", "where", "when", "how", "why", "who"],
        }

        group_centroids = {}

        for group_name, words in semantic_groups.items():
            activations = []
            found_words = []

            for word in words:
                prompt = f"The word is {word}"
                token_data = self.get_all_token_activations(prompt, [layer])

                # Find the target word's activation
                for ta in token_data[layer]:
                    if word.lower() in ta.token_str.lower():
                        activations.append(ta.activation)
                        found_words.append(word)
                        break

            if activations:
                group_centroids[group_name] = {
                    "centroid": np.mean(activations, axis=0),
                    "words": found_words,
                    "activations": activations,
                }

        # Group similarity matrix
        groups = sorted(group_centroids.keys())
        print("\nSemantic Group Similarity (cosine):")
        print("             ", end="")
        for g in groups:
            print(f" {g[:8]:<9}", end="")
        print()

        for g1 in groups:
            print(f"{g1:<12}", end=" ")
            for g2 in groups:
                v1 = group_centroids[g1]["centroid"]
                v2 = group_centroids[g2]["centroid"]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(v1, v2) / (norm1 * norm2)
                else:
                    sim = 0
                if g1 == g2:
                    print(f" {sim:.2f}* ", end=" ")
                elif sim > 0.8:
                    print(f" {sim:.2f}! ", end=" ")
                else:
                    print(f" {sim:.2f}  ", end=" ")
            print()

        # Within-group vs between-group similarity
        print("\nWithin-group vs between-group similarity:")

        within_sims = []
        between_sims = []

        for g1 in groups:
            acts1 = group_centroids[g1]["activations"]
            for i, a1 in enumerate(acts1):
                for j, a2 in enumerate(acts1):
                    if i < j:
                        norm1 = np.linalg.norm(a1)
                        norm2 = np.linalg.norm(a2)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(a1, a2) / (norm1 * norm2)
                            within_sims.append(sim)

                for g2 in groups:
                    if g1 != g2:
                        acts2 = group_centroids[g2]["activations"]
                        for a2 in acts2:
                            norm1 = np.linalg.norm(a1)
                            norm2 = np.linalg.norm(a2)
                            if norm1 > 0 and norm2 > 0:
                                sim = np.dot(a1, a2) / (norm1 * norm2)
                                between_sims.append(sim)

        within_mean = np.mean(within_sims) if within_sims else 0
        between_mean = np.mean(between_sims) if between_sims else 0

        print(f"  Within-group mean similarity:  {within_mean:.3f}")
        print(f"  Between-group mean similarity: {between_mean:.3f}")
        print(f"  Separation: {within_mean - between_mean:.3f}")

        if within_mean > between_mean + 0.05:
            print("  → Semantic clustering IS present at this layer")
        else:
            print("  → Semantic clustering is WEAK at this layer")

        return {
            "group_centroids": {k: v["centroid"] for k, v in group_centroids.items()},
            "within_similarity": within_mean,
            "between_similarity": between_mean,
        }

    # ==========================================
    # PCA of Early Layers
    # ==========================================

    def analyze_principal_directions(self, prompts: list[str], layers: list[int] | None = None):
        """What are the dominant directions in each layer?"""

        if layers is None:
            layers = [0, 1, 2, 3]

        print(f"\n{'=' * 60}")
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 60)

        from sklearn.decomposition import PCA

        for layer in layers:
            all_activations = []
            all_positions = []

            for prompt in prompts:
                token_data = self.get_all_token_activations(prompt, [layer])
                for ta in token_data[layer]:
                    all_activations.append(ta.activation)
                    all_positions.append(ta.position)

            if len(all_activations) < 20:
                print(f"\nLayer {layer}: Not enough samples")
                continue

            X = np.array(all_activations)

            n_components = min(20, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            pca.fit(X)

            print(f"\nLayer {layer}:")
            print("  Variance explained by top components:")
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            for i in [0, 1, 2, 4, 9, 19]:
                if i < len(pca.explained_variance_ratio_):
                    print(
                        f"    PC{i + 1}: {pca.explained_variance_ratio_[i] * 100:.1f}% (cumulative: {cumvar[i] * 100:.1f}%)"
                    )

            # How many components for 90% variance?
            n_90 = np.searchsorted(cumvar, 0.9) + 1
            n_95 = np.searchsorted(cumvar, 0.95) + 1
            print(f"  Components for 90% variance: {n_90}")
            print(f"  Components for 95% variance: {n_95}")

            # Check what top PCs correlate with
            projections = pca.transform(X)
            positions = np.array(all_positions)

            print("  Top PC correlations with position:")
            for i in range(min(5, projections.shape[1])):
                corr = np.corrcoef(projections[:, i], positions)[0, 1]
                if abs(corr) > 0.3:
                    print(f"    PC{i + 1}: r={corr:.3f} ***")
                else:
                    print(f"    PC{i + 1}: r={corr:.3f}")

    # ==========================================
    # Layer-by-Layer Comparison
    # ==========================================

    def analyze_layer_evolution(self, prompts: list[str], layers: list[int] | None = None):
        """How do representations change from L0 to L3?"""

        if layers is None:
            layers = [0, 1, 2, 3]

        print(f"\n{'=' * 60}")
        print("LAYER EVOLUTION ANALYSIS")
        print("=" * 60)

        # Collect same-token activations across layers
        token_evolution = defaultdict(lambda: {})

        for prompt in prompts[:20]:  # Limit for speed
            token_data = self.get_all_token_activations(prompt, layers)

            for layer in layers:
                for ta in token_data[layer]:
                    key = (prompt, ta.position)
                    token_evolution[key][layer] = ta.activation

        # Compute layer-to-layer similarity
        print("\nLayer-to-layer similarity (same tokens):")

        for l1 in layers:
            print(f"  L{l1}: ", end="")
            for l2 in layers:
                sims = []
                for key, layer_acts in token_evolution.items():
                    if l1 in layer_acts and l2 in layer_acts:
                        v1 = layer_acts[l1]
                        v2 = layer_acts[l2]
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(v1, v2) / (norm1 * norm2)
                            sims.append(sim)

                if sims:
                    mean_sim = np.mean(sims)
                    if l1 == l2:
                        print("  1.00*", end="")
                    else:
                        print(f"  {mean_sim:.2f} ", end="")
                else:
                    print("  -   ", end="")
            print()

        # How much does each layer change from previous?
        print("\nRepresentation change per layer:")
        for i, layer in enumerate(layers[1:], 1):
            prev_layer = layers[i - 1]
            changes = []
            for key, layer_acts in token_evolution.items():
                if prev_layer in layer_acts and layer in layer_acts:
                    v1 = layer_acts[prev_layer]
                    v2 = layer_acts[layer]
                    norm1 = np.linalg.norm(v1)
                    if norm1 > 0:
                        change = np.linalg.norm(v2 - v1) / norm1
                        changes.append(change)

            if changes:
                print(f"  L{prev_layer} → L{layer}: {np.mean(changes):.3f} relative change")

    # ==========================================
    # Tool-Relevant Feature Detection
    # ==========================================

    def analyze_tool_features_early(self, layer: int = 0):
        """Are tool-relevant features present at L0?"""

        print(f"\n{'=' * 60}")
        print(f"TOOL-RELEVANT FEATURES - Layer {layer}")
        print("=" * 60)

        # Test specific features
        feature_tests = {
            "question_words": {
                "positive": ["What is the weather?", "Where is the restaurant?", "How do I send?"],
                "negative": ["Send the email.", "Create an event.", "Get the temperature."],
            },
            "imperative_verbs": {
                "positive": ["Send an email now.", "Create the event.", "Get the weather."],
                "negative": ["The weather is nice.", "I sent an email.", "The event exists."],
            },
            "location_mentions": {
                "positive": ["Weather in Tokyo", "Fly to Paris", "Restaurant near London"],
                "negative": ["Send an email", "Create a meeting", "Search for news"],
            },
            "api_like_syntax": {
                "positive": ["get_weather()", "send_email to:", "search: query"],
                "negative": ["The weather is nice", "I sent mail", "I searched"],
            },
        }

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        for feature_name, prompts_dict in feature_tests.items():
            pos_prompts = prompts_dict["positive"]
            neg_prompts = prompts_dict["negative"]

            pos_acts = []
            neg_acts = []

            for prompt in pos_prompts:
                token_data = self.get_all_token_activations(prompt, [layer])
                # Use last token representation
                if token_data[layer]:
                    pos_acts.append(token_data[layer][-1].activation)

            for prompt in neg_prompts:
                token_data = self.get_all_token_activations(prompt, [layer])
                if token_data[layer]:
                    neg_acts.append(token_data[layer][-1].activation)

            if len(pos_acts) < 2 or len(neg_acts) < 2:
                print(f"\n{feature_name}: Not enough samples")
                continue

            X = np.vstack(pos_acts + neg_acts)
            y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

            clf = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(clf, X, y, cv=min(3, len(X) // 2))

            # Compute direction
            clf.fit(X, y)
            direction = clf.coef_[0] / np.linalg.norm(clf.coef_[0])

            # Separation
            pos_proj = np.mean([np.dot(a, direction) for a in pos_acts])
            neg_proj = np.mean([np.dot(a, direction) for a in neg_acts])
            separation = pos_proj - neg_proj

            print(f"\n{feature_name}:")
            print(f"  Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
            print(f"  Separation: {separation:.3f}")
            if scores.mean() > 0.7:
                print(f"  → Feature IS detectable at L{layer}")
            else:
                print(f"  → Feature is WEAK at L{layer}")

    # ==========================================
    # Full Analysis
    # ==========================================

    def run_full_analysis(self, output_dir: str = "early_layer_analysis"):
        """Run complete early layer analysis"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("=" * 60)
        print("EARLY LAYER ANALYSIS (L0-L3)")
        print("Understanding what embeddings already encode")
        print("=" * 60)
        print(f"\nModel: {self.model_id}")
        print(f"Layers: {self.num_layers}, Hidden: {self.hidden_size}")

        # Test prompts
        prompts = [
            # Tool-calling
            "What is the weather in Tokyo?",
            "Send an email to John about the meeting",
            "Create a calendar event for tomorrow at 3pm",
            "Search for Italian restaurants nearby",
            "Get the stock price of Apple",
            "Set a timer for 10 minutes",
            # Non-tool
            "The capital of France is Paris",
            "Once upon a time there was a princess",
            "Explain how photosynthesis works",
            "What is the meaning of life?",
            "2 + 2 equals 4",
            "The quick brown fox jumps over the lazy dog",
            # Mixed/ambiguous
            "Tell me about Tokyo",
            "I need help with email",
            "Weather information please",
            "Calculate something for me",
        ]

        # Run analyses
        self.analyze_position_encoding(prompts, layer=0)
        self.analyze_token_types(prompts, layer=0)
        self.analyze_semantic_similarity(layer=0)
        self.analyze_principal_directions(prompts, layers=[0, 1, 2, 3])
        self.analyze_layer_evolution(prompts, layers=[0, 1, 2, 3])
        self.analyze_tool_features_early(layer=0)

        # Compare L0 vs L3
        print(f"\n{'=' * 60}")
        print("L0 vs L3 COMPARISON")
        print("=" * 60)

        print("\nRunning same analyses on L3...")
        self.analyze_token_types(prompts, layer=3)
        self.analyze_semantic_similarity(layer=3)
        self.analyze_tool_features_early(layer=3)

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        print("""
Key questions answered:
1. Is position encoding present at L0? (Check R² score)
2. Do token types cluster? (Check classification accuracy)
3. Is semantic similarity present? (Check within vs between group)
4. How dominant is PC1? (Position vs other structure)
5. Are tool-relevant features detectable? (Check feature accuracies)

If features are strong at L0:
  → Embeddings are doing heavy lifting
  → Later layers combine, not compute
  → Router can work with shallow architecture

If features emerge later:
  → Computation builds representations
  → Need deeper analysis of L4-10
  → Router needs to replicate more computation
        """)


# ==========================================
# Main
# ==========================================


def main():
    MODEL_ID = "mlx-community/functiongemma-270m-it-bf16"

    print("Loading model...")
    analyzer = EarlyLayerAnalyzer.from_pretrained(MODEL_ID)
    print(f"Loaded: {analyzer.num_layers} layers, {analyzer.hidden_size} hidden")

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
