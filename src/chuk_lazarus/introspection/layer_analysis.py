"""
Layer analysis tools for understanding what specific layers do.

This module helps answer questions like:
- What does layer N "see"?
- Do similar inputs cluster together at layer N?
- What attention patterns emerge at layer N?
- What happens if we skip layer N?

Key insight from format sensitivity research:
- L4 in Gemma 3 4B seems to be where format (trailing space) affects computation
- This tool helps understand WHY and WHAT L4's actual function is

Example:
    >>> from chuk_lazarus.introspection.layer_analysis import LayerAnalyzer
    >>>
    >>> analyzer = LayerAnalyzer.from_pretrained("mlx-community/gemma-3-4b-it-bf16")
    >>>
    >>> prompts = [
    ...     "100 - 37 = ",   # working math
    ...     "100 - 37 =",    # broken math
    ...     "50 + 25 = ",    # working math
    ...     "50 + 25 =",     # broken math
    ... ]
    >>>
    >>> result = analyzer.analyze_representations(prompts, layers=[2, 4, 6, 8])
    >>> analyzer.print_similarity_matrix(result, layer=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class RepresentationResult:
    """Result of representation analysis at a layer."""

    layer_idx: int
    """Layer index."""

    prompts: list[str]
    """Prompts analyzed."""

    representations: dict[str, mx.array]
    """Hidden state for each prompt (last position)."""

    similarity_matrix: list[list[float]]
    """Cosine similarity between all prompt pairs."""

    labels: list[str] | None = None
    """Optional labels for prompts (e.g., 'working', 'broken')."""

    def get_similarity(self, prompt1: str, prompt2: str) -> float:
        """Get similarity between two prompts."""
        idx1 = self.prompts.index(prompt1)
        idx2 = self.prompts.index(prompt2)
        return self.similarity_matrix[idx1][idx2]


@dataclass
class AttentionResult:
    """Result of attention pattern analysis."""

    layer_idx: int
    """Layer index."""

    prompt: str
    """The analyzed prompt."""

    tokens: list[str]
    """Tokenized prompt."""

    attention_weights: mx.array
    """Attention weights [num_heads, seq_len, seq_len]."""

    @property
    def num_heads(self) -> int:
        return self.attention_weights.shape[0]

    @property
    def seq_len(self) -> int:
        return self.attention_weights.shape[1]

    def get_head_pattern(self, head_idx: int) -> mx.array:
        """Get attention pattern for a specific head."""
        return self.attention_weights[head_idx]

    def get_attention_to_token(self, token_idx: int, from_position: int = -1) -> mx.array:
        """Get attention weights to a specific token from a position."""
        return self.attention_weights[:, from_position, token_idx]


@dataclass
class ClusterResult:
    """Result of clustering analysis."""

    layer_idx: int
    """Layer index."""

    labels: list[str]
    """Labels for each prompt."""

    within_cluster_similarity: dict[str, float]
    """Average similarity within each label group."""

    between_cluster_similarity: dict[tuple[str, str], float]
    """Average similarity between label groups."""

    separation_score: float
    """How well clusters separate (within - between)."""


@dataclass
class LayerAnalysisResult:
    """Complete analysis result for multiple layers."""

    prompts: list[str]
    """Prompts analyzed."""

    labels: list[str] | None
    """Optional labels for prompts."""

    layers: list[int]
    """Layers analyzed."""

    representations: dict[int, RepresentationResult]
    """Per-layer representation results."""

    attention: dict[int, dict[str, AttentionResult]] | None = None
    """Per-layer, per-prompt attention results."""

    clusters: dict[int, ClusterResult] | None = None
    """Per-layer clustering results (if labels provided)."""


class LayerAnalyzer:
    """
    Analyzer for understanding what specific layers do.

    Helps answer:
    - Do working/broken prompts cluster separately?
    - What attention patterns emerge?
    - Where do features emerge?
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
        config: Any | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._config = config

    @classmethod
    def from_pretrained(cls, model_id: str) -> LayerAnalyzer:
        """Load model and create analyzer."""
        import json

        from ..inference.loader import DType, HFLoader
        from ..models_v2.families.registry import detect_model_family, get_family_info

        # Download model
        result = HFLoader.download(model_id)
        model_path = result.model_path

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        # Detect family
        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError("Unsupported model family")

        family_info = get_family_info(family_type)
        config_class = family_info.config_class
        model_class = family_info.model_class

        config = config_class.from_hf_config(config_data)
        model = model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"Loaded {model_id}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden size: {config.hidden_size}")

        return cls(model, tokenizer, model_id, config)

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        if self._config:
            return self._config.num_hidden_layers
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return len(self._model.model.layers)
        return 32

    def analyze_representations(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
        labels: list[str] | None = None,
        position: int = -1,
    ) -> LayerAnalysisResult:
        """
        Analyze representations at specified layers.

        Args:
            prompts: List of prompts to analyze
            layers: Layers to capture (default: key layers)
            labels: Optional labels for clustering analysis
            position: Sequence position to extract (-1 = last)

        Returns:
            LayerAnalysisResult with similarity matrices per layer
        """
        from .hooks import CaptureConfig, ModelHooks, PositionSelection

        if layers is None:
            # Default: analyze key layers
            n = self.num_layers
            layers = [0, n // 8, n // 4, n // 2, 3 * n // 4, n - 1]
            layers = sorted(set(layers))

        representations: dict[int, RepresentationResult] = {}
        clusters: dict[int, ClusterResult] | None = {} if labels else None

        # Collect representations for each prompt
        prompt_reps: dict[int, dict[str, mx.array]] = {layer: {} for layer in layers}

        for prompt in prompts:
            input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]

            hooks = ModelHooks(self._model)
            hooks.configure(
                CaptureConfig(
                    layers=layers,
                    capture_hidden_states=True,
                    positions=PositionSelection.ALL,
                )
            )

            hooks.forward(input_ids)

            for layer_idx in layers:
                hidden = hooks.state.hidden_states.get(layer_idx)
                if hidden is not None:
                    # Get representation at specified position
                    if hidden.ndim == 3:
                        rep = hidden[0, position, :]
                    else:
                        rep = hidden[position, :]
                    prompt_reps[layer_idx][prompt] = rep

        # Compute similarity matrices
        for layer_idx in layers:
            reps = prompt_reps[layer_idx]
            sim_matrix = self._compute_similarity_matrix(prompts, reps)

            representations[layer_idx] = RepresentationResult(
                layer_idx=layer_idx,
                prompts=prompts,
                representations=reps,
                similarity_matrix=sim_matrix,
                labels=labels,
            )

            # Compute clustering if labels provided
            if labels and clusters is not None:
                clusters[layer_idx] = self._compute_clustering(prompts, labels, sim_matrix)

        return LayerAnalysisResult(
            prompts=prompts,
            labels=labels,
            layers=layers,
            representations=representations,
            clusters=clusters,
        )

    def analyze_attention(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
    ) -> dict[int, dict[str, AttentionResult]]:
        """
        Analyze attention patterns at specified layers.

        Args:
            prompts: Prompts to analyze
            layers: Layers to capture

        Returns:
            Dict mapping layer -> prompt -> AttentionResult
        """
        from .hooks import CaptureConfig, ModelHooks, PositionSelection

        if layers is None:
            n = self.num_layers
            layers = [n // 4, n // 2]  # Default to quarter and half

        results: dict[int, dict[str, AttentionResult]] = {layer: {} for layer in layers}

        for prompt in prompts:
            input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
            tokens = [self._tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

            hooks = ModelHooks(self._model)
            hooks.configure(
                CaptureConfig(
                    layers=layers,
                    capture_hidden_states=False,
                    capture_attention_weights=True,
                    positions=PositionSelection.ALL,
                )
            )

            hooks.forward(input_ids)

            for layer_idx in layers:
                attn = hooks.state.attention_weights.get(layer_idx)
                if attn is not None:
                    # Remove batch dim if present
                    if attn.ndim == 4:
                        attn = attn[0]

                    results[layer_idx][prompt] = AttentionResult(
                        layer_idx=layer_idx,
                        prompt=prompt,
                        tokens=tokens,
                        attention_weights=attn,
                    )

        return results

    def _compute_similarity_matrix(
        self,
        prompts: list[str],
        representations: dict[str, mx.array],
    ) -> list[list[float]]:
        """Compute cosine similarity between all prompt pairs."""
        n = len(prompts)
        matrix = [[0.0] * n for _ in range(n)]

        for i, p1 in enumerate(prompts):
            for j, p2 in enumerate(prompts):
                if i <= j:
                    rep1 = representations[p1]
                    rep2 = representations[p2]

                    # Cosine similarity
                    dot = float(mx.sum(rep1 * rep2))
                    norm1 = float(mx.sqrt(mx.sum(rep1 * rep1)))
                    norm2 = float(mx.sqrt(mx.sum(rep2 * rep2)))
                    sim = dot / (norm1 * norm2 + 1e-8)

                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def _compute_clustering(
        self,
        prompts: list[str],
        labels: list[str],
        similarity_matrix: list[list[float]],
    ) -> ClusterResult:
        """Compute clustering metrics."""
        # Group prompts by label
        label_groups: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)

        # Within-cluster similarity
        within_sim: dict[str, float] = {}
        for label, indices in label_groups.items():
            if len(indices) < 2:
                within_sim[label] = 1.0
                continue
            sims = []
            for i in indices:
                for j in indices:
                    if i < j:
                        sims.append(similarity_matrix[i][j])
            within_sim[label] = sum(sims) / len(sims) if sims else 1.0

        # Between-cluster similarity
        between_sim: dict[tuple[str, str], float] = {}
        unique_labels = list(label_groups.keys())
        for i, l1 in enumerate(unique_labels):
            for j, l2 in enumerate(unique_labels):
                if i < j:
                    sims = []
                    for idx1 in label_groups[l1]:
                        for idx2 in label_groups[l2]:
                            sims.append(similarity_matrix[idx1][idx2])
                    between_sim[(l1, l2)] = sum(sims) / len(sims) if sims else 0.0

        # Separation score
        avg_within = sum(within_sim.values()) / len(within_sim)
        avg_between = sum(between_sim.values()) / len(between_sim) if between_sim else 0.0
        separation = avg_within - avg_between

        return ClusterResult(
            layer_idx=0,  # Will be set by caller
            labels=list(label_groups.keys()),
            within_cluster_similarity=within_sim,
            between_cluster_similarity=between_sim,
            separation_score=separation,
        )

    def print_similarity_matrix(
        self,
        result: LayerAnalysisResult,
        layer: int,
    ) -> None:
        """Print similarity matrix for a layer."""
        rep_result = result.representations[layer]
        prompts = rep_result.prompts
        matrix = rep_result.similarity_matrix
        labels = rep_result.labels

        print(f"\n=== Layer {layer} Similarity Matrix ===\n")

        # Header
        max_len = max(len(p[:20]) for p in prompts)
        header = " " * (max_len + 2)
        for i, p in enumerate(prompts):
            label = f"[{labels[i]}]" if labels else ""
            header += f"  {i:>3}"
        print(header)

        # Rows
        for i, p in enumerate(prompts):
            label = f"[{labels[i]}]" if labels else ""
            row = f"{p[:20]:<{max_len}} {label:>8}"
            for j in range(len(prompts)):
                sim = matrix[i][j]
                # Highlight diagonal and high similarities
                if i == j:
                    row += " 1.00"
                elif sim > 0.95:
                    row += f" {sim:.2f}*"
                else:
                    row += f" {sim:.2f}"
            print(row)

        # Clustering summary if available
        if result.clusters and layer in result.clusters:
            cluster = result.clusters[layer]
            print("\n--- Clustering Analysis ---")
            print("Within-cluster similarity:")
            for label, sim in cluster.within_cluster_similarity.items():
                print(f"  {label}: {sim:.4f}")
            print("Between-cluster similarity:")
            for (l1, l2), sim in cluster.between_cluster_similarity.items():
                print(f"  {l1} <-> {l2}: {sim:.4f}")
            print(f"Separation score: {cluster.separation_score:.4f}")

    def print_attention_comparison(
        self,
        attention_results: dict[int, dict[str, AttentionResult]],
        layer: int,
        prompts: list[str],
        focus_token: str | int = -1,
    ) -> None:
        """Compare attention patterns between prompts at a layer."""
        print(f"\n=== Layer {layer} Attention Patterns ===\n")

        layer_results = attention_results.get(layer, {})

        for prompt in prompts:
            if prompt not in layer_results:
                continue

            result = layer_results[prompt]
            tokens = result.tokens

            # Find focus token index
            if isinstance(focus_token, str):
                try:
                    focus_idx = tokens.index(focus_token)
                except ValueError:
                    focus_idx = -1
            else:
                focus_idx = focus_token

            print(f"Prompt: {prompt!r}")
            print(f"Tokens: {tokens}")
            print(f"Attention from position {focus_idx} ({tokens[focus_idx]!r}):")

            # Average attention across heads
            attn = result.attention_weights[:, focus_idx, :]  # [heads, seq_len]
            avg_attn = mx.mean(attn, axis=0)  # [seq_len]

            for i, (tok, weight) in enumerate(zip(tokens, avg_attn.tolist())):
                bar = "#" * int(weight * 50)
                marker = " <--" if i == focus_idx else ""
                print(f"  {i:2d} {tok!r:15s}: {weight:.4f} {bar}{marker}")
            print()


def analyze_format_sensitivity(
    model_id: str,
    base_prompts: list[str],
    layers: list[int] | None = None,
) -> LayerAnalysisResult:
    """
    Convenience function to analyze format sensitivity.

    Takes base prompts and creates working/broken variants
    (with/without trailing space).

    Args:
        model_id: Model to analyze
        base_prompts: Base prompts (will add/remove trailing space)
        layers: Layers to analyze

    Returns:
        LayerAnalysisResult with working/broken labels
    """
    analyzer = LayerAnalyzer.from_pretrained(model_id)

    prompts = []
    labels = []

    for base in base_prompts:
        base = base.rstrip()
        prompts.append(base + " ")  # With trailing space
        labels.append("working")
        prompts.append(base)  # Without trailing space
        labels.append("broken")

    return analyzer.analyze_representations(prompts, layers=layers, labels=labels)
