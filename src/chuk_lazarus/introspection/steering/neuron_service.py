"""Service layer for neuron analysis CLI commands.

This module provides the NeuronAnalysisService class that provides
functionality for analyzing individual neuron activations.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..ablation import AblationStudy
from ..hooks import CaptureConfig, ModelHooks, PositionSelection


class NeuronActivationResult(BaseModel):
    """Result of neuron activation analysis."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    neuron_idx: int = Field(..., description="Neuron index")
    min_val: float = Field(..., description="Minimum activation")
    max_val: float = Field(..., description="Maximum activation")
    mean_val: float = Field(..., description="Mean activation")
    std_val: float = Field(..., description="Standard deviation")
    weight: float | None = Field(default=None, description="Weight from direction file")
    separation: float | None = Field(default=None, description="Separation score (auto-discover)")


class DiscoveredNeuron(BaseModel):
    """Result of auto-discovered discriminative neuron."""

    model_config = ConfigDict(frozen=True)

    idx: int = Field(..., description="Neuron index")
    separation: float = Field(..., description="Separation score")
    best_pair: tuple[str, str] | None = Field(default=None, description="Best label pair")
    overall_std: float = Field(..., description="Overall standard deviation")
    mean_range: float = Field(..., description="Range of group means")
    group_means: dict[str, float] = Field(default_factory=dict, description="Mean per label group")


class NeuronAnalysisServiceConfig(BaseModel):
    """Configuration for NeuronAnalysisService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    layers: list[int] = Field(..., description="Layers to analyze")
    neurons: list[int] | None = Field(default=None, description="Specific neurons to analyze")
    top_k: int = Field(default=10, description="Top K neurons for auto-discovery")


class NeuronAnalysisService:
    """Service class for neuron analysis operations.

    Provides a high-level interface for CLI commands to run neuron analysis
    without needing to understand the internal architecture.
    """

    Config = NeuronAnalysisServiceConfig

    @classmethod
    def load_neurons_from_direction(
        cls,
        direction_path: str,
        top_k: int = 10,
    ) -> tuple[list[int], dict[int, float], dict[str, str]]:
        """Load top neurons from a saved direction file.

        Args:
            direction_path: Path to direction file.
            top_k: Number of top neurons to load.

        Returns:
            Tuple of (neuron_indices, neuron_weights, metadata).
        """
        data = np.load(direction_path, allow_pickle=True)
        direction = data["direction"]

        # Get top neurons by absolute weight
        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        neurons = [int(i) for i in top_indices]
        weights = {int(i): float(direction[i]) for i in top_indices}

        metadata = {}
        if "label_positive" in data:
            metadata["positive_label"] = str(data["label_positive"])
            metadata["negative_label"] = str(data["label_negative"])

        return neurons, weights, metadata

    @classmethod
    async def auto_discover_neurons(
        cls,
        model: str,
        prompts: list[str],
        labels: list[str],
        layer: int,
        top_k: int = 10,
    ) -> list[DiscoveredNeuron]:
        """Auto-discover discriminative neurons based on label groups.

        Args:
            model: Model path or name.
            prompts: List of prompts.
            labels: Labels for each prompt.
            layer: Layer to analyze.
            top_k: Number of top neurons to return.

        Returns:
            List of discovered neurons sorted by separation score.
        """
        study = AblationStudy.from_pretrained(model)
        model_obj = study.adapter.model
        tokenizer = study.adapter.tokenizer
        config = study.adapter.config

        # Collect hidden states
        full_activations = []
        for prompt in prompts:
            hooks = ModelHooks(model_obj, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=[layer],
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))
            h = hooks.state.hidden_states[layer][0, 0, :]
            h_np = np.array(h.astype(mx.float32), copy=False)
            full_activations.append(h_np)

        full_activations = np.array(full_activations)
        num_neurons = full_activations.shape[1]

        # Group by label
        unique_labels = sorted(set(labels))
        label_groups = {lbl: [] for lbl in unique_labels}
        for i, lbl in enumerate(labels):
            label_groups[lbl].append(full_activations[i])
        for lbl in unique_labels:
            label_groups[lbl] = np.array(label_groups[lbl])

        # Calculate separation for each neuron
        single_sample = all(len(label_groups[lbl]) == 1 for lbl in unique_labels)
        neuron_scores = []

        for neuron_idx in range(num_neurons):
            group_means = []
            group_stds = []
            for lbl in unique_labels:
                vals = label_groups[lbl][:, neuron_idx]
                group_means.append(np.mean(vals))
                group_stds.append(np.std(vals))

            overall_std = np.std(full_activations[:, neuron_idx])

            # Find max pairwise separation
            max_separation = 0.0
            best_pair = None
            for i, lbl1 in enumerate(unique_labels):
                for j, lbl2 in enumerate(unique_labels):
                    if i >= j:
                        continue
                    mean_diff = abs(group_means[i] - group_means[j])

                    if single_sample:
                        separation = mean_diff / overall_std if overall_std > 1e-6 else 0.0
                    else:
                        pooled_std = np.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                        separation = mean_diff / pooled_std if pooled_std > 1e-6 else 0.0

                    if separation > max_separation:
                        max_separation = separation
                        best_pair = (lbl1, lbl2)

            mean_range = max(group_means) - min(group_means)

            neuron_scores.append(
                DiscoveredNeuron(
                    idx=neuron_idx,
                    separation=max_separation,
                    best_pair=best_pair,
                    overall_std=overall_std,
                    mean_range=mean_range,
                    group_means={lbl: group_means[i] for i, lbl in enumerate(unique_labels)},
                )
            )

        # Sort and return top-k
        neuron_scores.sort(key=lambda x: -x.separation)
        return neuron_scores[:top_k]

    @classmethod
    async def analyze_neurons(
        cls,
        model: str,
        prompts: list[str],
        neurons: list[int],
        layers: list[int],
        steer_config: dict[str, Any] | None = None,
    ) -> dict[int, list[NeuronActivationResult]]:
        """Analyze neuron activations across prompts.

        Args:
            model: Model path or name.
            prompts: List of prompts to analyze.
            neurons: Neuron indices to analyze.
            layers: Layers to analyze.
            steer_config: Optional steering configuration.

        Returns:
            Dict mapping layer -> list of neuron results.
        """
        study = AblationStudy.from_pretrained(model)
        model_obj = study.adapter.model
        tokenizer = study.adapter.tokenizer
        config = study.adapter.config

        # Collect activations
        all_activations = {layer: [] for layer in layers}

        steerer = None
        if steer_config:
            from . import ActivationSteering

            steerer = ActivationSteering(model_obj, tokenizer)
            steerer.add_direction(
                steer_config["layer"],
                mx.array(steer_config["direction"]),
            )
            steerer._wrap_layer(steer_config["layer"], steer_config["coefficient"])

        try:
            for prompt in prompts:
                hooks = ModelHooks(model_obj, model_config=config)
                hooks.configure(
                    CaptureConfig(
                        layers=layers,
                        capture_hidden_states=True,
                        positions=PositionSelection.LAST,
                    )
                )
                input_ids = tokenizer.encode(prompt, return_tensors="np")
                hooks.forward(mx.array(input_ids))

                for layer in layers:
                    h = hooks.state.hidden_states[layer][0, 0, :]
                    h_np = np.array(h.astype(mx.float32), copy=False)
                    all_activations[layer].append(h_np)
        finally:
            if steerer:
                steerer._unwrap_layers()

        # Compute statistics per layer
        results = {}
        for layer in layers:
            activations = np.array(all_activations[layer])
            layer_results = []

            for neuron in neurons:
                vals = activations[:, neuron]
                layer_results.append(
                    NeuronActivationResult(
                        neuron_idx=neuron,
                        min_val=float(vals.min()),
                        max_val=float(vals.max()),
                        mean_val=float(vals.mean()),
                        std_val=float(vals.std()),
                    )
                )

            results[layer] = layer_results

        return results


__all__ = [
    "NeuronAnalysisService",
    "NeuronAnalysisServiceConfig",
    "NeuronActivationResult",
    "DiscoveredNeuron",
]
