"""
Activation collector for circuit analysis.

Collects and stores activations from model forward passes for later analysis.
Supports collecting hidden states, attention weights, and MLP outputs.

This module is generic - works with any dataset type (arithmetic, tool-calling,
factual, safety, etc.)

Example:
    >>> from chuk_lazarus.introspection.circuit import (
    ...     ActivationCollector, CollectorConfig, create_arithmetic_dataset
    ... )
    >>>
    >>> dataset = create_arithmetic_dataset()
    >>> collector = ActivationCollector.from_pretrained("mlx-community/gemma-3-4b-it-bf16")
    >>>
    >>> activations = collector.collect(dataset)
    >>> activations.save("arithmetic_activations.safetensors")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .dataset import CircuitDataset, LabeledPrompt


class CollectorConfig(BaseModel):
    """Configuration for activation collection."""

    model_config = ConfigDict(frozen=True)

    # Which layers to capture
    layers: list[int] | str = Field(
        default="all", description="'all', 'decision', or explicit list"
    )
    decision_layer_range: tuple[int, int] = Field(
        default=(8, 14), description="Layer range for 'decision' mode"
    )

    # What to capture
    capture_hidden_states: bool = Field(default=True, description="Capture hidden states")
    capture_attention_weights: bool = Field(default=False, description="Capture attention")
    capture_mlp_intermediate: bool = Field(default=False, description="Capture MLP intermediate")

    # Position to capture (usually last token for next-token prediction)
    position: int = Field(default=-1, description="Position to capture")

    # Storage settings
    dtype: str = Field(default="float32", description="float32, float16, bfloat16")

    # Generation settings for criterion evaluation
    max_new_tokens: int = Field(default=30, ge=1, description="Max tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature")


class CollectedActivations(BaseModel):
    """Container for collected activations with metadata.

    Generic container that works with any label scheme.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    # Activations: shape [num_samples, hidden_size] per layer
    hidden_states: dict[int, mx.array] = Field(
        default_factory=dict, description="Hidden states per layer"
    )

    # Optional: attention weights per layer
    attention_weights: dict[int, mx.array] = Field(
        default_factory=dict, description="Attention weights per layer"
    )

    # Optional: MLP intermediate activations
    mlp_intermediates: dict[int, mx.array] = Field(
        default_factory=dict, description="MLP intermediates per layer"
    )

    # Labels and metadata (generic)
    labels: list[int] = Field(default_factory=list, description="Sample labels")
    label_names: list[str] = Field(default_factory=list, description="Label name per sample")
    categories: list[str] = Field(default_factory=list, description="Category per sample")
    prompts: list[str] = Field(default_factory=list, description="Prompt texts")
    expected_outputs: list[str | None] = Field(default_factory=list, description="Expected outputs")
    model_outputs: list[str] = Field(default_factory=list, description="Model outputs")

    # Model info
    model_id: str = Field(default="", description="Model identifier")
    hidden_size: int = Field(default=0, description="Hidden dimension size")
    num_layers: int = Field(default=0, description="Number of layers")

    # Dataset metadata
    dataset_name: str = Field(default="", description="Dataset name")
    dataset_label_names: dict[int, str] = Field(
        default_factory=dict, description="Label int to name mapping"
    )

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def category_labels(self) -> list[str]:
        """Alias for categories (backwards compatibility)."""
        return self.categories

    @property
    def tool_labels(self) -> list[str | None]:
        """Extract tool labels from categories for tool-type probing."""
        # Return categories as tool labels (None for generic categories)
        return [
            cat if cat not in ("default", "positive", "negative", "") else None
            for cat in self.categories
        ]

    @property
    def captured_layers(self) -> list[int]:
        """Get sorted list of captured layer indices."""
        return sorted(self.hidden_states.keys())

    def get_layer_activations(self, layer: int) -> mx.array | None:
        """Get hidden states for a specific layer."""
        return self.hidden_states.get(layer)

    def get_activations_numpy(self, layer: int) -> np.ndarray | None:
        """Get hidden states as numpy array for sklearn compatibility."""
        acts = self.hidden_states.get(layer)
        if acts is None:
            return None
        # Cast to float32 in MLX first to handle bfloat16
        acts_f32 = acts.astype(mx.float32)
        return np.array(acts_f32, copy=False)

    def get_by_label(self, label: int) -> tuple[np.ndarray, list[int]]:
        """Get indices of samples with a specific label."""
        indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
        return np.array(indices), indices

    def get_label_mask(self, label: int) -> np.ndarray:
        """Get boolean mask for samples with a specific label."""
        return np.array(self.labels) == label

    def split_by_label(self, layer: int) -> dict[int, mx.array]:
        """Split activations by label."""
        acts = self.hidden_states[layer]
        result = {}
        for label in set(self.labels):
            mask = mx.array(self.labels) == label
            result[label] = acts[mask]
        return result

    def get_positive_negative(self, layer: int) -> tuple[mx.array, mx.array]:
        """Get activations split into positive (label=1) and negative (label=0)."""
        acts = self.hidden_states[layer]
        pos_mask = mx.array(self.labels) == 1
        neg_mask = mx.array(self.labels) == 0
        return acts[pos_mask], acts[neg_mask]

    def summary(self) -> dict:
        """Get summary statistics."""
        label_counts = {}
        for label in set(self.labels):
            name = self.dataset_label_names.get(label, f"label_{label}")
            label_counts[name] = self.labels.count(label)

        category_counts = {}
        for cat in set(self.categories):
            category_counts[cat] = self.categories.count(cat)

        return {
            "num_samples": len(self),
            "by_label": label_counts,
            "by_category": category_counts,
            "captured_layers": self.captured_layers,
            "hidden_size": self.hidden_size,
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
        }

    def save(self, path: str | Path, include_outputs: bool = False) -> None:
        """
        Save activations to safetensors format with JSON metadata.

        Creates two files:
        - {path}.safetensors: The activation tensors
        - {path}.json: Metadata (labels, prompts, etc.)
        """
        path = Path(path)

        # Prepare tensors for safetensors
        tensors = {}

        def to_numpy_float32(arr: mx.array) -> np.ndarray:
            """Convert MLX array to numpy float32, handling bfloat16."""
            arr_f32 = arr.astype(mx.float32)
            return np.array(arr_f32, copy=False)

        # Hidden states: layer_{idx}
        for layer_idx, acts in self.hidden_states.items():
            key = f"hidden_states.layer_{layer_idx}"
            tensors[key] = to_numpy_float32(acts)

        # Attention weights if present
        for layer_idx, attn in self.attention_weights.items():
            key = f"attention.layer_{layer_idx}"
            tensors[key] = to_numpy_float32(attn)

        # MLP intermediates if present
        for layer_idx, mlp in self.mlp_intermediates.items():
            key = f"mlp.layer_{layer_idx}"
            tensors[key] = to_numpy_float32(mlp)

        # Save tensors
        safetensor_path = path.with_suffix(".safetensors")
        self._save_safetensors(tensors, safetensor_path)

        # Save metadata
        metadata = {
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_samples": len(self),
            "captured_layers": self.captured_layers,
            "labels": self.labels,
            "label_names": self.label_names,
            "categories": self.categories,
            "prompts": self.prompts,
            "expected_outputs": self.expected_outputs,
            "dataset_name": self.dataset_name,
            "dataset_label_names": {str(k): v for k, v in self.dataset_label_names.items()},
        }

        if include_outputs:
            metadata["model_outputs"] = self.model_outputs

        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved activations to: {safetensor_path}")
        print(f"Saved metadata to: {json_path}")

    @staticmethod
    def _save_safetensors(tensors: dict[str, np.ndarray], path: Path) -> None:
        """Save tensors to safetensors format."""
        try:
            from safetensors.numpy import save_file

            save_file(tensors, str(path))
        except ImportError:
            # Fallback to numpy format
            np_path = path.with_suffix(".npz")
            np.savez(np_path, **tensors)
            print(f"Note: safetensors not installed, saved as {np_path}")

    @classmethod
    def load(cls, path: str | Path) -> CollectedActivations:
        """Load activations from safetensors + JSON."""
        path = Path(path)

        # Load metadata
        json_path = path.with_suffix(".json")
        with open(json_path) as f:
            metadata = json.load(f)

        # Load tensors
        safetensor_path = path.with_suffix(".safetensors")
        tensors = cls._load_safetensors(safetensor_path)

        # Parse into structure
        hidden_states = {}
        attention_weights = {}
        mlp_intermediates = {}

        for key, arr in tensors.items():
            if key.startswith("hidden_states.layer_"):
                layer_idx = int(key.split("_")[-1])
                hidden_states[layer_idx] = mx.array(arr)
            elif key.startswith("attention.layer_"):
                layer_idx = int(key.split("_")[-1])
                attention_weights[layer_idx] = mx.array(arr)
            elif key.startswith("mlp.layer_"):
                layer_idx = int(key.split("_")[-1])
                mlp_intermediates[layer_idx] = mx.array(arr)

        # Parse label names back to int keys
        dataset_label_names = {
            int(k): v for k, v in metadata.get("dataset_label_names", {}).items()
        }

        return cls(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            mlp_intermediates=mlp_intermediates,
            labels=metadata["labels"],
            label_names=metadata.get("label_names", []),
            categories=metadata.get("categories", []),
            prompts=metadata["prompts"],
            expected_outputs=metadata.get("expected_outputs", []),
            model_outputs=metadata.get("model_outputs", []),
            model_id=metadata["model_id"],
            hidden_size=metadata["hidden_size"],
            num_layers=metadata["num_layers"],
            dataset_name=metadata.get("dataset_name", ""),
            dataset_label_names=dataset_label_names,
        )

    @staticmethod
    def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
        """Load tensors from safetensors or npz."""
        try:
            from safetensors.numpy import load_file

            return load_file(str(path))
        except ImportError:
            npz_path = path.with_suffix(".npz")
            if npz_path.exists():
                return dict(np.load(npz_path))
            raise


class ActivationCollector:
    """
    Collects activations from a model across a dataset.

    Works with any CircuitDataset (arithmetic, tool-calling, factual, etc.)

    Example:
        >>> collector = ActivationCollector.from_pretrained("model_id")
        >>> config = CollectorConfig(layers=[8, 10, 12])
        >>> activations = collector.collect(dataset, config)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        model_id: str = "unknown",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = config
        self.model_id = model_id

        # Detect model structure
        self._detect_structure()

    def _detect_structure(self):
        """Detect model structure for accessing layers."""
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
        if hasattr(self.model_config, "hidden_size"):
            self.hidden_size = self.model_config.hidden_size
        elif hasattr(self._backbone, "hidden_size"):
            self.hidden_size = self._backbone.hidden_size
        else:
            self.hidden_size = 768  # Fallback

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_family: str | None = None,
    ) -> ActivationCollector:
        """Load a model for activation collection."""
        from ..ablation import AblationStudy

        # Reuse AblationStudy's model loading logic
        study = AblationStudy.from_pretrained(model_id, model_family)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            config=study.adapter.config,
            model_id=model_id,
        )

    def _get_layers_to_capture(self, config: CollectorConfig) -> list[int]:
        """Determine which layers to capture based on config."""
        if isinstance(config.layers, list):
            return config.layers

        if config.layers == "all":
            return list(range(self.num_layers))

        if config.layers == "decision":
            # Focus on decision layers (typically middle-to-late layers)
            start, end = config.decision_layer_range
            start = max(0, min(start, self.num_layers - 1))
            end = max(0, min(end, self.num_layers))
            return list(range(start, end))

        return [self.num_layers // 2, self.num_layers - 1]  # Fallback

    def collect_single(
        self,
        prompt: str,
        config: CollectorConfig | None = None,
    ) -> dict[int, mx.array]:
        """
        Collect activations for a single prompt.

        Returns dict mapping layer index to hidden state tensor.
        """
        if config is None:
            config = CollectorConfig()

        from ..hooks import CaptureConfig, ModelHooks, PositionSelection

        layers = self._get_layers_to_capture(config)

        # Setup hooks
        hooks = ModelHooks(self.model, model_config=self.model_config)
        position = PositionSelection.LAST if config.position == -1 else PositionSelection.ALL
        hooks.configure(
            CaptureConfig(
                layers=layers,
                capture_hidden_states=config.capture_hidden_states,
                capture_attention_weights=config.capture_attention_weights,
                positions=position,
            )
        )

        # Tokenize and forward
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        hooks.forward(input_ids)

        # Extract hidden states at the target position
        result = {}
        for layer_idx in layers:
            if layer_idx in hooks.state.hidden_states:
                h = hooks.state.hidden_states[layer_idx]
                # Shape is [batch, seq, hidden] or [batch, 1, hidden] for LAST
                if h.ndim == 3:
                    h = h[0, -1, :]  # [hidden]
                elif h.ndim == 2:
                    h = h[-1, :]  # [hidden]
                result[layer_idx] = h

        return result

    def collect(
        self,
        dataset: CircuitDataset | list[LabeledPrompt],
        config: CollectorConfig | None = None,
        progress: bool = True,
    ) -> CollectedActivations:
        """
        Collect activations across entire dataset.

        Args:
            dataset: Dataset of prompts to collect activations for
            config: Collection configuration
            progress: Whether to show progress

        Returns:
            CollectedActivations with all hidden states and labels
        """
        if config is None:
            config = CollectorConfig()

        if isinstance(dataset, CircuitDataset):
            prompts = list(dataset)
            dataset_name = dataset.name
            dataset_label_names = dataset.label_names
        else:
            prompts = dataset
            dataset_name = "custom"
            dataset_label_names = {}

        layers = self._get_layers_to_capture(config)

        # Initialize storage
        hidden_by_layer: dict[int, list[mx.array]] = {layer: [] for layer in layers}
        labels = []
        label_names = []
        categories = []
        prompt_texts = []
        expected_outputs = []
        outputs = []

        n = len(prompts)
        for i, prompt in enumerate(prompts):
            if progress and (i + 1) % 10 == 0:
                print(f"Collecting {i + 1}/{n}...")

            # Collect activations
            acts = self.collect_single(prompt.text, config)
            for layer_idx, h in acts.items():
                hidden_by_layer[layer_idx].append(h)

            # Collect labels
            labels.append(prompt.label)
            label_names.append(prompt.label_name or "")
            categories.append(prompt.category)
            prompt_texts.append(prompt.text)
            expected_outputs.append(prompt.expected_output)

            # Optionally generate output
            if config.max_new_tokens > 0:
                output = self._generate(prompt.text, config)
                outputs.append(output)

        # Stack activations
        hidden_states = {}
        for layer_idx, acts_list in hidden_by_layer.items():
            if acts_list:
                hidden_states[layer_idx] = mx.stack(acts_list)

        return CollectedActivations(
            hidden_states=hidden_states,
            labels=labels,
            label_names=label_names,
            categories=categories,
            prompts=prompt_texts,
            expected_outputs=expected_outputs,
            model_outputs=outputs,
            model_id=self.model_id,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dataset_name=dataset_name,
            dataset_label_names=dataset_label_names,
        )

    def _generate(self, prompt: str, config: CollectorConfig) -> str:
        """Generate output for criterion evaluation."""
        from ..ablation import ModelAdapter

        adapter = ModelAdapter(self.model, self.tokenizer, self.model_config)
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        return adapter.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )


def collect_activations(
    model_id: str,
    dataset: CircuitDataset | None = None,
    layers: list[int] | str = "all",
    output_path: str | None = None,
) -> CollectedActivations:
    """
    Convenience function to collect activations.

    Args:
        model_id: HuggingFace model ID
        dataset: Dataset to use (creates arithmetic dataset if None)
        layers: Which layers to capture
        output_path: Optional path to save activations

    Returns:
        CollectedActivations
    """
    from .dataset import create_arithmetic_dataset

    if dataset is None:
        dataset = create_arithmetic_dataset()

    collector = ActivationCollector.from_pretrained(model_id)
    config = CollectorConfig(layers=layers)

    activations = collector.collect(dataset, config)

    if output_path:
        activations.save(output_path)

    return activations
