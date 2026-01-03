"""
Linear probe battery for understanding layer-wise feature encoding.

Tests what each layer encodes to understand the computational stratigraphy.
This module is generic - works with any binary or multi-class classification task.

Example use cases:
- Arithmetic: Does this layer encode "computation mode" vs "retrieval mode"?
- Tool-calling: Does this layer encode "needs tool" vs "no tool"?
- Factual: Does this layer encode "consistent" vs "contradictory"?
- Safety: Does this layer encode "safe" vs "unsafe"?

Example:
    >>> from chuk_lazarus.introspection.circuit import ProbeBattery, ProbeDataset
    >>>
    >>> # Create custom probe dataset
    >>> probe = ProbeDataset(
    ...     name="arithmetic_mode",
    ...     description="Detect arithmetic vs retrieval mode",
    ...     prompts=["6 * 7 =", "The capital of France is"],
    ...     labels=[1, 0],
    ...     label_names=["arithmetic", "retrieval"],
    ... )
    >>>
    >>> battery = ProbeBattery.from_pretrained("gemma-3-4b-it")
    >>> battery.add_dataset(probe)
    >>> results = battery.run_all_probes(layers=[20, 22, 24, 26, 28, 30])
    >>> battery.print_results_table(results)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class ProbeDataset(BaseModel):
    """A labeled dataset for probing a specific feature.

    Generic - can probe any binary or multi-class feature.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Dataset name")
    description: str = Field(description="Description of what is being probed")
    prompts: list[str] = Field(description="List of prompts")
    labels: list[int] = Field(description="Labels for each prompt")
    label_names: list[str] = Field(
        default_factory=lambda: ["class_0", "class_1"], description="Names for label classes"
    )
    category: str = Field(default="custom", description="Category for grouping probes")

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> ProbeDataset:
        """Load from dict (parsed from YAML/JSON)."""
        return cls(
            name=name,
            description=data.get("description", ""),
            prompts=data["prompts"],
            labels=data["labels"],
            label_names=data.get("label_names", ["class_0", "class_1"]),
            category=data.get("category", "custom"),
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "description": self.description,
            "category": self.category,
            "label_names": self.label_names,
            "prompts": self.prompts,
            "labels": self.labels,
        }

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy of always predicting majority class."""
        if not self.labels:
            return 0.5
        unique, counts = np.unique(self.labels, return_counts=True)
        return counts.max() / len(self.labels)

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        return len(set(self.labels))


class ProbeResult(BaseModel):
    """Result of running a probe at a specific layer."""

    model_config = ConfigDict(frozen=True)

    probe_name: str = Field(description="Name of the probe")
    layer: int = Field(description="Layer index")
    accuracy: float = Field(description="Test accuracy")
    cv_std: float = Field(description="Cross-validation standard deviation")
    baseline: float = Field(description="Baseline (majority class) accuracy")
    above_chance: float = Field(description="Accuracy above chance")
    n_samples: int = Field(description="Number of samples")

    @property
    def is_significant(self) -> bool:
        """Is this probe significantly above chance?"""
        return self.above_chance > 0.1 and self.accuracy > 0.6


class StratigraphyResult(BaseModel):
    """Results of probing across all layers."""

    model_config = ConfigDict(validate_default=True)

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(description="Number of layers in model")
    probes: dict[str, dict[int, ProbeResult]] = Field(
        default_factory=dict, description="Probe results by name and layer"
    )

    def get_accuracy_matrix(self, layers: list[int] | None = None) -> dict[str, list[float]]:
        """Get accuracy matrix for visualization."""
        if layers is None:
            layers = sorted({layer for r in self.probes.values() for layer in r.keys()})

        matrix = {}
        for probe_name, layer_results in self.probes.items():
            matrix[probe_name] = [
                layer_results[layer].accuracy if layer in layer_results else 0.0 for layer in layers
            ]
        return matrix

    def find_emergence_layer(self, probe_name: str, threshold: float = 0.75) -> int | None:
        """Find first layer where probe exceeds threshold."""
        if probe_name not in self.probes:
            return None

        results = self.probes[probe_name]
        for layer in sorted(results.keys()):
            r = results[layer]
            if r.accuracy >= threshold and r.above_chance > 0.1:
                return layer
        return None

    def find_destruction_layer(self, probe_name: str, threshold: float = 0.5) -> int | None:
        """Find layer where probe drops below threshold after being high.

        Useful for finding suppression circuits like Gemma's arithmetic destruction.
        """
        if probe_name not in self.probes:
            return None

        results = self.probes[probe_name]
        sorted_layers = sorted(results.keys())
        was_high = False

        for layer in sorted_layers:
            r = results[layer]
            if r.accuracy >= 0.75:
                was_high = True
            elif was_high and r.accuracy < threshold:
                return layer

        return None

    def get_all_emergence_layers(self, threshold: float = 0.75) -> dict[str, int | None]:
        """Get emergence layer for all probes."""
        return {name: self.find_emergence_layer(name, threshold) for name in self.probes.keys()}

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        data = {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "probes": {
                name: {
                    str(layer): {
                        "accuracy": r.accuracy,
                        "cv_std": r.cv_std,
                        "baseline": r.baseline,
                        "above_chance": r.above_chance,
                        "n_samples": r.n_samples,
                    }
                    for layer, r in layer_results.items()
                }
                for name, layer_results in self.probes.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> StratigraphyResult:
        """Load results from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        probes = {}
        for name, layer_results in data["probes"].items():
            probes[name] = {}
            for layer_str, r in layer_results.items():
                layer = int(layer_str)
                probes[name][layer] = ProbeResult(
                    probe_name=name,
                    layer=layer,
                    accuracy=r["accuracy"],
                    cv_std=r["cv_std"],
                    baseline=r["baseline"],
                    above_chance=r["above_chance"],
                    n_samples=r["n_samples"],
                )

        return cls(
            model_id=data["model_id"],
            num_layers=data["num_layers"],
            probes=probes,
        )


class ProbeBattery:
    """
    Comprehensive probe battery for layer-wise feature analysis.

    Runs linear probes across layers to understand what each layer encodes.
    Generic - works with any probe dataset.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str = "unknown",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.datasets: dict[str, ProbeDataset] = {}

        # Detect model structure
        self._detect_structure()

    def _detect_structure(self):
        """Detect model structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            self._layers = self.model.layers
        else:
            raise ValueError("Cannot detect model layer structure")

        self.num_layers = len(self._layers)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dataset_dir: str | Path | None = None,
    ) -> ProbeBattery:
        """Load model and probe datasets."""
        from ..ablation import AblationStudy

        # Load model
        study = AblationStudy.from_pretrained(model_id)

        battery = cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

        # Load datasets from directory
        if dataset_dir:
            battery.load_datasets(dataset_dir)
        else:
            # Try default location
            default_dir = Path(__file__).parent / "probe_datasets"
            if default_dir.exists():
                battery.load_datasets(default_dir)

        return battery

    def load_datasets(self, path: str | Path) -> None:
        """Load probe datasets from a directory or file."""
        path = Path(path)

        if path.is_file():
            self._load_dataset_file(path)
        elif path.is_dir():
            for file in path.glob("*.yaml"):
                self._load_dataset_file(file)
            for file in path.glob("*.json"):
                self._load_dataset_file(file)
        else:
            raise ValueError(f"Path not found: {path}")

    def _load_dataset_file(self, path: Path) -> None:
        """Load datasets from a single file."""
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                print("PyYAML not installed, skipping YAML files")
                return
        else:
            with open(path) as f:
                data = json.load(f)

        # File can contain single dataset or multiple
        if "prompts" in data:
            # Single dataset
            name = path.stem
            self.datasets[name] = ProbeDataset.from_dict(name, data)
        else:
            # Multiple datasets
            for name, ds_data in data.items():
                self.datasets[name] = ProbeDataset.from_dict(name, ds_data)

    def add_dataset(self, dataset: ProbeDataset) -> None:
        """Add a probe dataset."""
        self.datasets[dataset.name] = dataset

    def get_activations(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Get activations for a prompt at a specific layer."""
        import mlx.core as mx

        from ..hooks import CaptureConfig, ModelHooks, PositionSelection

        hooks = ModelHooks(self.model)
        pos_sel = PositionSelection.LAST if position == -1 else PositionSelection.ALL

        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
                positions=pos_sel,
            )
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        hooks.forward(input_ids)

        h = hooks.state.hidden_states[layer]
        # Cast to float32 to handle bfloat16 before numpy conversion
        h_f32 = h.astype(mx.float32)
        if h_f32.ndim == 3:
            return np.array(h_f32[0, position, :], copy=False)
        return np.array(h_f32[position, :], copy=False)

    def collect_dataset_activations(
        self,
        dataset: ProbeDataset,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect activations for all prompts in a dataset."""
        activations = []
        for prompt in dataset.prompts:
            act = self.get_activations(prompt, layer)
            activations.append(act)

        X = np.stack(activations)
        y = np.array(dataset.labels)
        return X, y

    def train_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> tuple[float, float]:
        """Train logistic regression probe with cross-validation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Handle case where we have too few samples for CV
        n_samples = len(y)
        actual_folds = min(cv_folds, n_samples)
        if actual_folds < 2:
            # Not enough samples for CV
            return 0.5, 0.0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        probe = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X_scaled, y, cv=actual_folds)

        return scores.mean(), scores.std()

    def run_probe(
        self,
        dataset_name: str,
        layer: int,
    ) -> ProbeResult:
        """Run a single probe at a specific layer."""
        dataset = self.datasets[dataset_name]
        X, y = self.collect_dataset_activations(dataset, layer)

        accuracy, std = self.train_probe(X, y)
        baseline = dataset.baseline_accuracy

        return ProbeResult(
            probe_name=dataset_name,
            layer=layer,
            accuracy=accuracy,
            cv_std=std,
            baseline=baseline,
            above_chance=accuracy - baseline,
            n_samples=len(y),
        )

    def run_all_probes(
        self,
        layers: list[int] | None = None,
        categories: list[str] | None = None,
        progress: bool = True,
    ) -> StratigraphyResult:
        """
        Run all probes across specified layers.

        Args:
            layers: Layers to probe (default: evenly spaced)
            categories: Filter by category
            progress: Show progress

        Returns:
            StratigraphyResult with all probe results
        """
        if layers is None:
            # Default: evenly spaced layers
            layers = list(range(0, self.num_layers, max(1, self.num_layers // 10)))
            if (self.num_layers - 1) not in layers:
                layers.append(self.num_layers - 1)

        result = StratigraphyResult(
            model_id=self.model_id,
            num_layers=self.num_layers,
        )

        datasets_to_run = list(self.datasets.items())
        if categories:
            datasets_to_run = [
                (name, ds) for name, ds in datasets_to_run if ds.category in categories
            ]

        for name, dataset in datasets_to_run:
            if progress:
                print(f"Probing: {name} ({dataset.category})")

            result.probes[name] = {}
            for layer in layers:
                probe_result = self.run_probe(name, layer)
                result.probes[name][layer] = probe_result

                if progress:
                    star = "*" if probe_result.is_significant else " "
                    print(f"  L{layer:2d}: {probe_result.accuracy:.3f}{star}")

        return result

    def print_results_table(self, results: StratigraphyResult) -> None:
        """Print formatted results table."""
        layers = sorted({layer for r in results.probes.values() for layer in r.keys()})
        probes = sorted(results.probes.keys())

        # Header
        header = f"{'Probe':<20}"
        for layer in layers:
            header += f" L{layer:<3}"
        print("\n" + "=" * len(header))
        print("PROBE ACCURACY BY LAYER")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Rows
        for probe in probes:
            row = f"{probe:<20}"
            for layer in layers:
                if layer in results.probes[probe]:
                    acc = results.probes[probe][layer].accuracy
                    star = "*" if acc > 0.85 else " "
                    row += f" {acc:.2f}{star}"
                else:
                    row += "  -  "
            print(row)

        print("=" * len(header))
        print("* = accuracy > 0.85")

    def print_stratigraphy(
        self,
        results: StratigraphyResult,
        threshold: float = 0.75,
    ) -> None:
        """Print computational stratigraphy showing feature emergence."""
        emergence = results.get_all_emergence_layers(threshold)

        print("\n" + "=" * 60)
        print("COMPUTATIONAL STRATIGRAPHY")
        print(f"Model: {results.model_id}")
        print(f"Threshold: {threshold:.0%}")
        print("=" * 60)

        # Group by category
        by_category: dict[str, list[tuple[str, int | None]]] = {}

        for name, layer in emergence.items():
            cat = self.datasets[name].category if name in self.datasets else "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, layer))

        # Print by category
        for cat, items in sorted(by_category.items()):
            print(f"\n{cat.upper()}:")
            items_sorted = sorted(items, key=lambda x: x[1] if x[1] is not None else 999)
            for name, layer in items_sorted:
                layer_str = f"L{layer}" if layer is not None else "Never"
                desc = self.datasets[name].description if name in self.datasets else ""
                print(f"  {layer_str:<6}: {name}")
                if desc:
                    print(f"          {desc}")

        print("\n" + "=" * 60)


# =============================================================================
# Pre-built probe datasets for common analysis tasks
# =============================================================================


def create_arithmetic_probe() -> ProbeDataset:
    """Create probe for detecting arithmetic vs retrieval mode."""
    return ProbeDataset(
        name="arithmetic_mode",
        description="Detect arithmetic computation vs fact retrieval",
        category="computation",
        label_names=["retrieval", "arithmetic"],
        prompts=[
            # Arithmetic (label=1)
            "6 * 7 =",
            "3 + 5 =",
            "12 - 4 =",
            "8 * 9 =",
            "15 + 27 =",
            "100 - 37 =",
            "7 * 8 =",
            "25 + 17 =",
            "156 + 287 =",
            "23 * 17 =",
            # Retrieval (label=0)
            "The capital of France is",
            "The Eiffel Tower is in",
            "Water boils at",
            "Shakespeare wrote",
            "The largest planet is",
            "Oxygen's atomic number is",
            "The Mona Lisa was painted by",
            "Newton discovered",
            "DNA stands for",
            "The speed of light is",
        ],
        labels=[1] * 10 + [0] * 10,
    )


def create_code_trace_probe() -> ProbeDataset:
    """Create probe for detecting code tracing mode."""
    return ProbeDataset(
        name="code_trace",
        description="Detect code execution/tracing vs code discussion",
        category="computation",
        label_names=["discussion", "trace"],
        prompts=[
            # Trace (label=1)
            "x = 5\ny = 3\nprint(x + y)  # outputs:",
            "a = 10\nb = a * 2\nprint(b)  # prints:",
            "items = [1, 2, 3]\nprint(len(items))  # outputs:",
            "s = 'hello'\nprint(s.upper())  # outputs:",
            "x = 7\nx = x + 3\nprint(x)  # prints:",
            # Discussion (label=0)
            "What does the print function do in Python?",
            "Explain how for loops work",
            "What is the difference between list and tuple?",
            "How do you define a function in Python?",
            "What are Python decorators?",
        ],
        labels=[1] * 5 + [0] * 5,
    )


def create_factual_consistency_probe() -> ProbeDataset:
    """Create probe for detecting context-knowledge consistency."""
    return ProbeDataset(
        name="factual_consistency",
        description="Detect when context contradicts parametric knowledge",
        category="factual",
        label_names=["contradiction", "consistent"],
        prompts=[
            # Consistent (label=1)
            "The Eiffel Tower is in Paris. The Eiffel Tower is in",
            "Tokyo is in Japan. Tokyo is located in",
            "Water is H2O. Water is made of",
            "Shakespeare was English. Shakespeare was from",
            "The Earth orbits the Sun. The Earth orbits",
            # Contradictory (label=0)
            "The Eiffel Tower is in London. The Eiffel Tower is in",
            "Tokyo is in Brazil. Tokyo is located in",
            "Water is made of iron. Water is composed of",
            "Shakespeare was Chinese. Shakespeare was from",
            "The Earth orbits Mars. The Earth orbits",
        ],
        labels=[1] * 5 + [0] * 5,
    )


def create_tool_decision_probe() -> ProbeDataset:
    """Create probe for detecting tool-calling decision."""
    return ProbeDataset(
        name="tool_decision",
        description="Detect when model should call an external tool",
        category="decision",
        label_names=["no_tool", "tool"],
        prompts=[
            # Needs tool (label=1)
            "What's the weather in Tokyo?",
            "Send an email to John",
            "Search for Italian restaurants nearby",
            "Set a timer for 5 minutes",
            "Create a meeting for tomorrow at 3pm",
            # No tool (label=0)
            "What is the capital of France?",
            "Explain quantum computing",
            "Write a haiku about the ocean",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
        ],
        labels=[1] * 5 + [0] * 5,
    )


def create_suppression_probe() -> ProbeDataset:
    """Create probe for detecting computation suppression.

    This is specifically for the Gemma alignment circuit analysis -
    detecting when the model is in "suppress internal computation" mode.
    """
    return ProbeDataset(
        name="suppression_mode",
        description="Detect when model suppresses internal computation for delegation",
        category="alignment",
        label_names=["compute", "suppress"],
        prompts=[
            # Should suppress (label=1) - complex arithmetic
            "156 + 287 =",
            "324 - 189 =",
            "23 * 17 =",
            "789 + 456 =",
            "99 * 11 =",
            # Should compute (label=0) - simple retrieval
            "The capital of France is",
            "Water is made of",
            "The largest planet is",
            "2 + 2 =",  # Simple enough to compute
            "10 - 5 =",
        ],
        labels=[1] * 5 + [0] * 5,
    )


def get_default_probe_datasets() -> dict[str, ProbeDataset]:
    """Get all default probe datasets."""
    return {
        "arithmetic_mode": create_arithmetic_probe(),
        "code_trace": create_code_trace_probe(),
        "factual_consistency": create_factual_consistency_probe(),
        "tool_decision": create_tool_decision_probe(),
        "suppression_mode": create_suppression_probe(),
    }
