"""Classifier service for CLI commands.

This module provides services for multi-class classifier training on activations.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassifierConfig(BaseModel):
    """Configuration for classifier training."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    categories: dict[str, list[str]] = Field(..., description="Category -> prompts mapping")
    layers: list[int] | None = Field(default=None, description="Target layers")
    all_layers: bool = Field(default=False, description="Use all layers")
    layer_depth_ratio: float | None = Field(default=None, description="Layer depth ratio")
    max_iter: int = Field(default=1000, description="Max iterations")
    random_seed: int = Field(default=42, description="Random seed")
    bar_width: int = Field(default=50, description="Display bar width")


class ClassifierResult(BaseModel):
    """Result of classifier training."""

    model_config = ConfigDict(frozen=True)

    layer_results: list[dict[str, Any]] = Field(default_factory=list)
    best_layer: int | None = Field(default=None)
    best_accuracy: float = Field(default=0.0)
    model_id: str = Field(default="")
    categories: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CLASSIFIER TRAINING RESULTS",
            f"{'=' * 70}",
            f"Model: {self.model_id}",
            f"Categories: {', '.join(self.categories)}",
            "",
            f"{'Layer':<8} {'Accuracy':<12} {'F1-Macro':<12}",
            "-" * 40,
        ]

        for r in self.layer_results:
            lines.append(f"{r['layer']:<8} {r['accuracy']:<12.3f} {r.get('f1_macro', 0):<12.3f}")

        lines.extend(
            [
                "-" * 40,
                f"\nBest layer: {self.best_layer}",
                f"Best accuracy: {self.best_accuracy:.3f}",
            ]
        )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save results to file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class ClassifierService:
    """Service for classifier training."""

    @classmethod
    async def train_and_evaluate(cls, config: ClassifierConfig) -> ClassifierResult:
        """Train and evaluate multi-class classifiers.

        Uses logistic regression to train classifiers that can distinguish
        between multiple categories of prompts.
        """
        import mlx.core as mx
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        from ...models_v2 import load_model
        from ..accessor import ModelAccessor

        # Load model using framework loader
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer
        model_config = load_result.config

        accessor = ModelAccessor(model=model, config=model_config)
        num_layers = accessor.num_layers

        def get_all_hidden_states(prompt: str) -> list[np.ndarray]:
            """Get hidden state at each layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            h = accessor.embed(input_ids)

            seq_len = input_ids.shape[1]
            mask = accessor.create_causal_mask(seq_len, h.dtype)

            hidden_states = []
            for idx, lyr in enumerate(accessor.layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                hidden_states.append(np.array(h[0, -1, :].tolist()))

            return hidden_states

        # Determine target layers
        if config.all_layers:
            target_layers = list(range(num_layers))
        elif config.layers:
            target_layers = config.layers
        elif config.layer_depth_ratio:
            target_layers = [int(num_layers * config.layer_depth_ratio)]
        else:
            # Default: sample 8 evenly spaced layers
            target_layers = [int(i * num_layers / 8) for i in range(8)]

        # Collect activations
        all_activations = {layer: [] for layer in range(num_layers)}
        all_labels = []
        categories = list(config.categories.keys())

        for category, prompts in config.categories.items():
            for prompt in prompts:
                hiddens = get_all_hidden_states(prompt)
                for layer, h in enumerate(hiddens):
                    all_activations[layer].append(h)
                all_labels.append(category)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(all_labels)

        # Train classifiers at each target layer
        layer_results = []
        best_layer = None
        best_accuracy = 0.0

        for layer in target_layers:
            X = np.array(all_activations[layer])

            # Train logistic regression
            clf = LogisticRegression(
                max_iter=config.max_iter,
                random_state=config.random_seed,
                multi_class="multinomial",
            )

            # Cross-validation
            n_samples = len(y)
            cv_folds = min(5, n_samples)
            if cv_folds >= 2:
                cv_scores = cross_val_score(clf, X, y, cv=cv_folds)
                accuracy = float(np.mean(cv_scores))
            else:
                accuracy = 0.0

            # Fit on full data for F1
            clf.fit(X, y)
            y_pred = clf.predict(X)
            f1_macro = float(f1_score(y, y_pred, average="macro"))

            layer_results.append(
                {
                    "layer": layer,
                    "accuracy": accuracy,
                    "f1_macro": f1_macro,
                }
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

        return ClassifierResult(
            layer_results=layer_results,
            best_layer=best_layer,
            best_accuracy=best_accuracy,
            model_id=config.model,
            categories=categories,
        )


__all__ = [
    "ClassifierConfig",
    "ClassifierResult",
    "ClassifierService",
]
