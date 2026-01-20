"""
GSM-8K Type Probe Training.

Trains a linear probe on layer 4 hidden states to classify
GSM-8K problem types for routing to specialized experts.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


@dataclass
class GSM8KProbe:
    """Trained probe for GSM-8K problem type classification."""
    classifier: LogisticRegression
    label_encoder: LabelEncoder
    layer: int
    accuracy: float
    labels: list[str]

    def predict(self, hidden_state: np.ndarray) -> str:
        """
        Predict problem type from hidden state.

        Args:
            hidden_state: Shape (hidden_dim,) or (1, hidden_dim)

        Returns:
            Problem type string
        """
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        pred_idx = self.classifier.predict(hidden_state)[0]
        return self.label_encoder.inverse_transform([pred_idx])[0]

    def predict_proba(self, hidden_state: np.ndarray) -> dict[str, float]:
        """
        Get probability distribution over types.
        """
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)

        probs = self.classifier.predict_proba(hidden_state)[0]
        return dict(zip(self.labels, probs))

    def save(self, path: str | Path):
        """Save probe to file."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "layer": self.layer,
                "accuracy": self.accuracy,
                "labels": self.labels,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "GSM8KProbe":
        """Load probe from file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            classifier=data["classifier"],
            label_encoder=data["label_encoder"],
            layer=data["layer"],
            accuracy=data["accuracy"],
            labels=data["labels"],
        )


def extract_hidden_states(
    model,
    tokenizer,
    problems: list[str],
    labels: list[str],
    layer: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states from model for probe training.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        problems: List of problem strings
        labels: List of problem type labels
        layer: Which layer to extract from (default: 4)

    Returns:
        (X, y) tuple of hidden states and encoded labels
    """
    import mlx.core as mx

    hidden_states = []

    for problem in problems:
        # Tokenize
        tokens = tokenizer.encode(problem)
        x = mx.array([tokens])

        # Forward pass with hidden state capture
        # This assumes model has a method to get hidden states
        # Adjust based on actual model architecture
        if hasattr(model, "get_hidden_states"):
            all_hidden = model.get_hidden_states(x)
            h = all_hidden[layer]  # Get specific layer
        else:
            # Fallback: run forward and extract from intermediate
            output = model(x, return_hidden_states=True)
            if hasattr(output, "hidden_states"):
                h = output.hidden_states[layer]
            else:
                # Last resort: use output
                h = output.logits if hasattr(output, "logits") else output

        # Take last token's hidden state
        h = np.array(h[0, -1, :])  # Shape: (hidden_dim,)
        hidden_states.append(h)

    X = np.vstack(hidden_states)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return X, y, le


def train_gsm8k_probe(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    layer: int = 4,
    test_size: float = 0.2,
) -> GSM8KProbe:
    """
    Train a linear probe on extracted hidden states.

    Args:
        X: Hidden states, shape (n_samples, hidden_dim)
        y: Encoded labels, shape (n_samples,)
        label_encoder: Label encoder used for y
        layer: Layer number (for metadata)
        test_size: Fraction for test split

    Returns:
        Trained GSM8KProbe
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Train logistic regression
    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
    ))

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\nCross-validation: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    return GSM8KProbe(
        classifier=clf,
        label_encoder=label_encoder,
        layer=layer,
        accuracy=accuracy,
        labels=list(label_encoder.classes_),
    )


def train_from_annotations(
    model,
    tokenizer,
    annotations_path: str | Path,
    layer: int = 4,
) -> GSM8KProbe:
    """
    Train probe from annotated GSM-8K problems.

    Args:
        model: Language model
        tokenizer: Tokenizer
        annotations_path: Path to annotated problems JSON
        layer: Layer to extract from

    Returns:
        Trained GSM8KProbe
    """
    annotations_path = Path(annotations_path)

    with open(annotations_path) as f:
        annotations = json.load(f)

    problems = [a["problem"] for a in annotations]
    labels = [a["primary_type"] for a in annotations]

    print(f"Loaded {len(problems)} annotated problems")
    print(f"Types: {set(labels)}")

    # Extract hidden states
    print(f"\nExtracting hidden states from layer {layer}...")
    X, y, le = extract_hidden_states(model, tokenizer, problems, labels, layer)

    print(f"Hidden state shape: {X.shape}")

    # Train probe
    print("\nTraining probe...")
    probe = train_gsm8k_probe(X, y, le, layer)

    return probe


def layer_sweep(
    X_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[int, float]:
    """
    Sweep across layers to find best classification layer.

    Args:
        X_by_layer: Dict mapping layer number to hidden states
        y: Labels
        label_encoder: Label encoder

    Returns:
        Dict mapping layer number to accuracy
    """
    results = {}

    for layer, X in sorted(X_by_layer.items()):
        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        scores = cross_val_score(clf, X, y, cv=5)
        results[layer] = scores.mean()
        print(f"Layer {layer:2d}: {scores.mean():.2%} ± {scores.std():.2%}")

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer} ({results[best_layer]:.2%})")

    return results


if __name__ == "__main__":
    print("GSM-8K Type Probe Training")
    print("=" * 60)

    # Demo with synthetic data
    print("\nDemo with synthetic hidden states...")

    # Simulate 8 problem types, 20 examples each
    np.random.seed(42)
    n_per_type = 20
    hidden_dim = 256

    types = [
        "arithmetic_chain", "rate_ratio", "allocation", "comparison",
        "scheduling_time", "geometry", "percentage", "multi_constraint"
    ]

    # Create separable synthetic data
    X = []
    labels = []
    for i, ptype in enumerate(types):
        # Each type gets a different region in hidden space
        center = np.zeros(hidden_dim)
        center[i * 32:(i + 1) * 32] = 1.0  # Different active region per type
        samples = center + np.random.randn(n_per_type, hidden_dim) * 0.3
        X.extend(samples)
        labels.extend([ptype] * n_per_type)

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Synthetic data: {X.shape}")
    print(f"Types: {list(le.classes_)}")

    # Train probe
    probe = train_gsm8k_probe(X, y, le, layer=4)

    # Test single prediction
    print("\nTest prediction:")
    test_hidden = np.random.randn(hidden_dim)
    pred_type = probe.predict(test_hidden)
    probs = probe.predict_proba(test_hidden)
    print(f"  Predicted type: {pred_type}")
    print(f"  Top 3 probabilities: {sorted(probs.items(), key=lambda x: -x[1])[:3]}")

    # Save probe
    probe_path = Path(__file__).parent.parent / "results" / "gsm8k_probe_demo.pkl"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(probe_path)
    print(f"\nProbe saved to: {probe_path}")
