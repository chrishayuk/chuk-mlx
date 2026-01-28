#!/usr/bin/env python3
"""
Tool Selection Probe

Trains linear probes to classify which tool will be called,
given that the model will call a tool.

Research Question: How does the model select which tool to use?
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class ToolSelectionProbe:
    """
    Probe for classifying which tool will be called.

    Labels:
        0 = calculator
        1 = search
        2 = code_exec
        3 = get_weather
    """

    TOOL_LABELS = {
        "calculator": 0,
        "search": 1,
        "code_exec": 2,
        "get_weather": 3,
    }

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.probe_layers = config["tool_selection"]["probe_layers"]
        self.train_samples = config["tool_selection"]["train_samples"]
        self.test_samples = config["tool_selection"]["test_samples"]

        self.model = None
        self.tokenizer = None
        self.probes: dict[int, LogisticRegression] = {}

    def load_model(self):
        """Load the model and tokenizer."""
        from mlx_lm import load

        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        logger.info("Model loaded successfully")

    def prepare_prompts(self) -> tuple[list[str], list[int], list[str], list[int]]:
        """
        Prepare training and test prompts with tool labels.

        Returns:
            train_prompts, train_labels, test_prompts, test_labels
        """
        prompts_config = self.config["prompts"]["tool_required"]

        all_prompts = []
        all_labels = []

        for tool_name, prompts in prompts_config.items():
            if tool_name not in self.TOOL_LABELS:
                continue
            label = self.TOOL_LABELS[tool_name]
            for prompt in prompts:
                all_prompts.append(prompt)
                all_labels.append(label)

        # Shuffle
        np.random.seed(42)
        perm = np.random.permutation(len(all_prompts))
        all_prompts = [all_prompts[i] for i in perm]
        all_labels = [all_labels[i] for i in perm]

        # Split train/test
        n_train = min(self.train_samples, int(len(all_prompts) * 0.8))
        n_test = min(self.test_samples, len(all_prompts) - n_train)

        train_prompts = all_prompts[:n_train]
        train_labels = all_labels[:n_train]
        test_prompts = all_prompts[n_train:n_train + n_test]
        test_labels = all_labels[n_train:n_train + n_test]

        return train_prompts, train_labels, test_prompts, test_labels

    def extract_hidden_states(
        self,
        prompts: list[str],
        layer: int
    ) -> np.ndarray:
        """
        Extract hidden states at a specific layer for all prompts.

        Args:
            prompts: List of input prompts
            layer: Layer index to extract from

        Returns:
            Array of shape (n_prompts, hidden_dim)
        """
        hidden_states = []

        # Get model components
        if hasattr(self.model, 'model'):
            embed_tokens = self.model.model.embed_tokens
            layers = self.model.model.layers
        else:
            embed_tokens = self.model.embed_tokens
            layers = self.model.layers

        for prompt in prompts:
            # Tokenize
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Get embeddings
            h = embed_tokens(input_ids)
            batch_size, seq_len, hidden_dim = h.shape

            # Create causal mask
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

            # Forward through layers up to target
            for i, layer_module in enumerate(layers):
                h = layer_module(h, mask=mask)
                if i == layer:
                    break

            mx.eval(h)

            # Take the last token's hidden state
            h_last = h[0, -1, :]
            hidden_states.append(np.array(h_last.astype(mx.float32)))

            gc.collect()

        return np.stack(hidden_states)

    def train_probe(
        self,
        layer: int,
        train_features: np.ndarray,
        train_labels: list[int]
    ) -> LogisticRegression:
        """Train a multi-class logistic regression probe."""
        probe = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class="multinomial"
        )
        probe.fit(train_features, train_labels)
        self.probes[layer] = probe
        return probe

    def evaluate_probe(
        self,
        layer: int,
        test_features: np.ndarray,
        test_labels: list[int]
    ) -> dict[str, Any]:
        """Evaluate a trained probe."""
        probe = self.probes[layer]
        predictions = probe.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        tool_names = list(self.TOOL_LABELS.keys())

        # Per-tool accuracy
        per_tool_acc = {}
        for tool, i in self.TOOL_LABELS.items():
            if i < len(cm) and cm[i].sum() > 0:
                per_tool_acc[tool] = float(cm[i, i] / cm[i].sum())
            else:
                per_tool_acc[tool] = 0.0

        return {
            "layer": layer,
            "accuracy": float(accuracy),
            "predictions": predictions.tolist(),
            "confusion_matrix": cm.tolist(),
            "tool_names": tool_names,
            "per_tool_accuracy": per_tool_acc,
        }

    def analyze_decision_boundary(
        self,
        layer: int,
        test_features: np.ndarray,
        test_labels: list[int]
    ) -> dict[str, Any]:
        """
        Analyze the decision boundary geometry.
        """
        probe = self.probes[layer]
        weights = probe.coef_  # Shape: (n_tools, hidden_dim)

        # Weight norms
        weight_norms = np.linalg.norm(weights, axis=1)

        # Pairwise distances
        n_tools = min(len(self.TOOL_LABELS), weights.shape[0])
        pairwise_distances = {}
        tool_names = list(self.TOOL_LABELS.keys())
        for i in range(n_tools):
            for j in range(i + 1, n_tools):
                dist = np.linalg.norm(weights[i] - weights[j])
                pairwise_distances[f"{tool_names[i]}_vs_{tool_names[j]}"] = float(dist)

        # Confidence scores
        probas = probe.predict_proba(test_features)
        avg_confidence = probas.max(axis=1).mean()

        return {
            "weight_norms": {
                tool: float(weight_norms[i])
                for tool, i in self.TOOL_LABELS.items()
                if i < len(weight_norms)
            },
            "pairwise_distances": pairwise_distances,
            "avg_confidence": float(avg_confidence),
        }

    def run(self) -> dict[str, Any]:
        """
        Run the full tool selection probing experiment.

        Returns:
            Dictionary with results for each layer and summary statistics
        """
        logger.info("Starting tool selection probe experiment")

        # Load model
        self.load_model()

        # Prepare data
        train_prompts, train_labels, test_prompts, test_labels = self.prepare_prompts()
        logger.info(f"Train: {len(train_prompts)} prompts, Test: {len(test_prompts)} prompts")

        if len(train_prompts) == 0:
            return {"error": "No training prompts available"}

        # Label distribution
        train_dist = {tool: train_labels.count(i) for tool, i in self.TOOL_LABELS.items()}
        logger.info(f"Train distribution: {train_dist}")

        results = {
            "layers": {},
            "train_size": len(train_prompts),
            "test_size": len(test_prompts),
            "train_distribution": train_dist,
        }

        best_accuracy = 0
        best_layer = None

        for layer in self.probe_layers:
            logger.info(f"Processing layer {layer}...")

            # Extract features
            train_features = self.extract_hidden_states(train_prompts, layer)

            if len(test_prompts) > 0:
                test_features = self.extract_hidden_states(test_prompts, layer)
            else:
                test_features = train_features
                test_labels = train_labels

            # Train probe
            self.train_probe(layer, train_features, train_labels)

            # Evaluate
            layer_results = self.evaluate_probe(layer, test_features, test_labels)

            # Analyze decision boundary
            boundary_analysis = self.analyze_decision_boundary(
                layer, test_features, test_labels
            )
            layer_results["boundary_analysis"] = boundary_analysis

            results["layers"][layer] = layer_results

            accuracy = layer_results["accuracy"]
            logger.info(f"Layer {layer}: {accuracy:.1%} accuracy")

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

            gc.collect()

        # Summary
        results["best_layer"] = best_layer
        results["best_accuracy"] = best_accuracy

        # Tool-specific insights
        if best_layer is not None and best_layer in results["layers"]:
            best_results = results["layers"][best_layer]
            results["tool_rankings"] = best_results.get("per_tool_accuracy", {})

        logger.info(f"Best layer: L{best_layer} with {best_accuracy:.1%}")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    probe = ToolSelectionProbe(config)
    results = probe.run()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
