#!/usr/bin/env python3
"""
Tool Intent Probe

Trains linear probes to detect "will call tool" vs "will answer directly"
at each layer of the model.

Research Question: At which layer can we reliably detect tool intent?
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


class ToolIntentProbe:
    """
    Probe for detecting tool calling intent at each layer.

    Labels:
        0 = will answer directly (no tool)
        1 = will call tool
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["primary"]
        self.probe_layers = config["tool_intent"]["probe_layers"]
        self.train_samples = config["tool_intent"]["train_samples"]
        self.test_samples = config["tool_intent"]["test_samples"]
        self.accuracy_threshold = config["tool_intent"]["accuracy_threshold"]

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
        Prepare training and test prompts with labels.

        Returns:
            train_prompts, train_labels, test_prompts, test_labels
        """
        prompts_config = self.config["prompts"]

        # Collect tool-required prompts (label=1)
        tool_prompts = []
        for tool_name, prompts in prompts_config["tool_required"].items():
            tool_prompts.extend(prompts)

        # Collect direct-answer prompts (label=0)
        direct_prompts = prompts_config["direct_answer"]

        # Balance the dataset
        n_per_class_train = min(self.train_samples // 2, len(tool_prompts), len(direct_prompts))
        n_per_class_test = min(self.test_samples // 2,
                               len(tool_prompts) - n_per_class_train,
                               len(direct_prompts) - n_per_class_train)

        # Shuffle and split
        np.random.seed(42)
        tool_idx = np.random.permutation(len(tool_prompts))
        direct_idx = np.random.permutation(len(direct_prompts))

        train_prompts = (
            [tool_prompts[i] for i in tool_idx[:n_per_class_train]] +
            [direct_prompts[i % len(direct_prompts)] for i in range(n_per_class_train)]
        )
        train_labels = [1] * n_per_class_train + [0] * n_per_class_train

        test_prompts = (
            [tool_prompts[i] for i in tool_idx[n_per_class_train:n_per_class_train + n_per_class_test]] +
            [direct_prompts[i % len(direct_prompts)] for i in range(n_per_class_test)]
        )
        test_labels = [1] * n_per_class_test + [0] * n_per_class_test

        # Shuffle
        if train_prompts:
            train_perm = np.random.permutation(len(train_prompts))
            train_prompts = [train_prompts[i] for i in train_perm]
            train_labels = [train_labels[i] for i in train_perm]

        if test_prompts:
            test_perm = np.random.permutation(len(test_prompts))
            test_prompts = [test_prompts[i] for i in test_perm]
            test_labels = [test_labels[i] for i in test_perm]

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
            h_last = h[0, -1, :]  # Shape: (hidden_dim,)
            hidden_states.append(np.array(h_last.astype(mx.float32)))

            gc.collect()

        return np.stack(hidden_states)

    def train_probe(
        self,
        layer: int,
        train_features: np.ndarray,
        train_labels: list[int]
    ) -> LogisticRegression:
        """Train a logistic regression probe for a specific layer."""
        probe = LogisticRegression(max_iter=1000, random_state=42)
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

        return {
            "layer": layer,
            "accuracy": float(accuracy),
            "predictions": predictions.tolist(),
            "classification_report": classification_report(
                test_labels, predictions, output_dict=True
            )
        }

    def run(self) -> dict[str, Any]:
        """
        Run the full tool intent probing experiment.

        Returns:
            Dictionary with results for each layer and summary statistics
        """
        logger.info("Starting tool intent probe experiment")

        # Load model
        self.load_model()

        # Prepare data
        train_prompts, train_labels, test_prompts, test_labels = self.prepare_prompts()
        logger.info(f"Train: {len(train_prompts)} prompts, Test: {len(test_prompts)} prompts")

        if len(train_prompts) == 0:
            return {"error": "No training prompts available"}

        results = {
            "layers": {},
            "train_size": len(train_prompts),
            "test_size": len(test_prompts),
        }

        best_accuracy = 0
        best_layer = None
        first_reliable_layer = None

        for layer in self.probe_layers:
            logger.info(f"Processing layer {layer}...")

            # Extract features
            train_features = self.extract_hidden_states(train_prompts, layer)

            if len(test_prompts) > 0:
                test_features = self.extract_hidden_states(test_prompts, layer)
            else:
                # Use train for test if no test data
                test_features = train_features
                test_labels = train_labels

            # Train probe
            self.train_probe(layer, train_features, train_labels)

            # Evaluate
            layer_results = self.evaluate_probe(layer, test_features, test_labels)
            results["layers"][layer] = layer_results

            accuracy = layer_results["accuracy"]
            logger.info(f"Layer {layer}: {accuracy:.1%} accuracy")

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer

            # Track first reliable layer
            if first_reliable_layer is None and accuracy >= self.accuracy_threshold:
                first_reliable_layer = layer

            gc.collect()

        # Summary
        results["best_layer"] = best_layer
        results["best_accuracy"] = best_accuracy
        results["first_reliable_layer"] = first_reliable_layer
        results["accuracy_threshold"] = self.accuracy_threshold

        logger.info(f"Best layer: L{best_layer} with {best_accuracy:.1%}")
        logger.info(f"First reliable layer (>{self.accuracy_threshold:.0%}): L{first_reliable_layer}")

        return results


def main():
    """Run as standalone script."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    probe = ToolIntentProbe(config)
    results = probe.run()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
