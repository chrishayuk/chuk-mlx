"""
Format Gate Detection Experiment

Research Question:
Can we detect WHERE the model decides "symbolic → direct" vs "semantic → CoT"?

Hypothesis:
There's a format classifier at early layers (L2-4) that gates generation mode.
This should be detectable via linear probe.

Tests:
1. FORMAT PROBE: Can we classify symbolic vs semantic at each layer?
2. LAYER EMERGENCE: At what layer does format become classifiable?
3. BASE vs INSTRUCT: Does instruction tuning change format encoding?
4. GENERATION CORRELATION: Does probe prediction match actual output format?
5. STEERING: Can we flip generation mode by steering format representation?
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class FormatProbeResult:
    """Result for a single layer's format probe."""
    layer: int
    train_accuracy: float
    test_accuracy: float
    symbolic_confidence: float  # avg confidence on symbolic inputs
    semantic_confidence: float  # avg confidence on semantic inputs


@dataclass
class GenerationResult:
    """Result for generation analysis."""
    prompt: str
    expected_format: str  # "direct" or "cot"
    probe_prediction: str
    probe_confidence: float
    generated_text: str
    actual_format: str  # detected from generation
    prediction_correct: bool


class FormatGateExperiment:
    """
    Detect the format gate that decides CoT vs direct generation.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.probes = {}  # layer -> (W, b)

    def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        logger.info("Starting Format Gate Detection experiment")

        # Load model
        self._load_model()

        # 1. Train format probes at each layer
        probe_results = self._train_format_probes()

        # 2. Find emergence layer
        emergence_layer = self._find_emergence_layer(probe_results)

        # 3. Test generation correlation
        generation_results = self._test_generation_correlation()

        # 4. Steering experiment (if enabled)
        steering_results = None
        if self.config.get("parameters", {}).get("run_steering", False):
            steering_results = self._run_steering_experiment()

        # Aggregate and save
        results = self._aggregate_results(
            probe_results, emergence_layer, generation_results, steering_results
        )
        self._save_results(results)

        return results

    def _load_model(self):
        """Load the model and tokenizer."""
        from chuk_lazarus.models_v2.loader import load_model

        model_name = self.config["model"]
        logger.info(f"Loading model: {model_name}")

        loaded = load_model(model_name)
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        # Get model info
        self.num_layers = len(self.model.model.layers)
        # Hidden dim will be inferred from first hidden state
        self.hidden_dim = None
        logger.info(f"Model loaded: {self.num_layers} layers")

    def _get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at specified layer for last token."""
        # Use chat template for instruct models
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            if "Instruct" in self.config["model"]:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                tokens = self.tokenizer(formatted, return_tensors="np")
            else:
                tokens = self.tokenizer(prompt, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])

        output = self.model(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[layer]  # [1, seq_len, hidden_dim]

        return hidden[0, -1, :]  # last token

    def _train_format_probes(self) -> list[FormatProbeResult]:
        """Train format classification probes at each layer."""
        logger.info("Training format probes at each layer")

        # Training data: symbolic vs semantic
        train_data = self.config["parameters"]["train_data"]
        test_data = self.config["parameters"]["test_data"]

        layers_to_test = self.config["parameters"]["layers_to_test"]
        results = []

        for layer in layers_to_test:
            if layer >= self.num_layers:
                continue

            logger.info(f"Training probe at layer {layer}")

            # Extract hidden states for training
            train_hiddens = []
            train_labels = []

            for item in train_data:
                h = self._get_hidden_state(item["prompt"], layer)
                train_hiddens.append(h)
                train_labels.append(0 if item["format"] == "symbolic" else 1)

            X_train = mx.stack(train_hiddens)
            y_train = mx.array(train_labels)

            # Train probe
            W, b = self._train_probe(X_train, y_train)
            self.probes[layer] = (W, b)

            # Evaluate on training data
            train_logits = X_train @ W + b
            train_preds = mx.argmax(train_logits, axis=-1)
            train_acc = mx.mean(train_preds == y_train).item()

            # Evaluate on test data
            test_hiddens = []
            test_labels = []
            for item in test_data:
                h = self._get_hidden_state(item["prompt"], layer)
                test_hiddens.append(h)
                test_labels.append(0 if item["format"] == "symbolic" else 1)

            X_test = mx.stack(test_hiddens)
            y_test = mx.array(test_labels)

            test_logits = X_test @ W + b
            test_preds = mx.argmax(test_logits, axis=-1)
            test_acc = mx.mean(test_preds == y_test).item()

            # Get confidence scores
            test_probs = mx.softmax(test_logits, axis=-1)

            # Calculate confidence for each class using masks
            symbolic_mask = (y_test == 0).astype(mx.float32)
            semantic_mask = (y_test == 1).astype(mx.float32)

            # Weighted mean using masks
            symbolic_count = mx.sum(symbolic_mask).item()
            semantic_count = mx.sum(semantic_mask).item()

            if symbolic_count > 0:
                symbolic_conf = (mx.sum(test_probs[:, 0] * symbolic_mask) / symbolic_count).item()
            else:
                symbolic_conf = 0.0

            if semantic_count > 0:
                semantic_conf = (mx.sum(test_probs[:, 1] * semantic_mask) / semantic_count).item()
            else:
                semantic_conf = 0.0

            result = FormatProbeResult(
                layer=layer,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                symbolic_confidence=symbolic_conf,
                semantic_confidence=semantic_conf,
            )
            results.append(result)

            logger.info(f"  Layer {layer}: train={train_acc:.1%}, test={test_acc:.1%}")

        return results

    def _train_probe(self, X: mx.array, y: mx.array, epochs: int = 100) -> tuple[mx.array, mx.array]:
        """Train a linear probe using gradient descent."""
        hidden_dim = X.shape[1]
        num_classes = 2

        W = mx.random.normal((hidden_dim, num_classes)) * 0.01
        b = mx.zeros((num_classes,))

        lr = 0.1

        for _ in range(epochs):
            logits = X @ W + b
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            # Gradient computation
            probs = mx.softmax(logits, axis=-1)
            grad_logits = probs
            grad_logits = grad_logits.at[mx.arange(len(y)), y].add(-1)
            grad_logits = grad_logits / len(y)

            grad_W = X.T @ grad_logits
            grad_b = mx.sum(grad_logits, axis=0)

            W = W - lr * grad_W
            b = b - lr * grad_b

            mx.eval(W, b)

        return W, b

    def _find_emergence_layer(self, probe_results: list[FormatProbeResult]) -> dict[str, Any]:
        """Find where format classification becomes reliable."""
        threshold = self.config["parameters"].get("emergence_threshold", 0.9)

        emergence_layer = None
        for result in probe_results:
            if result.test_accuracy >= threshold:
                emergence_layer = result.layer
                break

        # Also find peak layer
        peak_result = max(probe_results, key=lambda r: r.test_accuracy)

        return {
            "emergence_layer": emergence_layer,
            "emergence_threshold": threshold,
            "peak_layer": peak_result.layer,
            "peak_accuracy": peak_result.test_accuracy,
        }

    def _test_generation_correlation(self) -> list[GenerationResult]:
        """Test if probe predictions correlate with actual generation format."""
        logger.info("Testing generation correlation")

        probe_layer = self.config["parameters"].get("probe_layer_for_generation", 4)

        if probe_layer not in self.probes:
            logger.warning(f"No probe trained at layer {probe_layer}, skipping generation test")
            return []

        W, b = self.probes[probe_layer]
        test_prompts = self.config["parameters"]["generation_test"]

        results = []
        for item in test_prompts:
            prompt = item["prompt"]
            expected_format = item["expected_format"]

            # Get probe prediction
            h = self._get_hidden_state(prompt, probe_layer)
            logits = h @ W + b
            probs = mx.softmax(logits)
            pred_idx = mx.argmax(logits).item()
            pred_format = "symbolic" if pred_idx == 0 else "semantic"
            confidence = probs[pred_idx].item()

            # Map to generation expectation
            probe_prediction = "direct" if pred_format == "symbolic" else "cot"

            # Generate actual output
            generated = self._generate(prompt)

            # Detect actual format from generation
            actual_format = self._detect_generation_format(generated)

            results.append(GenerationResult(
                prompt=prompt,
                expected_format=expected_format,
                probe_prediction=probe_prediction,
                probe_confidence=confidence,
                generated_text=generated,
                actual_format=actual_format,
                prediction_correct=(probe_prediction == actual_format),
            ))

            logger.info(f"  '{prompt[:30]}...' → probe: {probe_prediction}, actual: {actual_format}")

        return results

    def _generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response from model."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            if "Instruct" in self.config["model"]:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                tokens = self.tokenizer(formatted, return_tensors="np")
            else:
                tokens = self.tokenizer(prompt, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _detect_generation_format(self, text: str) -> str:
        """Detect if generation is CoT or direct answer."""
        # CoT indicators
        cot_patterns = [
            r"to find",
            r"we need to",
            r"let's",
            r"first,",
            r"step \d",
            r"therefore",
            r"so,",
            r"this means",
            r"calculate",
            r"the answer is",
            r"\d+\s*[\+\-\*\/]\s*\d+\s*=",  # equation in text
        ]

        text_lower = text.lower()

        # Check for CoT patterns
        cot_matches = sum(1 for p in cot_patterns if re.search(p, text_lower))

        # Check text length (CoT tends to be longer)
        word_count = len(text.split())

        # Heuristic: CoT if multiple indicators OR long response
        if cot_matches >= 2 or word_count > 20:
            return "cot"
        else:
            return "direct"

    def _run_steering_experiment(self) -> dict[str, Any]:
        """Test if we can flip generation mode by steering format representation."""
        logger.info("Running steering experiment")

        steering_layer = self.config["parameters"].get("steering_layer", 4)
        steering_strength = self.config["parameters"].get("steering_strength", 2.0)

        # Compute format direction vector
        symbolic_prompts = [item["prompt"] for item in self.config["parameters"]["train_data"]
                          if item["format"] == "symbolic"]
        semantic_prompts = [item["prompt"] for item in self.config["parameters"]["train_data"]
                          if item["format"] == "semantic"]

        symbolic_hiddens = [self._get_hidden_state(p, steering_layer) for p in symbolic_prompts]
        semantic_hiddens = [self._get_hidden_state(p, steering_layer) for p in semantic_prompts]

        symbolic_mean = mx.mean(mx.stack(symbolic_hiddens), axis=0)
        semantic_mean = mx.mean(mx.stack(semantic_hiddens), axis=0)

        # Direction: semantic - symbolic (steering toward CoT)
        cot_direction = semantic_mean - symbolic_mean
        cot_direction = cot_direction / (mx.linalg.norm(cot_direction) + 1e-8)

        # Test steering on symbolic inputs
        results = []
        test_symbolic = [
            "7 * 8 = ",
            "45 + 45 = ",
            "100 - 37 = ",
        ]

        for prompt in test_symbolic:
            # Generate without steering
            normal_output = self._generate(prompt)
            normal_format = self._detect_generation_format(normal_output)

            # Generate with steering (would need model modification)
            # For now, just report the direction magnitude
            h = self._get_hidden_state(prompt, steering_layer)
            projection = mx.sum(h * cot_direction).item()

            results.append({
                "prompt": prompt,
                "normal_output": normal_output[:100],
                "normal_format": normal_format,
                "cot_direction_projection": projection,
            })

        return {
            "steering_layer": steering_layer,
            "steering_strength": steering_strength,
            "cot_direction_norm": mx.linalg.norm(cot_direction).item(),
            "test_results": results,
        }

    def _aggregate_results(
        self,
        probe_results: list[FormatProbeResult],
        emergence: dict[str, Any],
        generation_results: list[GenerationResult],
        steering_results: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Aggregate all results."""

        # Probe accuracy by layer
        probe_by_layer = {
            r.layer: {
                "train_accuracy": r.train_accuracy,
                "test_accuracy": r.test_accuracy,
                "symbolic_confidence": r.symbolic_confidence,
                "semantic_confidence": r.semantic_confidence,
            }
            for r in probe_results
        }

        # Generation correlation
        if generation_results:
            correct_predictions = sum(1 for r in generation_results if r.prediction_correct)
            generation_correlation = correct_predictions / len(generation_results)
        else:
            generation_correlation = None

        return {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "num_layers": self.num_layers,
            "findings": {
                "format_classification": {
                    "description": "Can probe classify symbolic vs semantic?",
                    "probe_accuracy_by_layer": probe_by_layer,
                    "emergence_layer": emergence["emergence_layer"],
                    "peak_layer": emergence["peak_layer"],
                    "peak_accuracy": emergence["peak_accuracy"],
                },
                "generation_correlation": {
                    "description": "Does format probe predict generation mode?",
                    "correlation": generation_correlation,
                    "details": [
                        {
                            "prompt": r.prompt,
                            "expected": r.expected_format,
                            "probe_prediction": r.probe_prediction,
                            "probe_confidence": r.probe_confidence,
                            "actual_format": r.actual_format,
                            "correct": r.prediction_correct,
                            "generated": r.generated_text[:200],
                        }
                        for r in generation_results
                    ] if generation_results else [],
                },
                "steering": steering_results,
            },
        }

    def _save_results(self, results: dict[str, Any]):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"run_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results: dict[str, Any]):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("FORMAT GATE DETECTION RESULTS")
        print("=" * 60)

        findings = results["findings"]
        fc = findings["format_classification"]

        print(f"\nModel: {results['model']}")
        print(f"Layers: {results['num_layers']}")

        print(f"\n1. FORMAT CLASSIFICATION (symbolic vs semantic)")
        print(f"   Emergence layer: {fc['emergence_layer']} (≥90% accuracy)")
        print(f"   Peak layer: {fc['peak_layer']} ({fc['peak_accuracy']:.1%} accuracy)")
        print(f"\n   Layer-by-layer accuracy:")
        for layer, data in sorted(fc["probe_accuracy_by_layer"].items()):
            bar = "█" * int(data["test_accuracy"] * 20)
            print(f"   L{layer:2d}: {data['test_accuracy']:5.1%} {bar}")

        gc = findings["generation_correlation"]
        if gc["correlation"] is not None:
            print(f"\n2. GENERATION CORRELATION")
            print(f"   Probe predicts generation mode: {gc['correlation']:.1%}")
            for detail in gc["details"]:
                status = "✓" if detail["correct"] else "✗"
                print(f"   {status} '{detail['prompt'][:25]}...' → {detail['actual_format']}")

        if findings.get("steering"):
            print(f"\n3. STEERING POTENTIAL")
            print(f"   CoT direction norm: {findings['steering']['cot_direction_norm']:.3f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment = FormatGateExperiment()
    experiment.run()
