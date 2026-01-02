#!/usr/bin/env python3
"""
Gemma Probe Lens: A learned projection system for decoding intermediate layers.

Unlike standard logit lens (norm(h) @ embed.T), this trains linear probes
at each layer to decode the hidden states into meaningful predictions.

Key insight: Gemma's hidden states contain the answer, but need a learned
projection direction to extract it.

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_probe_lens.py
    uv run python examples/introspection/experiments/model_specific/gemma_probe_lens.py --task multiplication
    uv run python examples/introspection/experiments/model_specific/gemma_probe_lens.py --decode "7 * 8 = "
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class ProbeResult:
    """Result from probing a hidden state."""
    layer_idx: int
    predicted_token: str
    probability: float
    all_probs: dict[str, float]


@dataclass
class LayerProbe:
    """A trained probe for a specific layer."""
    layer_idx: int
    probe: LogisticRegression
    label_encoder: LabelEncoder
    accuracy: float


class GemmaProbeLens:
    """
    Probe-based lens for decoding Gemma intermediate layers.

    Instead of projecting to vocabulary with embed.T, we train
    linear probes at each layer to decode task-specific information.
    """

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None
        self.probes: dict[int, LayerProbe] = {}

    def load_model(self):
        """Load the model."""
        print(f"Loading model: {self.model_id}")

        result = HFLoader.download(self.model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        family_info = get_family_info(family_type)
        self.config = family_info.config_class.from_hf_config(config_data)
        self.model = family_info.model_class(self.config)

        HFLoader.apply_weights_to_model(self.model, model_path, self.config, dtype=DType.BFLOAT16)
        self.tokenizer = HFLoader.load_tokenizer(model_path)

        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        norm = getattr(backbone, "norm", None)

        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
            embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, norm, embed_scale

    def get_all_hidden_states(self, prompt: str) -> list[np.ndarray]:
        """
        Get hidden states from all layers for a prompt.

        Returns list of hidden states, one per layer, at the last token position.
        """
        layers, embed, norm, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        hidden_states = []

        for layer in layers:
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

            # Extract last token position
            h_last = np.array(h[0, -1, :].tolist())
            hidden_states.append(h_last)

        return hidden_states

    def get_hidden_state(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get hidden state from a specific layer."""
        all_states = self.get_all_hidden_states(prompt)
        return all_states[layer_idx]

    def create_multiplication_dataset(self) -> tuple[list[str], list[str], list[str]]:
        """
        Create dataset for multiplication task.

        Returns:
            prompts: List of prompts like "7 * 8 = "
            answers: List of full answers like "56"
            first_digits: List of first digits like "5"
        """
        prompts = []
        answers = []
        first_digits = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompt = f"{a} * {b} = "
                answer = str(a * b)
                first_digit = answer[0]

                prompts.append(prompt)
                answers.append(answer)
                first_digits.append(first_digit)

        return prompts, answers, first_digits

    def create_addition_dataset(self) -> tuple[list[str], list[str], list[str]]:
        """Create dataset for addition task."""
        prompts = []
        answers = []
        first_digits = []

        for a in range(1, 20):
            for b in range(1, 20):
                prompt = f"{a} + {b} = "
                answer = str(a + b)
                first_digit = answer[0]

                prompts.append(prompt)
                answers.append(answer)
                first_digits.append(first_digit)

        return prompts, answers, first_digits

    def train_probe(
        self,
        layer_idx: int,
        prompts: list[str],
        labels: list[str],
        test_split: float = 0.2,
    ) -> LayerProbe:
        """
        Train a probe for a specific layer.

        Args:
            layer_idx: Which layer to probe
            prompts: List of input prompts
            labels: List of target labels
            test_split: Fraction to use for testing

        Returns:
            Trained LayerProbe
        """
        print(f"  Training probe for layer {layer_idx}...")

        # Collect hidden states
        X = []
        for prompt in prompts:
            h = self.get_hidden_state(prompt, layer_idx)
            X.append(h)
        X = np.array(X)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Split data
        n_test = int(len(X) * test_split)
        indices = np.random.permutation(len(X))
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train probe
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_train, y_train)

        # Evaluate
        accuracy = probe.score(X_test, y_test)

        layer_probe = LayerProbe(
            layer_idx=layer_idx,
            probe=probe,
            label_encoder=le,
            accuracy=accuracy,
        )

        self.probes[layer_idx] = layer_probe
        return layer_probe

    def train_all_layers(
        self,
        prompts: list[str],
        labels: list[str],
        layer_step: int = 1,
    ) -> dict[int, LayerProbe]:
        """
        Train probes for all layers (or every nth layer).

        Args:
            prompts: List of input prompts
            labels: List of target labels
            layer_step: Train every nth layer

        Returns:
            Dict mapping layer_idx to LayerProbe
        """
        print(f"\nTraining probes for {self.num_layers} layers (step={layer_step})...")
        print(f"Dataset size: {len(prompts)} examples")

        # Pre-compute all hidden states for efficiency
        print("Collecting hidden states...")
        all_hidden = []
        for i, prompt in enumerate(prompts):
            if i % 20 == 0:
                print(f"  {i}/{len(prompts)}")
            states = self.get_all_hidden_states(prompt)
            all_hidden.append(states)

        # Encode labels once
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Split data
        n_test = max(1, int(len(y) * 0.2))
        indices = np.random.permutation(len(y))
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]

        y_train, y_test = y[train_idx], y[test_idx]

        print("\nTraining probes...")
        results = {}

        for layer_idx in range(0, self.num_layers, layer_step):
            # Extract hidden states for this layer
            X = np.array([all_hidden[i][layer_idx] for i in range(len(all_hidden))])
            X_train, X_test = X[train_idx], X[test_idx]

            # Train
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(X_train, y_train)

            # Evaluate
            accuracy = probe.score(X_test, y_test)

            layer_probe = LayerProbe(
                layer_idx=layer_idx,
                probe=probe,
                label_encoder=le,
                accuracy=accuracy,
            )

            self.probes[layer_idx] = layer_probe
            results[layer_idx] = layer_probe

            print(f"  L{layer_idx:2d}: accuracy = {accuracy:.1%}")

        return results

    def decode(self, prompt: str, layer_idx: Optional[int] = None) -> list[ProbeResult]:
        """
        Decode a prompt using trained probes.

        Args:
            prompt: Input prompt to decode
            layer_idx: Specific layer to decode (or all if None)

        Returns:
            List of ProbeResult for each decoded layer
        """
        if not self.probes:
            raise ValueError("No probes trained. Call train_all_layers first.")

        # Get hidden states
        all_states = self.get_all_hidden_states(prompt)

        results = []
        layers_to_decode = [layer_idx] if layer_idx is not None else sorted(self.probes.keys())

        for idx in layers_to_decode:
            if idx not in self.probes:
                continue

            probe = self.probes[idx]
            h = all_states[idx].reshape(1, -1)

            # Predict
            pred_idx = probe.probe.predict(h)[0]
            pred_token = probe.label_encoder.inverse_transform([pred_idx])[0]

            # Get probabilities
            proba = probe.probe.predict_proba(h)[0]
            all_probs = {
                probe.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }

            results.append(ProbeResult(
                layer_idx=idx,
                predicted_token=pred_token,
                probability=float(proba[pred_idx]),
                all_probs=all_probs,
            ))

        return results

    def decode_full_answer(
        self,
        prompt: str,
        layer_idx: int,
        max_digits: int = 3,
    ) -> tuple[str, float]:
        """
        Decode full answer by training probes for each digit position.

        This is a simplified version - for full answers we need probes
        trained on different digit positions.
        """
        # For now just use the first digit probe
        results = self.decode(prompt, layer_idx)
        if results:
            return results[0].predicted_token, results[0].probability
        return "", 0.0

    def visualize_layer_accuracy(self):
        """Print accuracy by layer."""
        if not self.probes:
            print("No probes trained.")
            return

        print("\n" + "=" * 60)
        print("PROBE ACCURACY BY LAYER")
        print("=" * 60)

        print(f"\n{'Layer':<8} {'Accuracy':<12} {'Bar'}")
        print("-" * 50)

        for idx in sorted(self.probes.keys()):
            probe = self.probes[idx]
            bar_len = int(probe.accuracy * 40)
            bar = "#" * bar_len
            print(f"L{idx:<6} {probe.accuracy:>10.1%}  {bar}")

    def save_probes(self, path: str):
        """Save trained probes to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "probes": {
                idx: {
                    "probe": probe.probe,
                    "label_encoder": probe.label_encoder,
                    "accuracy": probe.accuracy,
                }
                for idx, probe in self.probes.items()
            }
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"Saved probes to: {path}")

    def load_probes(self, path: str):
        """Load trained probes from disk."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        for idx, data in save_data["probes"].items():
            self.probes[idx] = LayerProbe(
                layer_idx=idx,
                probe=data["probe"],
                label_encoder=data["label_encoder"],
                accuracy=data["accuracy"],
            )

        print(f"Loaded {len(self.probes)} probes from: {path}")


def run_multiplication_experiment(lens: GemmaProbeLens):
    """Run the multiplication probe experiment."""
    print("\n" + "=" * 70)
    print("MULTIPLICATION PROBE LENS EXPERIMENT")
    print("=" * 70)

    # Create dataset
    prompts, answers, first_digits = lens.create_multiplication_dataset()
    print(f"\nDataset: {len(prompts)} multiplication problems")
    print(f"Labels: first digit of answer (classes: {sorted(set(first_digits))})")

    # Train probes for all layers
    lens.train_all_layers(prompts, first_digits, layer_step=1)

    # Visualize
    lens.visualize_layer_accuracy()

    # Test decoding
    print("\n" + "=" * 70)
    print("DECODING TEST CASES")
    print("=" * 70)

    test_cases = [
        ("7 * 8 = ", "56"),
        ("9 * 9 = ", "81"),
        ("6 * 7 = ", "42"),
        ("3 * 4 = ", "12"),
        ("8 * 9 = ", "72"),
    ]

    for prompt, expected in test_cases:
        first_digit = expected[0]
        print(f"\n{prompt}{expected}")
        print("-" * 40)

        results = lens.decode(prompt)

        # Show key layers
        key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 33]
        for r in results:
            if r.layer_idx in key_layers:
                correct = "YES" if r.predicted_token == first_digit else "NO"
                marker = " <--" if r.predicted_token == first_digit else ""
                print(f"  L{r.layer_idx:2d}: pred={r.predicted_token} P={r.probability:.3f} correct={correct}{marker}")


def run_addition_experiment(lens: GemmaProbeLens):
    """Run the addition probe experiment."""
    print("\n" + "=" * 70)
    print("ADDITION PROBE LENS EXPERIMENT")
    print("=" * 70)

    # Create dataset
    prompts, answers, first_digits = lens.create_addition_dataset()
    print(f"\nDataset: {len(prompts)} addition problems")
    print(f"Labels: first digit of answer (classes: {sorted(set(first_digits))})")

    # Train probes
    lens.train_all_layers(prompts, first_digits, layer_step=2)

    # Visualize
    lens.visualize_layer_accuracy()

    # Test
    print("\n" + "=" * 70)
    print("DECODING TEST CASES")
    print("=" * 70)

    test_cases = [
        ("15 + 23 = ", "38"),
        ("7 + 8 = ", "15"),
        ("19 + 19 = ", "38"),
        ("11 + 11 = ", "22"),
    ]

    for prompt, expected in test_cases:
        first_digit = expected[0]
        print(f"\n{prompt}{expected}")
        results = lens.decode(prompt)

        for r in results:
            if r.layer_idx % 8 == 0 or r.layer_idx == 33:
                correct = "YES" if r.predicted_token == first_digit else "NO"
                print(f"  L{r.layer_idx:2d}: pred={r.predicted_token} P={r.probability:.3f} correct={correct}")


def run_full_answer_experiment(lens: GemmaProbeLens):
    """
    Experiment: Can we decode the FULL answer (both digits)?

    Train separate probes for:
    - First digit (1-8)
    - Second digit (0-9)
    - Full answer (4-81)
    """
    print("\n" + "=" * 70)
    print("FULL ANSWER DECODING EXPERIMENT")
    print("=" * 70)

    prompts = []
    first_digits = []
    second_digits = []
    full_answers = []

    for a in range(2, 10):
        for b in range(2, 10):
            prompt = f"{a} * {b} = "
            answer = str(a * b)

            prompts.append(prompt)
            first_digits.append(answer[0])
            second_digits.append(answer[-1])  # Last digit (handles 1 and 2 digit)
            full_answers.append(answer)

    print(f"\nDataset: {len(prompts)} problems")
    print(f"First digit classes: {sorted(set(first_digits))}")
    print(f"Second digit classes: {sorted(set(second_digits))}")
    print(f"Full answer classes: {len(set(full_answers))}")

    # Pre-compute hidden states
    print("\nCollecting hidden states...")
    all_hidden = []
    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            print(f"  {i}/{len(prompts)}")
        states = lens.get_all_hidden_states(prompt)
        all_hidden.append(states)

    # Train probes for different targets at layer 20 (computation phase)
    layer_idx = 20
    print(f"\nTraining probes at layer {layer_idx}...")

    X = np.array([all_hidden[i][layer_idx] for i in range(len(all_hidden))])

    # Split
    n = len(X)
    n_test = max(1, int(n * 0.2))
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[n_test:], indices[:n_test]

    X_train, X_test = X[train_idx], X[test_idx]

    results = {}

    for name, labels in [
        ("first_digit", first_digits),
        ("second_digit", second_digits),
        ("full_answer", full_answers),
    ]:
        le = LabelEncoder()
        y = le.fit_transform(labels)
        y_train, y_test = y[train_idx], y[test_idx]

        probe = LogisticRegression(max_iter=2000, C=0.5)
        probe.fit(X_train, y_train)

        accuracy = probe.score(X_test, y_test)
        results[name] = {
            "accuracy": accuracy,
            "n_classes": len(le.classes_),
            "probe": probe,
            "encoder": le,
        }

        print(f"  {name}: {accuracy:.1%} ({len(le.classes_)} classes)")

    # Test decoding
    print("\n" + "-" * 50)
    print("Test decoding at layer 20:")
    print("-" * 50)

    test_prompts = ["7 * 8 = ", "9 * 9 = ", "6 * 7 = ", "5 * 5 = "]

    for prompt in test_prompts:
        h = lens.get_hidden_state(prompt, layer_idx).reshape(1, -1)

        d1 = results["first_digit"]["encoder"].inverse_transform(
            results["first_digit"]["probe"].predict(h)
        )[0]
        d2 = results["second_digit"]["encoder"].inverse_transform(
            results["second_digit"]["probe"].predict(h)
        )[0]
        full = results["full_answer"]["encoder"].inverse_transform(
            results["full_answer"]["probe"].predict(h)
        )[0]

        # Get probabilities
        p1 = results["first_digit"]["probe"].predict_proba(h).max()
        p2 = results["second_digit"]["probe"].predict_proba(h).max()
        pf = results["full_answer"]["probe"].predict_proba(h).max()

        print(f"\n{prompt}")
        print(f"  First digit:  {d1} (P={p1:.3f})")
        print(f"  Second digit: {d2} (P={p2:.3f})")
        print(f"  Full answer:  {full} (P={pf:.3f})")
        print(f"  Combined:     {d1}{d2}")


def main():
    parser = argparse.ArgumentParser(description="Gemma Probe Lens")
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument(
        "--task", "-t",
        choices=["multiplication", "addition", "full_answer", "all"],
        default="multiplication",
    )
    parser.add_argument("--decode", "-d", help="Decode a specific prompt")
    parser.add_argument("--save", "-s", help="Save probes to file")
    parser.add_argument("--load", "-l", help="Load probes from file")
    args = parser.parse_args()

    lens = GemmaProbeLens(model_id=args.model)
    lens.load_model()

    if args.load:
        lens.load_probes(args.load)
        lens.visualize_layer_accuracy()

    if args.decode and lens.probes:
        print(f"\nDecoding: {args.decode}")
        results = lens.decode(args.decode)
        for r in results:
            print(f"  L{r.layer_idx:2d}: {r.predicted_token} (P={r.probability:.3f})")
    elif args.task == "multiplication" or args.task == "all":
        run_multiplication_experiment(lens)

    if args.task == "addition" or args.task == "all":
        run_addition_experiment(lens)

    if args.task == "full_answer" or args.task == "all":
        run_full_answer_experiment(lens)

    if args.save:
        lens.save_probes(args.save)


if __name__ == "__main__":
    main()
