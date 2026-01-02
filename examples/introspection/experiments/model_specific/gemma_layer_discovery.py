#!/usr/bin/env python3
"""
Gemma Layer Discovery: Mechanistic Interpretability Study

Hypothesis: Gemma exposes computations through activations rather than logits.
This script systematically probes Gemma's internal representations to discover:

1. CLASSIFICATION CIRCUITS: Does Gemma have distinct "modes" for different tasks?
   - Arithmetic mode vs language mode
   - Tool-calling detection
   - Code vs natural language

2. LOOKUP TABLE STRUCTURE: Does arithmetic use lookup tables?
   - Same-row activation (7*3 activates 7*1, 7*2, 7*4...)
   - Same-column activation (7*3 activates 2*3, 3*3, 4*3...)
   - Product proximity (21 activates 18, 24, 20, 22...)
   - Factor-based clustering

3. GHOST COMPUTATIONS: Are correct answers computed but suppressed?
   - Train probes at each layer
   - Compare probe accuracy vs model output accuracy
   - Identify "ghost layers" where computation happens

4. NEURON SPECIALIZATION: Are there "arithmetic neurons"?
   - Find neurons that activate specifically for math
   - Ablate them and measure impact
   - Can we find "classifier neurons" that route tasks?

Usage:
    # Full discovery sweep
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py

    # Quick mode
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py --quick

    # Specific experiment
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py --experiment classification
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py --experiment lookup_table
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py --experiment ghost
    uv run python examples/introspection/experiments/model_specific/gemma_layer_discovery.py --experiment neurons
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerActivation:
    """Captured activation at a layer."""
    layer: int
    hidden_state: np.ndarray  # [hidden_size] - last token
    mlp_activation: np.ndarray | None = None  # [intermediate_size] - MLP internals


@dataclass
class ProbeResult:
    """Result of probing a layer for a feature."""
    layer: int
    accuracy: float
    auc: float
    above_chance: float  # accuracy - 0.5
    is_significant: bool  # accuracy > 0.6 and above_chance > 0.1


@dataclass
class ClassificationCircuitResult:
    """Result of classification circuit analysis."""
    task_name: str
    emergence_layer: int | None  # First layer with >70% accuracy
    peak_layer: int  # Layer with highest accuracy
    peak_accuracy: float
    layer_accuracies: dict[int, float]
    layer_aucs: dict[int, float]


@dataclass
class LookupTableAnalysis:
    """Analysis of multiplication table structure."""
    operand_a: int
    operand_b: int
    correct_answer: int
    correct_rank: int | None
    correct_prob: float
    same_row: list[dict]  # Activations from a*x
    same_col: list[dict]  # Activations from x*b
    adjacent_products: list[dict]  # Products close to correct
    interpretation: str


@dataclass
class NeuronProfile:
    """Profile of a single neuron."""
    layer: int
    neuron_idx: int
    mean_positive: float
    mean_negative: float
    separation_score: float  # Cohen's d
    auc: float
    top_activating_prompts: list[tuple[str, float]]


# =============================================================================
# Core Discovery Class
# =============================================================================

class GemmaLayerDiscovery:
    """
    Systematic layer-by-layer discovery of Gemma's internal circuits.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/gemma-3-4b-it-bf16",
        cache_dir: Path | None = None,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir or Path("gemma_discovery_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self):
        """Load the Gemma model."""
        print(f"Loading model: {self.model_id}")

        result = HFLoader.download(self.model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {self.model_id}")

        print(f"  Family: {family_type.value}")

        family_info = get_family_info(family_type)
        self.config = family_info.config_class.from_hf_config(config_data)
        self.model = family_info.model_class(self.config)

        HFLoader.apply_weights_to_model(self.model, model_path, self.config, dtype=DType.BFLOAT16)
        self.tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Intermediate size: {self.config.intermediate_size}")

    def _get_layers(self):
        """Get model layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        return list(self.model.layers)

    def _get_embed(self):
        """Get embedding layer."""
        if hasattr(self.model, "model"):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    def _get_norm(self):
        """Get final norm."""
        if hasattr(self.model, "model"):
            return getattr(self.model.model, "norm", None)
        return getattr(self.model, "norm", None)

    def _get_head(self):
        """Get LM head."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        embed = self._get_embed()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def _get_embed_scale(self) -> float:
        """Get embedding scale (critical for Gemma)."""
        if hasattr(self.config, "embedding_scale"):
            return self.config.embedding_scale
        # Gemma default: sqrt(hidden_size)
        return float(self.config.hidden_size ** 0.5)

    def extract_layer_activations(
        self,
        prompt: str,
        layers: list[int] | None = None,
        include_mlp: bool = False,
    ) -> dict[int, LayerActivation]:
        """
        Extract hidden state activations at specified layers.

        Args:
            prompt: Input text
            layers: Which layers to capture (None = all)
            include_mlp: Also capture MLP intermediate activations

        Returns:
            Dict mapping layer index to LayerActivation
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        model_layers = self._get_layers()
        embed = self._get_embed()
        embed_scale = self._get_embed_scale()

        if layers is None:
            layers = list(range(len(model_layers)))

        # Embeddings with Gemma scaling
        h = embed(input_ids)
        h = h * embed_scale

        # Create mask
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        results = {}

        for layer_idx, layer in enumerate(model_layers):
            # Run layer forward
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            if layer_idx in layers:
                # Capture hidden state (last token)
                hidden = np.array(h[0, -1, :].tolist())

                mlp_act = None
                if include_mlp and hasattr(layer, "mlp"):
                    # Try to get MLP intermediate activations
                    mlp_act = self._get_mlp_intermediate(layer, h, mask)

                results[layer_idx] = LayerActivation(
                    layer=layer_idx,
                    hidden_state=hidden,
                    mlp_activation=mlp_act,
                )

        return results

    def _get_mlp_intermediate(self, layer, h: mx.array, mask: mx.array) -> np.ndarray | None:
        """Extract MLP intermediate activations (before down projection)."""
        try:
            # Get input to MLP (after attention + residual, with norm)
            if hasattr(layer, "post_attention_layernorm"):
                mlp_input = layer.post_attention_layernorm(h)
            else:
                mlp_input = h

            mlp = layer.mlp

            # Gemma uses gated GELU: gate_proj * gelu(up_proj) -> down_proj
            if hasattr(mlp, "gate_proj"):
                gate = mlp.gate_proj(mlp_input)
                up = mlp.up_proj(mlp_input)
                gate_activated = nn.gelu(gate)
                intermediate = gate_activated * up
            elif hasattr(mlp, "up"):
                intermediate = mlp.up(mlp_input)
            else:
                return None

            return np.array(intermediate[0, -1, :].tolist())
        except Exception:
            return None

    def get_model_prediction(self, prompt: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Get model's top-k predictions for next token."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        output = self.model(input_ids)

        logits = output.logits if hasattr(output, "logits") else output
        probs = mx.softmax(logits[0, -1, :])

        sorted_idx = mx.argsort(probs)[::-1][:top_k].tolist()

        results = []
        for idx in sorted_idx:
            token = self.tokenizer.decode([idx])
            prob = float(probs[idx])
            results.append((token, prob))

        return results

    # =========================================================================
    # Experiment 1: Classification Circuits
    # =========================================================================

    def discover_classification_circuits(
        self,
        quick: bool = False,
    ) -> dict[str, ClassificationCircuitResult]:
        """
        Discover task classification circuits.

        Tests whether Gemma has distinct "modes" for:
        - Arithmetic vs language
        - Code vs natural language
        - Factual vs creative
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: CLASSIFICATION CIRCUITS")
        print("=" * 70)
        print("\nHypothesis: Gemma has internal 'mode switches' detectable in activations")

        # Define task pairs for classification
        task_pairs = {
            "arithmetic_vs_language": {
                "positive": [
                    "What is 7 * 8?",
                    "Calculate 15 + 23",
                    "What's 144 / 12?",
                    "Solve: 25 - 17 =",
                    "What is 9 * 9?",
                    "How much is 100 + 50?",
                    "Calculate 81 / 9",
                    "What's 12 * 12?",
                    "Compute 999 + 1",
                    "What is 7 * 6?",
                ] if not quick else [
                    "What is 7 * 8?",
                    "Calculate 15 + 23",
                    "What is 9 * 9?",
                ],
                "negative": [
                    "What is the capital of France?",
                    "Explain quantum physics",
                    "Who wrote Romeo and Juliet?",
                    "Describe the solar system",
                    "What is photosynthesis?",
                    "Tell me about Einstein",
                    "How does gravity work?",
                    "What causes earthquakes?",
                    "Explain democracy",
                    "Who invented the telephone?",
                ] if not quick else [
                    "What is the capital of France?",
                    "Explain quantum physics",
                    "What is photosynthesis?",
                ],
            },
            "code_vs_natural": {
                "positive": [
                    "def fibonacci(n):",
                    "for i in range(10):",
                    "if __name__ == '__main__':",
                    "import numpy as np",
                    "class MyClass:",
                    "async def fetch():",
                    "lambda x: x * 2",
                    "try:\n    result = ",
                    "return sorted(items)",
                    "with open('file.txt') as f:",
                ] if not quick else [
                    "def fibonacci(n):",
                    "for i in range(10):",
                    "import numpy as np",
                ],
                "negative": [
                    "The weather today is",
                    "I enjoy reading books about",
                    "The best restaurants in Paris",
                    "How to make pasta",
                    "My favorite movie is",
                    "Tips for better sleep",
                    "The history of ancient Rome",
                    "Benefits of exercise",
                    "How to learn a new language",
                    "Best vacation destinations",
                ] if not quick else [
                    "The weather today is",
                    "I enjoy reading books about",
                    "How to make pasta",
                ],
            },
        }

        results = {}
        num_layers = self.config.num_hidden_layers

        # Which layers to test
        if quick:
            test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        else:
            test_layers = list(range(num_layers))

        for task_name, task_data in task_pairs.items():
            print(f"\n--- Task: {task_name} ---")

            positive_prompts = task_data["positive"]
            negative_prompts = task_data["negative"]

            # Split train/test
            n_train = int(len(positive_prompts) * 0.7)
            train_pos = positive_prompts[:n_train]
            train_neg = negative_prompts[:n_train]
            test_pos = positive_prompts[n_train:]
            test_neg = negative_prompts[n_train:]

            layer_accuracies = {}
            layer_aucs = {}

            for layer in test_layers:
                # Collect activations
                X_train, y_train = [], []
                for prompt in train_pos:
                    act = self.extract_layer_activations(prompt, layers=[layer])
                    X_train.append(act[layer].hidden_state)
                    y_train.append(1)
                for prompt in train_neg:
                    act = self.extract_layer_activations(prompt, layers=[layer])
                    X_train.append(act[layer].hidden_state)
                    y_train.append(0)

                X_test, y_test = [], []
                for prompt in test_pos:
                    act = self.extract_layer_activations(prompt, layers=[layer])
                    X_test.append(act[layer].hidden_state)
                    y_test.append(1)
                for prompt in test_neg:
                    act = self.extract_layer_activations(prompt, layers=[layer])
                    X_test.append(act[layer].hidden_state)
                    y_test.append(0)

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_test = np.array(X_test)
                y_test = np.array(y_test)

                # Train probe
                probe = LogisticRegression(max_iter=1000, C=1.0)
                probe.fit(X_train, y_train)

                # Evaluate
                y_pred = probe.predict(X_test)
                y_prob = probe.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = 0.5

                layer_accuracies[layer] = accuracy
                layer_aucs[layer] = auc

                print(f"  L{layer:2d}: accuracy={accuracy:.1%}, AUC={auc:.3f}")

            # Find emergence and peak
            emergence_layer = None
            for layer in sorted(layer_accuracies.keys()):
                if layer_accuracies[layer] >= 0.7:
                    emergence_layer = layer
                    break

            peak_layer = max(layer_accuracies, key=layer_accuracies.get)
            peak_accuracy = layer_accuracies[peak_layer]

            results[task_name] = ClassificationCircuitResult(
                task_name=task_name,
                emergence_layer=emergence_layer,
                peak_layer=peak_layer,
                peak_accuracy=peak_accuracy,
                layer_accuracies=layer_accuracies,
                layer_aucs=layer_aucs,
            )

            print(f"\n  Emergence layer (>70%): L{emergence_layer}")
            print(f"  Peak layer: L{peak_layer} ({peak_accuracy:.1%})")

        return results

    # =========================================================================
    # Experiment 2: Lookup Table Analysis
    # =========================================================================

    def analyze_lookup_tables(
        self,
        layer: int | None = None,
        quick: bool = False,
    ) -> list[LookupTableAnalysis]:
        """
        Analyze whether arithmetic uses lookup table structure.

        For each multiplication a*b, examine:
        - What other products activate alongside the correct answer?
        - Are they from the same row (a*x) or column (x*b)?
        - Is there product proximity (nearby numbers)?
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: LOOKUP TABLE STRUCTURE")
        print("=" * 70)
        print("\nHypothesis: Gemma stores multiplication as associative memory")

        if layer is None:
            # Use layer ~70% through the model (often where arithmetic happens)
            layer = int(self.config.num_hidden_layers * 0.7)

        print(f"\nAnalyzing layer {layer}")

        # Build product lookup
        product_sources = defaultdict(list)
        for i in range(2, 10):
            for j in range(2, 10):
                product_sources[i * j].append((i, j))

        results = []

        # Test multiplications
        if quick:
            test_cases = [(7, 8), (6, 9), (8, 7), (9, 9)]
        else:
            test_cases = [(a, b) for a in range(2, 10) for b in range(2, 10)]

        for a, b in test_cases:
            correct = a * b
            prompt = f"{a} * {b} = "

            print(f"\n{prompt}", end="")

            # Get model's predictions
            preds = self.get_model_prediction(prompt, top_k=30)

            # Find correct answer rank and prob
            correct_rank = None
            correct_prob = 0.0
            for i, (token, prob) in enumerate(preds):
                try:
                    if int(token.strip()) == correct:
                        correct_rank = i + 1
                        correct_prob = prob
                        break
                except:
                    pass

            print(f"-> {correct} (rank={correct_rank}, prob={correct_prob:.3f})")

            # Categorize other predictions
            same_row = []
            same_col = []
            adjacent = []

            for token, prob in preds:
                try:
                    val = int(token.strip())
                except:
                    continue

                if val == correct:
                    continue

                sources = product_sources.get(val, [])

                # Check same row (a*x)
                for src_a, src_b in sources:
                    if src_a == a or src_b == a:
                        same_row.append({"value": val, "prob": prob, "from": f"{src_a}*{src_b}"})
                        break

                # Check same column (x*b)
                for src_a, src_b in sources:
                    if src_a == b or src_b == b:
                        same_col.append({"value": val, "prob": prob, "from": f"{src_a}*{src_b}"})
                        break

                # Check adjacent (within 10)
                if abs(val - correct) <= 10 and sources:
                    adjacent.append({"value": val, "prob": prob, "from": f"{sources[0][0]}*{sources[0][1]}"})

            # Interpretation
            if len(same_row) > len(same_col):
                interpretation = "ROW-organized (first operand anchors)"
            elif len(same_col) > len(same_row):
                interpretation = "COLUMN-organized (second operand anchors)"
            elif same_row or same_col:
                interpretation = "Mixed organization"
            else:
                interpretation = "Product-based or unique"

            results.append(LookupTableAnalysis(
                operand_a=a,
                operand_b=b,
                correct_answer=correct,
                correct_rank=correct_rank,
                correct_prob=correct_prob,
                same_row=same_row[:5],
                same_col=same_col[:5],
                adjacent_products=adjacent[:5],
                interpretation=interpretation,
            ))

            print(f"  Same row: {len(same_row)}, Same col: {len(same_col)}, Adjacent: {len(adjacent)}")
            print(f"  Interpretation: {interpretation}")

        # Summary
        print("\n" + "-" * 50)
        print("LOOKUP TABLE SUMMARY")
        row_biased = sum(1 for r in results if "ROW" in r.interpretation)
        col_biased = sum(1 for r in results if "COLUMN" in r.interpretation)
        mixed = sum(1 for r in results if "Mixed" in r.interpretation)

        print(f"Row-organized: {row_biased}/{len(results)}")
        print(f"Column-organized: {col_biased}/{len(results)}")
        print(f"Mixed: {mixed}/{len(results)}")

        return results

    # =========================================================================
    # Experiment 3: Ghost Computation Detection
    # =========================================================================

    def find_ghost_computations(
        self,
        quick: bool = False,
    ) -> dict:
        """
        Find "ghost computations" - layers where correct answers are encoded
        but not output by the model.

        Train probes at each layer and compare to model output.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: GHOST COMPUTATION DETECTION")
        print("=" * 70)
        print("\nHypothesis: Correct answers exist in hidden states even when model is wrong")

        # Generate arithmetic problems
        problems = []
        for a in range(2, 10):
            for b in range(2, 10):
                problems.append({
                    "prompt": f"{a} * {b} = ",
                    "answer": str(a * b),
                    "first_digit": str(a * b)[0],
                })

        if quick:
            problems = problems[::8]  # Sample every 8th

        print(f"\nTesting {len(problems)} problems")

        # For each problem, get model prediction and hidden states
        num_layers = self.config.num_hidden_layers

        if quick:
            test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        else:
            test_layers = list(range(0, num_layers, 2))  # Every other layer

        # Collect data
        layer_data = {layer: {"X": [], "y": []} for layer in test_layers}
        model_correct = 0
        model_total = 0

        for prob in problems:
            prompt = prob["prompt"]
            expected = prob["first_digit"]

            # Get model prediction
            preds = self.get_model_prediction(prompt, top_k=1)
            model_pred = preds[0][0].strip() if preds else ""
            is_correct = expected in model_pred
            model_correct += int(is_correct)
            model_total += 1

            # Get target token ID
            target_ids = self.tokenizer.encode(expected, add_special_tokens=False)
            target_id = target_ids[0] if target_ids else 0

            # Extract activations
            activations = self.extract_layer_activations(prompt, layers=test_layers)

            for layer in test_layers:
                layer_data[layer]["X"].append(activations[layer].hidden_state)
                layer_data[layer]["y"].append(target_id)

        print(f"\nModel accuracy: {model_correct}/{model_total} ({100*model_correct/model_total:.1f}%)")

        # Train probes at each layer
        print("\nTraining probes per layer...")

        layer_probe_accuracy = {}
        ghosts_found = {}

        for layer in test_layers:
            X = np.array(layer_data[layer]["X"])
            y = np.array(layer_data[layer]["y"])

            # Simple probe: can we predict the target token from hidden states?
            # Use cross-validation
            probe = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(probe, X, y, cv=min(5, len(y) // 2))
            accuracy = float(np.mean(scores))

            layer_probe_accuracy[layer] = accuracy

            # Count "ghosts": problems where probe is right but model is wrong
            probe.fit(X, y)
            probe_preds = probe.predict(X)

            n_ghosts = 0
            for i, prob in enumerate(problems):
                expected = prob["first_digit"]
                model_pred = self.get_model_prediction(prob["prompt"], top_k=1)[0][0].strip()
                model_is_correct = expected in model_pred
                probe_is_correct = (probe_preds[i] == layer_data[layer]["y"][i])

                if probe_is_correct and not model_is_correct:
                    n_ghosts += 1

            ghosts_found[layer] = n_ghosts

            print(f"  L{layer:2d}: probe_acc={accuracy:.1%}, ghosts={n_ghosts}")

        # Find best ghost layer
        best_ghost_layer = max(ghosts_found, key=ghosts_found.get)

        print(f"\n{'='*50}")
        print("GHOST COMPUTATION SUMMARY")
        print(f"{'='*50}")
        print(f"Model accuracy: {100*model_correct/model_total:.1f}%")
        print(f"Best ghost layer: L{best_ghost_layer} ({ghosts_found[best_ghost_layer]} ghosts)")
        print(f"Best probe accuracy: {max(layer_probe_accuracy.values()):.1%}")

        return {
            "model_accuracy": model_correct / model_total,
            "layer_probe_accuracy": layer_probe_accuracy,
            "ghosts_per_layer": ghosts_found,
            "best_ghost_layer": best_ghost_layer,
        }

    # =========================================================================
    # Experiment 4: Neuron Specialization
    # =========================================================================

    def profile_specialized_neurons(
        self,
        layer: int | None = None,
        quick: bool = False,
    ) -> dict:
        """
        Find neurons that specialize for arithmetic.

        Compare MLP activations for arithmetic vs non-arithmetic prompts.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: NEURON SPECIALIZATION")
        print("=" * 70)
        print("\nHypothesis: Specific neurons activate for arithmetic tasks")

        if layer is None:
            layer = int(self.config.num_hidden_layers * 0.7)

        print(f"\nAnalyzing layer {layer}")

        # Prompts
        arithmetic_prompts = [
            "7 * 8 = ", "15 + 23 = ", "100 - 37 = ", "144 / 12 = ",
            "9 * 9 = ", "25 + 17 = ", "50 - 8 = ", "81 / 9 = ",
        ]

        language_prompts = [
            "The capital of France is",
            "Water freezes at",
            "Shakespeare wrote",
            "The sun is a",
            "Elephants are known for",
            "The speed of light is",
            "Democracy means",
            "Photosynthesis converts",
        ]

        if quick:
            arithmetic_prompts = arithmetic_prompts[:4]
            language_prompts = language_prompts[:4]

        # Collect MLP activations
        arith_acts = []
        lang_acts = []

        for prompt in arithmetic_prompts:
            act = self.extract_layer_activations(prompt, layers=[layer], include_mlp=True)
            if act[layer].mlp_activation is not None:
                arith_acts.append(act[layer].mlp_activation)

        for prompt in language_prompts:
            act = self.extract_layer_activations(prompt, layers=[layer], include_mlp=True)
            if act[layer].mlp_activation is not None:
                lang_acts.append(act[layer].mlp_activation)

        if not arith_acts or not lang_acts:
            print("Could not extract MLP activations")
            return {}

        arith_acts = np.array(arith_acts)
        lang_acts = np.array(lang_acts)

        num_neurons = arith_acts.shape[1]
        print(f"  Analyzing {num_neurons} neurons")

        # Profile each neuron
        neuron_profiles = []

        for neuron_idx in range(num_neurons):
            arith_vals = arith_acts[:, neuron_idx]
            lang_vals = lang_acts[:, neuron_idx]

            mean_arith = float(np.mean(arith_vals))
            mean_lang = float(np.mean(lang_vals))
            std_arith = float(np.std(arith_vals))
            std_lang = float(np.std(lang_vals))

            # Cohen's d
            pooled_std = np.sqrt((std_arith**2 + std_lang**2) / 2)
            if pooled_std > 1e-6:
                separation = abs(mean_arith - mean_lang) / pooled_std
            else:
                separation = 0.0

            # AUC
            all_vals = np.concatenate([arith_vals, lang_vals])
            all_labels = np.array([1] * len(arith_vals) + [0] * len(lang_vals))
            try:
                auc = roc_auc_score(all_labels, all_vals)
                if auc < 0.5:
                    auc = 1 - auc
            except:
                auc = 0.5

            neuron_profiles.append({
                "neuron_idx": neuron_idx,
                "mean_arithmetic": mean_arith,
                "mean_language": mean_lang,
                "separation": separation,
                "auc": auc,
                "prefers": "arithmetic" if mean_arith > mean_lang else "language",
            })

        # Sort by separation
        neuron_profiles.sort(key=lambda x: -x["separation"])

        # Top arithmetic neurons
        top_arith = [n for n in neuron_profiles if n["prefers"] == "arithmetic"][:10]
        top_lang = [n for n in neuron_profiles if n["prefers"] == "language"][:10]

        print(f"\nTop 10 ARITHMETIC-preferring neurons:")
        print(f"{'Neuron':<10} {'Sep':<8} {'AUC':<8} {'μ_arith':<12} {'μ_lang':<12}")
        print("-" * 50)
        for n in top_arith:
            print(f"{n['neuron_idx']:<10} {n['separation']:<8.2f} {n['auc']:<8.3f} "
                  f"{n['mean_arithmetic']:<12.2f} {n['mean_language']:<12.2f}")

        print(f"\nTop 10 LANGUAGE-preferring neurons:")
        print(f"{'Neuron':<10} {'Sep':<8} {'AUC':<8} {'μ_arith':<12} {'μ_lang':<12}")
        print("-" * 50)
        for n in top_lang:
            print(f"{n['neuron_idx']:<10} {n['separation']:<8.2f} {n['auc']:<8.3f} "
                  f"{n['mean_arithmetic']:<12.2f} {n['mean_language']:<12.2f}")

        # How many neurons are highly specialized?
        high_sep = [n for n in neuron_profiles if n["separation"] > 1.0]
        print(f"\nHighly specialized neurons (sep > 1.0): {len(high_sep)}/{num_neurons}")

        return {
            "layer": layer,
            "total_neurons": num_neurons,
            "top_arithmetic_neurons": top_arith,
            "top_language_neurons": top_lang,
            "highly_specialized_count": len(high_sep),
        }

    # =========================================================================
    # Activation vs Logit Information Comparison
    # =========================================================================

    def compare_activation_vs_logit(
        self,
        quick: bool = False,
    ) -> dict:
        """
        Compare information content in activations vs logits.

        Your hypothesis: Gemma exposes more through activations than logits.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: ACTIVATION VS LOGIT INFORMATION")
        print("=" * 70)
        print("\nHypothesis: Activations contain more information than logits")

        # Problems where we know the answer
        problems = []
        for a in range(2, 10):
            for b in range(2, 10):
                problems.append({
                    "prompt": f"{a} * {b} = ",
                    "answer": a * b,
                })

        if quick:
            problems = problems[::8]

        # For each problem, measure:
        # 1. Logit: probability of correct answer
        # 2. Activation: probe accuracy at best layer

        logit_probs = []
        activation_info = []

        # First, collect data for probe training
        num_layers = self.config.num_hidden_layers
        best_layer = int(num_layers * 0.7)  # Hypothesis layer

        X_all = []
        y_all = []

        for prob in problems:
            prompt = prob["prompt"]
            answer = str(prob["answer"])

            # Logit probability
            preds = self.get_model_prediction(prompt, top_k=100)
            correct_prob = 0.0
            for token, p in preds:
                if answer in token:
                    correct_prob = p
                    break
            logit_probs.append(correct_prob)

            # Get activation
            act = self.extract_layer_activations(prompt, layers=[best_layer])
            X_all.append(act[best_layer].hidden_state)

            # Target: class label based on answer
            y_all.append(prob["answer"])

        # Train probe
        X_all = np.array(X_all)
        y_all = np.array(y_all)

        probe = LogisticRegression(max_iter=2000, C=0.1)
        probe.fit(X_all, y_all)

        # Get probe confidence for each problem
        probe_probs = probe.predict_proba(X_all)
        for i, prob in enumerate(problems):
            # Find probability assigned to correct class
            correct_class_idx = list(probe.classes_).index(prob["answer"])
            activation_info.append(float(probe_probs[i, correct_class_idx]))

        # Compare
        logit_mean = np.mean(logit_probs)
        activation_mean = np.mean(activation_info)

        correlation = np.corrcoef(logit_probs, activation_info)[0, 1]

        print(f"\nMean logit probability for correct answer: {logit_mean:.3f}")
        print(f"Mean probe probability for correct answer: {activation_mean:.3f}")
        print(f"Correlation between logit and probe probs: {correlation:.3f}")

        # Find cases where probe >> logit (activation has more info)
        activation_advantage = []
        for i, prob in enumerate(problems):
            diff = activation_info[i] - logit_probs[i]
            if diff > 0.2:  # Probe much more confident
                activation_advantage.append({
                    "problem": prob["prompt"],
                    "answer": prob["answer"],
                    "logit_prob": logit_probs[i],
                    "probe_prob": activation_info[i],
                    "advantage": diff,
                })

        activation_advantage.sort(key=lambda x: -x["advantage"])

        print(f"\nProblems where ACTIVATION >> LOGIT:")
        for case in activation_advantage[:5]:
            print(f"  {case['problem']} = {case['answer']}")
            print(f"    Logit: {case['logit_prob']:.3f}, Probe: {case['probe_prob']:.3f}, Advantage: {case['advantage']:+.3f}")

        return {
            "logit_mean": logit_mean,
            "activation_mean": activation_mean,
            "correlation": correlation,
            "activation_advantage_cases": activation_advantage,
        }

    # =========================================================================
    # Main Runner
    # =========================================================================

    def run_full_discovery(self, quick: bool = False) -> dict:
        """Run all experiments."""
        self.load_model()

        results = {}

        # Experiment 1: Classification circuits
        results["classification"] = self.discover_classification_circuits(quick=quick)

        # Experiment 2: Lookup tables
        results["lookup_tables"] = self.analyze_lookup_tables(quick=quick)

        # Experiment 3: Ghost computations
        results["ghost"] = self.find_ghost_computations(quick=quick)

        # Experiment 4: Neuron specialization
        results["neurons"] = self.profile_specialized_neurons(quick=quick)

        # Experiment 5: Activation vs Logit
        results["activation_vs_logit"] = self.compare_activation_vs_logit(quick=quick)

        # Summary
        print("\n" + "=" * 70)
        print("GEMMA LAYER DISCOVERY SUMMARY")
        print("=" * 70)

        print("\n1. CLASSIFICATION CIRCUITS:")
        for task, result in results["classification"].items():
            print(f"   {task}: emerges L{result.emergence_layer}, peaks L{result.peak_layer} ({result.peak_accuracy:.1%})")

        print("\n2. LOOKUP TABLE STRUCTURE:")
        row_bias = sum(1 for r in results["lookup_tables"] if "ROW" in r.interpretation)
        col_bias = sum(1 for r in results["lookup_tables"] if "COLUMN" in r.interpretation)
        print(f"   Row-organized: {row_bias}, Column-organized: {col_bias}")

        print("\n3. GHOST COMPUTATIONS:")
        ghost = results["ghost"]
        print(f"   Model accuracy: {ghost['model_accuracy']:.1%}")
        print(f"   Best ghost layer: L{ghost['best_ghost_layer']} ({ghost['ghosts_per_layer'][ghost['best_ghost_layer']]} ghosts)")

        print("\n4. NEURON SPECIALIZATION:")
        neurons = results["neurons"]
        if neurons:
            print(f"   Layer {neurons['layer']}: {neurons['highly_specialized_count']}/{neurons['total_neurons']} highly specialized")

        print("\n5. ACTIVATION VS LOGIT:")
        avl = results["activation_vs_logit"]
        print(f"   Logit mean: {avl['logit_mean']:.3f}, Activation mean: {avl['activation_mean']:.3f}")
        print(f"   Cases with activation advantage: {len(avl['activation_advantage_cases'])}")

        # Save results
        output_path = self.cache_dir / "discovery_results.json"

        # Convert to serializable format
        serializable = {
            "model_id": self.model_id,
            "classification": {
                k: {
                    "emergence_layer": v.emergence_layer,
                    "peak_layer": v.peak_layer,
                    "peak_accuracy": v.peak_accuracy,
                    "layer_accuracies": v.layer_accuracies,
                }
                for k, v in results["classification"].items()
            },
            "lookup_tables": [
                {
                    "operand_a": r.operand_a,
                    "operand_b": r.operand_b,
                    "correct_answer": r.correct_answer,
                    "correct_rank": r.correct_rank,
                    "interpretation": r.interpretation,
                }
                for r in results["lookup_tables"]
            ],
            "ghost": results["ghost"],
            "neurons": results["neurons"],
            "activation_vs_logit": {
                "logit_mean": avl["logit_mean"],
                "activation_mean": avl["activation_mean"],
                "correlation": avl["correlation"],
            },
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Gemma Layer Discovery")
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16",
                        help="Model ID")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode (fewer tests)")
    parser.add_argument("--experiment", "-e", choices=["classification", "lookup_table", "ghost", "neurons", "activation_logit", "all"],
                        default="all", help="Which experiment to run")
    args = parser.parse_args()

    discovery = GemmaLayerDiscovery(model_id=args.model)

    if args.experiment == "all":
        discovery.run_full_discovery(quick=args.quick)
    else:
        discovery.load_model()

        if args.experiment == "classification":
            discovery.discover_classification_circuits(quick=args.quick)
        elif args.experiment == "lookup_table":
            discovery.analyze_lookup_tables(quick=args.quick)
        elif args.experiment == "ghost":
            discovery.find_ghost_computations(quick=args.quick)
        elif args.experiment == "neurons":
            discovery.profile_specialized_neurons(quick=args.quick)
        elif args.experiment == "activation_logit":
            discovery.compare_activation_vs_logit(quick=args.quick)


if __name__ == "__main__":
    main()
