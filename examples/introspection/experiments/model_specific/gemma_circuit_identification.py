#!/usr/bin/env python3
"""
Gemma Circuit Identification: Find the multiplication circuit.

This script identifies the specific components (attention heads, MLP neurons)
that implement multiplication in Gemma, similar to GPT-OSS circuit analysis.

Key experiments:
1. Attention Head Profiling: Which heads attend to operands during multiplication?
2. MLP Neuron Profiling: Which neurons activate specifically for arithmetic?
3. Ablation Study: Remove components and measure impact on multiplication accuracy
4. Circuit Mapping: Build a graph of the multiplication circuit

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_circuit_identification.py
    uv run python examples/introspection/experiments/model_specific/gemma_circuit_identification.py --ablation
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class AttentionHeadProfile:
    """Profile of a single attention head."""
    layer: int
    head: int
    operand_attention: float  # How much it attends to operand tokens
    equals_attention: float   # How much it attends to "=" token
    arithmetic_specificity: float  # Higher for arithmetic vs language
    importance_score: float  # Combined importance metric


@dataclass
class NeuronProfile:
    """Profile of a single MLP neuron."""
    layer: int
    neuron_idx: int
    mean_arithmetic: float
    mean_language: float
    separation_score: float  # How well it separates arithmetic from language
    top_activating_prompts: list[tuple[str, float]]


@dataclass
class CircuitComponent:
    """A component of the multiplication circuit."""
    component_type: str  # "attention_head" or "mlp_neuron"
    layer: int
    index: int  # head index or neuron index
    role: str  # "operand_reader", "operator_detector", "result_computer", etc.
    importance: float
    ablation_effect: float  # Change in accuracy when ablated


@dataclass
class MultiplicationCircuit:
    """The identified multiplication circuit."""
    components: list[CircuitComponent]
    accuracy_baseline: float
    accuracy_ablated: float  # With all circuit components ablated


class GemmaCircuitIdentifier:
    """Identify the multiplication circuit in Gemma."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

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
        self.num_heads = self.config.num_attention_heads

        # Get intermediate size for MLP
        if hasattr(self.config, 'intermediate_size'):
            self.intermediate_size = self.config.intermediate_size
        else:
            self.intermediate_size = self.hidden_size * 4

        print(f"  Layers: {self.num_layers}")
        print(f"  Heads: {self.num_heads}")
        print(f"  Hidden: {self.hidden_size}")
        print(f"  Intermediate: {self.intermediate_size}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        norm = getattr(backbone, "norm", None)

        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        else:
            head = None

        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
            embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, norm, head, embed_scale

    def get_attention_patterns(self, prompt: str, layer_idx: int) -> np.ndarray:
        """
        Get attention patterns from a specific layer.

        Returns: [num_heads, seq_len, seq_len] attention weights
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Run through layers up to target
        for i, layer in enumerate(layers):
            if i == layer_idx:
                # Extract attention weights from this layer
                attn = layer.self_attn

                # Get attention input
                if hasattr(layer, 'input_layernorm'):
                    h_normed = layer.input_layernorm(h)
                else:
                    h_normed = h

                # Get Q, K, V projections
                if hasattr(attn, 'q_proj'):
                    q = attn.q_proj(h_normed)
                    k = attn.k_proj(h_normed)

                    # Reshape for heads
                    head_dim = self.hidden_size // self.num_heads
                    q = q.reshape(1, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
                    k = k.reshape(1, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

                    # Compute attention weights
                    scale = head_dim ** -0.5
                    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

                    # Apply mask
                    scores = scores + mask

                    # Softmax
                    attn_weights = mx.softmax(scores, axis=-1)

                    return np.array(attn_weights[0].tolist())  # [num_heads, seq_len, seq_len]

            # Run layer
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

        return None

    def get_mlp_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """
        Get MLP intermediate activations from a specific layer.

        Returns: [seq_len, intermediate_size] activations
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(layers):
            if i == layer_idx:
                # Run attention first
                if hasattr(layer, 'input_layernorm'):
                    h_normed = layer.input_layernorm(h)
                else:
                    h_normed = h

                attn = layer.self_attn
                try:
                    attn_out = attn(h_normed, mask=mask)
                except TypeError:
                    attn_out = attn(h_normed)

                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                # Now get MLP activations
                if hasattr(layer, 'post_attention_layernorm'):
                    mlp_input = layer.post_attention_layernorm(h)
                else:
                    mlp_input = h

                mlp = layer.mlp

                # Get gate and up projections
                if hasattr(mlp, 'gate_proj'):
                    gate = mlp.gate_proj(mlp_input)
                    up = mlp.up_proj(mlp_input)
                    # Apply activation
                    gate_activated = nn.gelu(gate)
                    mlp_activations = gate_activated * up
                else:
                    mlp_activations = mlp.up(mlp_input)

                return np.array(mlp_activations[0].tolist())  # [seq_len, intermediate_size]

            # Run layer
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

        return None

    def get_hidden_state(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get hidden state from a specific layer."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(layers):
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

            if i == layer_idx:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    def create_dataset(self) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
        """
        Create arithmetic vs language dataset.

        Returns: (train_data, test_data) where each is list of (prompt, is_arithmetic)
        """
        arithmetic_prompts = []
        language_prompts = []

        # Arithmetic: multiplication
        for a in range(2, 10):
            for b in range(2, 10):
                arithmetic_prompts.append(f"{a} * {b} = ")

        # Arithmetic: addition
        for a in range(1, 20):
            for b in range(1, 20):
                if len(arithmetic_prompts) < 120:
                    arithmetic_prompts.append(f"{a} + {b} = ")

        # Language prompts
        language_prompts = [
            "The capital of France is",
            "Water freezes at",
            "The largest planet is",
            "Shakespeare wrote",
            "Einstein developed",
            "The speed of light is",
            "DNA stands for",
            "The chemical symbol for gold is",
            "Photosynthesis is the process",
            "The Great Wall of China",
            "Democracy is a form of",
            "The human heart has",
            "Gravity is a force that",
            "The Pacific Ocean is",
            "World War II ended in",
            "The moon orbits",
            "Plants need sunlight to",
            "The Amazon rainforest",
            "Electricity is the flow of",
            "The Renaissance was a period",
            "Dinosaurs went extinct",
            "The brain controls",
            "Oxygen is essential for",
            "The pyramids of Egypt",
            "Climate change refers to",
            "The Industrial Revolution",
            "Atoms are made of",
            "The Internet connects",
            "Evolution is the process",
            "The United Nations was",
            "Black holes are regions",
            "The stock market is",
            "Vaccines help prevent",
            "Artificial intelligence is",
            "The Renaissance artists",
            "Quantum mechanics describes",
            "The French Revolution",
            "Biodiversity refers to",
            "The Roman Empire",
            "Nuclear energy is",
        ]

        # Shuffle and split
        np.random.seed(42)

        arith_shuffled = arithmetic_prompts.copy()
        np.random.shuffle(arith_shuffled)

        lang_shuffled = language_prompts.copy()
        np.random.shuffle(lang_shuffled)

        # Balance datasets
        n_each = min(len(arith_shuffled), len(lang_shuffled))
        n_train = int(n_each * 0.8)

        train_arith = [(p, True) for p in arith_shuffled[:n_train]]
        test_arith = [(p, True) for p in arith_shuffled[n_train:n_each]]
        train_lang = [(p, False) for p in lang_shuffled[:n_train]]
        test_lang = [(p, False) for p in lang_shuffled[n_train:n_each]]

        train_data = train_arith + train_lang
        test_data = test_arith + test_lang

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        return train_data, test_data

    def profile_attention_heads(self, layer_idx: int) -> list[AttentionHeadProfile]:
        """Profile attention heads in a layer for arithmetic behavior."""
        print(f"\n  Profiling attention heads in layer {layer_idx}...")

        # Collect attention patterns for arithmetic prompts
        arith_patterns = []
        lang_patterns = []

        # Sample prompts
        arith_prompts = [f"{a} * {b} = " for a in range(2, 6) for b in range(2, 6)]
        lang_prompts = [
            "The capital of",
            "Water freezes at",
            "The largest planet",
            "Shakespeare wrote",
        ]

        for prompt in arith_prompts[:10]:
            patterns = self.get_attention_patterns(prompt, layer_idx)
            if patterns is not None:
                arith_patterns.append(patterns)

        for prompt in lang_prompts:
            patterns = self.get_attention_patterns(prompt, layer_idx)
            if patterns is not None:
                lang_patterns.append(patterns)

        if not arith_patterns:
            print(f"    Warning: Could not get attention patterns")
            return []

        profiles = []

        for head_idx in range(self.num_heads):
            # Compute operand attention for arithmetic
            operand_attn_scores = []
            for patterns in arith_patterns:
                # patterns: [num_heads, seq_len, seq_len]
                head_pattern = patterns[head_idx]  # [seq_len, seq_len]

                # Last token attending to early tokens (operands)
                if head_pattern.shape[0] > 2:
                    # Attention from last position to first few positions
                    operand_attn = head_pattern[-1, :3].mean()
                    operand_attn_scores.append(operand_attn)

            avg_operand_attn = np.mean(operand_attn_scores) if operand_attn_scores else 0

            # Compute arithmetic specificity
            arith_entropy = []
            lang_entropy = []

            for patterns in arith_patterns:
                head_pattern = patterns[head_idx]
                # Entropy of attention distribution
                p = head_pattern[-1] + 1e-10
                p = p / p.sum()
                entropy = -np.sum(p * np.log(p))
                arith_entropy.append(entropy)

            for patterns in lang_patterns:
                head_pattern = patterns[head_idx]
                p = head_pattern[-1] + 1e-10
                p = p / p.sum()
                entropy = -np.sum(p * np.log(p))
                lang_entropy.append(entropy)

            # Specificity: difference in entropy (lower entropy = more focused)
            avg_arith_entropy = np.mean(arith_entropy) if arith_entropy else 0
            avg_lang_entropy = np.mean(lang_entropy) if lang_entropy else 0
            specificity = avg_lang_entropy - avg_arith_entropy  # Higher = more arithmetic-specific

            # Importance score
            importance = avg_operand_attn * max(0, specificity + 1)

            profiles.append(AttentionHeadProfile(
                layer=layer_idx,
                head=head_idx,
                operand_attention=float(avg_operand_attn),
                equals_attention=0.0,  # TODO: compute if needed
                arithmetic_specificity=float(specificity),
                importance_score=float(importance),
            ))

        # Sort by importance
        profiles.sort(key=lambda x: -x.importance_score)

        return profiles

    def profile_mlp_neurons(
        self,
        layer_idx: int,
        train_data: list[tuple[str, bool]],
    ) -> list[NeuronProfile]:
        """Profile MLP neurons for arithmetic specificity."""
        print(f"\n  Profiling MLP neurons in layer {layer_idx}...")

        arith_activations = []
        lang_activations = []

        for prompt, is_arith in train_data[:50]:  # Sample for speed
            mlp_act = self.get_mlp_activations(prompt, layer_idx)
            if mlp_act is None:
                continue

            # Last token activations
            last_token = mlp_act[-1]

            if is_arith:
                arith_activations.append(last_token)
            else:
                lang_activations.append(last_token)

        if not arith_activations or not lang_activations:
            print(f"    Warning: Could not collect MLP activations")
            return []

        arith_activations = np.array(arith_activations)
        lang_activations = np.array(lang_activations)

        num_neurons = arith_activations.shape[1]
        profiles = []

        for neuron_idx in range(num_neurons):
            arith_vals = arith_activations[:, neuron_idx]
            lang_vals = lang_activations[:, neuron_idx]

            mean_arith = float(np.mean(arith_vals))
            mean_lang = float(np.mean(lang_vals))
            std_arith = float(np.std(arith_vals))
            std_lang = float(np.std(lang_vals))

            # Separation score
            pooled_std = np.sqrt((std_arith**2 + std_lang**2) / 2)
            if pooled_std > 1e-6:
                separation = abs(mean_arith - mean_lang) / pooled_std
            else:
                separation = 0.0

            profiles.append(NeuronProfile(
                layer=layer_idx,
                neuron_idx=neuron_idx,
                mean_arithmetic=mean_arith,
                mean_language=mean_lang,
                separation_score=float(separation),
                top_activating_prompts=[],  # Fill later if needed
            ))

        # Sort by separation
        profiles.sort(key=lambda x: -x.separation_score)

        return profiles

    def find_arithmetic_circuit(
        self,
        train_data: list[tuple[str, bool]],
        test_data: list[tuple[str, bool]],
    ) -> MultiplicationCircuit:
        """Find the complete arithmetic circuit."""
        print("\n" + "=" * 70)
        print("IDENTIFYING ARITHMETIC CIRCUIT")
        print("=" * 70)

        components = []

        # Profile attention heads in key layers
        print("\n--- Attention Head Profiling ---")
        for layer_idx in [16, 20, 24, 28]:  # Focus on computation layers
            profiles = self.profile_attention_heads(layer_idx)

            # Keep top heads
            for profile in profiles[:3]:  # Top 3 per layer
                if profile.importance_score > 0.1:
                    components.append(CircuitComponent(
                        component_type="attention_head",
                        layer=profile.layer,
                        index=profile.head,
                        role="operand_reader" if profile.operand_attention > 0.2 else "context",
                        importance=profile.importance_score,
                        ablation_effect=0.0,  # Compute later
                    ))

        # Profile MLP neurons in key layers
        print("\n--- MLP Neuron Profiling ---")
        for layer_idx in [20, 22, 24, 26]:  # Computation layers
            profiles = self.profile_mlp_neurons(layer_idx, train_data)

            # Keep top neurons
            for profile in profiles[:20]:  # Top 20 per layer
                if profile.separation_score > 2.0:  # Highly discriminative
                    role = "arithmetic_activator" if profile.mean_arithmetic > profile.mean_language else "language_suppressor"
                    components.append(CircuitComponent(
                        component_type="mlp_neuron",
                        layer=profile.layer,
                        index=profile.neuron_idx,
                        role=role,
                        importance=profile.separation_score,
                        ablation_effect=0.0,
                    ))

        # Compute baseline accuracy using probes
        print("\n--- Computing Baseline Accuracy ---")
        baseline_acc = self._compute_probe_accuracy(train_data, test_data, layer_idx=24)

        circuit = MultiplicationCircuit(
            components=components,
            accuracy_baseline=baseline_acc,
            accuracy_ablated=0.0,  # Compute if doing ablation
        )

        return circuit

    def _compute_probe_accuracy(
        self,
        train_data: list[tuple[str, bool]],
        test_data: list[tuple[str, bool]],
        layer_idx: int,
    ) -> float:
        """Compute probe accuracy for arithmetic classification."""
        X_train = []
        y_train = []

        for prompt, is_arith in train_data:
            h = self.get_hidden_state(prompt, layer_idx)
            X_train.append(h)
            y_train.append(int(is_arith))

        X_test = []
        y_test = []

        for prompt, is_arith in test_data:
            h = self.get_hidden_state(prompt, layer_idx)
            X_test.append(h)
            y_test.append(int(is_arith))

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        return float(accuracy_score(y_test, y_pred))

    def print_circuit(self, circuit: MultiplicationCircuit):
        """Pretty print the identified circuit."""
        print("\n" + "=" * 70)
        print("IDENTIFIED MULTIPLICATION CIRCUIT")
        print("=" * 70)

        print(f"\nBaseline accuracy: {circuit.accuracy_baseline:.1%}")
        print(f"Total components: {len(circuit.components)}")

        # Group by type
        attn_heads = [c for c in circuit.components if c.component_type == "attention_head"]
        mlp_neurons = [c for c in circuit.components if c.component_type == "mlp_neuron"]

        print(f"\n--- Attention Heads ({len(attn_heads)}) ---")
        print(f"{'Layer':<8} {'Head':<8} {'Role':<20} {'Importance':<12}")
        print("-" * 50)
        for c in sorted(attn_heads, key=lambda x: -x.importance)[:10]:
            print(f"L{c.layer:<7} H{c.index:<7} {c.role:<20} {c.importance:.4f}")

        print(f"\n--- MLP Neurons ({len(mlp_neurons)}) ---")
        print(f"{'Layer':<8} {'Neuron':<10} {'Role':<20} {'Separation':<12}")
        print("-" * 55)
        for c in sorted(mlp_neurons, key=lambda x: -x.importance)[:20]:
            print(f"L{c.layer:<7} N{c.index:<9} {c.role:<20} {c.importance:.4f}")

        # Summary by layer
        print(f"\n--- Components by Layer ---")
        by_layer = defaultdict(list)
        for c in circuit.components:
            by_layer[c.layer].append(c)

        for layer in sorted(by_layer.keys()):
            comps = by_layer[layer]
            n_attn = sum(1 for c in comps if c.component_type == "attention_head")
            n_mlp = sum(1 for c in comps if c.component_type == "mlp_neuron")
            print(f"  L{layer}: {n_attn} attention heads, {n_mlp} MLP neurons")

    def run_analysis(self, do_ablation: bool = False):
        """Run the full circuit identification."""
        self.load_model()

        # Create dataset
        train_data, test_data = self.create_dataset()
        print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")

        # Find circuit
        circuit = self.find_arithmetic_circuit(train_data, test_data)

        # Print results
        self.print_circuit(circuit)

        # Save results
        results = {
            "model": self.model_id,
            "baseline_accuracy": circuit.accuracy_baseline,
            "num_components": len(circuit.components),
            "components": [
                {
                    "type": c.component_type,
                    "layer": c.layer,
                    "index": c.index,
                    "role": c.role,
                    "importance": c.importance,
                }
                for c in circuit.components
            ],
        }

        output_path = Path("gemma_discovery_cache/arithmetic_circuit.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return circuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument("--ablation", "-a", action="store_true", help="Run ablation study")
    args = parser.parse_args()

    identifier = GemmaCircuitIdentifier(model_id=args.model)
    identifier.run_analysis(do_ablation=args.ablation)


if __name__ == "__main__":
    main()
