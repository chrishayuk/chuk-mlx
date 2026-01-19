#!/usr/bin/env python3
"""
feature_emergence_neurons.py

Phase 3.3 + 4.1: Feature Emergence + Neuron Profiling

KEY QUESTIONS:
1. When does "needs_tool" become decodable? (Layer-by-layer probe accuracy)
2. Which neurons in L11-12 MLP are responsible for tool-calling?
3. Is emergence gradual or sudden?
4. How many neurons are needed for the decision?

Run: uv run python examples/introspection/feature_emergence_neurons.py
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


@dataclass
class FeatureEmergenceResult:
    """Result of probing for a feature across layers."""

    feature_name: str
    layer_accuracies: dict[int, float]  # layer -> accuracy
    layer_aucs: dict[int, float]  # layer -> AUC
    emergence_layer: int  # First layer with >70% accuracy
    peak_layer: int  # Layer with highest accuracy
    peak_accuracy: float


@dataclass
class NeuronProfile:
    """Profile of a single neuron."""

    layer: int
    neuron_idx: int
    mean_tool: float
    mean_no_tool: float
    std_tool: float
    std_no_tool: float
    separation_score: float  # (mean_tool - mean_no_tool) / pooled_std
    auc: float  # How well this single neuron classifies


@dataclass
class NeuronProfilingResult:
    """Result of profiling neurons in a layer."""

    layer: int
    total_neurons: int
    top_neurons: list[NeuronProfile]  # Top discriminative neurons
    cumulative_accuracy: list[tuple[int, float]]  # (num_neurons, accuracy)


class FeatureEmergenceExperiment:
    """
    Combined feature emergence and neuron profiling experiment.
    """

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model."""
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        self.embed_layer = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.hidden_size = self.model.model.hidden_size
        self.embed_scale = self.hidden_size**0.5
        self.num_layers = len(self.layers)

        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}")

    def get_layer_activations(self, tokens: list[int], layer: int) -> mx.array:
        """Get hidden state output of a specific layer."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(layer + 1):
            layer_module = self.layers[layer_idx]
            try:
                layer_out = layer_module(h, mask=mask)
            except TypeError:
                layer_out = layer_module(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        return h.astype(mx.float32)

    def get_mlp_activations(self, tokens: list[int], layer: int) -> mx.array:
        """
        Get MLP intermediate activations (post-gate, pre-down projection).
        This captures the neuron activations in the MLP.
        """
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(layer):
            layer_module = self.layers[layer_idx]
            try:
                layer_out = layer_module(h, mask=mask)
            except TypeError:
                layer_out = layer_module(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        # Now we're at the input to target layer
        # Run attention part
        target_layer = self.layers[layer]

        # Get attention output
        if hasattr(target_layer, "input_layernorm"):
            normed = target_layer.input_layernorm(h)
        else:
            normed = h

        # Attention
        attn = target_layer.self_attn
        attn_out = attn(normed, mask=mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = h + attn_out

        # Now get MLP activations
        if hasattr(target_layer, "post_attention_layernorm"):
            mlp_input = target_layer.post_attention_layernorm(h)
        else:
            mlp_input = h

        mlp = target_layer.mlp

        # Get gate and up projections
        if hasattr(mlp, "gate_proj"):
            # Gemma style: gate * up
            gate = mlp.gate_proj(mlp_input)
            up = mlp.up_proj(mlp_input)
            # Apply activation to gate
            if hasattr(nn, "gelu"):
                gate_activated = nn.gelu(gate)
            else:
                gate_activated = mx.maximum(gate, 0)  # fallback to ReLU
            # MLP activations are gate * up (before down projection)
            mlp_activations = gate_activated * up
        else:
            # Simple MLP
            mlp_activations = mlp.up(mlp_input)

        return mlp_activations.astype(mx.float32)

    def prepare_dataset(self) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
        """Prepare train and test prompts."""
        # Tool-calling prompts (label = True)
        tool_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "Create a calendar event for tomorrow",
            "Search for restaurants nearby",
            "Set a timer for 10 minutes",
            "Get the stock price of Apple",
            "Book a flight to Paris",
            "Check my schedule for today",
            "Find hotels in London",
            "Order a pizza for dinner",
            "Turn on the lights",
            "Play some music",
            "Set an alarm for 7am",
            "Call mom",
            "Navigate to the airport",
            "Add milk to shopping list",
            "What's the traffic like?",
            "Translate hello to Spanish",
            "Convert 100 dollars to euros",
            "Check the news",
            "Send a message to Sarah",
            "Schedule a meeting for Friday",
            "Remind me to buy groceries",
            "What's my next appointment?",
            "Pay the electricity bill",
        ]

        # Non-tool prompts (label = False)
        no_tool_prompts = [
            "What is the capital of France?",
            "Explain quantum physics",
            "Write a poem about the ocean",
            "What is 2 + 2?",
            "Tell me about Einstein",
            "How do I learn Python?",
            "What is the meaning of life?",
            "Tell me a joke",
            "What is photosynthesis?",
            "Describe a rainbow",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
            "Explain democracy",
            "How does gravity work?",
            "What is machine learning?",
            "Why is the sky blue?",
            "What causes earthquakes?",
            "Who invented the telephone?",
            "What is DNA?",
            "Explain the theory of relativity",
            "What is artificial intelligence?",
            "How do planes fly?",
            "What is philosophy?",
            "Describe the solar system",
            "What is consciousness?",
        ]

        # Split into train/test
        train_tool = [(p, True) for p in tool_prompts[:20]]
        test_tool = [(p, True) for p in tool_prompts[20:]]
        train_no_tool = [(p, False) for p in no_tool_prompts[:20]]
        test_no_tool = [(p, False) for p in no_tool_prompts[20:]]

        train = train_tool + train_no_tool
        test = test_tool + test_no_tool

        return train, test

    def probe_feature_emergence(
        self,
        train_data: list[tuple[str, bool]],
        test_data: list[tuple[str, bool]],
    ) -> FeatureEmergenceResult:
        """
        Probe for needs_tool feature at each layer.
        """
        print("\n" + "=" * 60)
        print("FEATURE EMERGENCE: needs_tool")
        print("=" * 60)

        layer_accuracies = {}
        layer_aucs = {}

        for layer in range(self.num_layers):
            # Collect activations
            X_train = []
            y_train = []
            for prompt, label in train_data:
                tokens = self.tokenizer.encode(prompt)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()

                act = self.get_layer_activations(tokens, layer)
                # Use last token activation
                last_token = np.array(act[0, -1, :].tolist())
                X_train.append(last_token)
                y_train.append(int(label))

            X_test = []
            y_test = []
            for prompt, label in test_data:
                tokens = self.tokenizer.encode(prompt)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()

                act = self.get_layer_activations(tokens, layer)
                last_token = np.array(act[0, -1, :].tolist())
                X_test.append(last_token)
                y_test.append(int(label))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            # Train logistic regression probe
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
        emergence_layer = -1
        for layer in range(self.num_layers):
            if layer_accuracies[layer] >= 0.70:
                emergence_layer = layer
                break

        peak_layer = max(layer_accuracies, key=layer_accuracies.get)
        peak_accuracy = layer_accuracies[peak_layer]

        print(f"\n  Emergence layer (>70%): L{emergence_layer}")
        print(f"  Peak layer: L{peak_layer} ({peak_accuracy:.1%})")

        return FeatureEmergenceResult(
            feature_name="needs_tool",
            layer_accuracies=layer_accuracies,
            layer_aucs=layer_aucs,
            emergence_layer=emergence_layer,
            peak_layer=peak_layer,
            peak_accuracy=peak_accuracy,
        )

    def profile_neurons(
        self,
        layer: int,
        train_data: list[tuple[str, bool]],
    ) -> NeuronProfilingResult:
        """
        Profile neurons in a layer's MLP for tool discrimination.
        """
        print("\n" + "=" * 60)
        print(f"NEURON PROFILING: L{layer} MLP")
        print("=" * 60)

        # Collect MLP activations
        tool_activations = []
        no_tool_activations = []

        for prompt, label in train_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            try:
                mlp_act = self.get_mlp_activations(tokens, layer)
                # Last token
                last_token = np.array(mlp_act[0, -1, :].tolist())

                if label:
                    tool_activations.append(last_token)
                else:
                    no_tool_activations.append(last_token)
            except Exception as e:
                print(f"  Warning: Error getting MLP activations: {e}")
                continue

        if not tool_activations or not no_tool_activations:
            print("  Error: Could not collect activations")
            return None

        tool_activations = np.array(tool_activations)
        no_tool_activations = np.array(no_tool_activations)

        num_neurons = tool_activations.shape[1]
        print(f"  Total neurons: {num_neurons}")

        # Profile each neuron
        neuron_profiles = []

        for neuron_idx in range(num_neurons):
            tool_vals = tool_activations[:, neuron_idx]
            no_tool_vals = no_tool_activations[:, neuron_idx]

            mean_tool = float(np.mean(tool_vals))
            mean_no_tool = float(np.mean(no_tool_vals))
            std_tool = float(np.std(tool_vals))
            std_no_tool = float(np.std(no_tool_vals))

            # Pooled std
            pooled_std = np.sqrt((std_tool**2 + std_no_tool**2) / 2)
            if pooled_std > 1e-6:
                separation = abs(mean_tool - mean_no_tool) / pooled_std
            else:
                separation = 0.0

            # AUC for this neuron
            all_vals = np.concatenate([tool_vals, no_tool_vals])
            all_labels = np.array([1] * len(tool_vals) + [0] * len(no_tool_vals))
            try:
                auc = roc_auc_score(all_labels, all_vals)
                # Handle inverted neurons
                if auc < 0.5:
                    auc = 1 - auc
            except:
                auc = 0.5

            neuron_profiles.append(
                NeuronProfile(
                    layer=layer,
                    neuron_idx=neuron_idx,
                    mean_tool=mean_tool,
                    mean_no_tool=mean_no_tool,
                    std_tool=std_tool,
                    std_no_tool=std_no_tool,
                    separation_score=separation,
                    auc=auc,
                )
            )

        # Sort by separation score
        neuron_profiles.sort(key=lambda x: -x.separation_score)

        # Show top neurons
        print("\n  Top 20 discriminative neurons:")
        print(f"  {'Idx':>6} {'Sep':>8} {'AUC':>8} {'μ_tool':>10} {'μ_no_tool':>10}")
        print("  " + "-" * 50)

        for profile in neuron_profiles[:20]:
            print(
                f"  {profile.neuron_idx:>6} {profile.separation_score:>8.2f} "
                f"{profile.auc:>8.3f} {profile.mean_tool:>10.2f} {profile.mean_no_tool:>10.2f}"
            )

        # Cumulative accuracy: what accuracy do we get with top-k neurons?
        print("\n  Cumulative accuracy (top-k neurons):")

        cumulative_accuracy = []
        top_neuron_indices = [p.neuron_idx for p in neuron_profiles]

        for k in [1, 5, 10, 20, 50, 100, 200, 500]:
            if k > num_neurons:
                break

            selected_indices = top_neuron_indices[:k]

            # Build dataset with only these neurons
            X = np.concatenate(
                [tool_activations[:, selected_indices], no_tool_activations[:, selected_indices]]
            )
            y = np.array([1] * len(tool_activations) + [0] * len(no_tool_activations))

            # Cross-validated accuracy
            probe = LogisticRegression(max_iter=1000)
            scores = cross_val_score(probe, X, y, cv=min(5, len(y) // 2))
            acc = float(np.mean(scores))

            cumulative_accuracy.append((k, acc))
            print(f"    Top {k:4d} neurons: {acc:.1%}")

        return NeuronProfilingResult(
            layer=layer,
            total_neurons=num_neurons,
            top_neurons=neuron_profiles[:100],  # Keep top 100
            cumulative_accuracy=cumulative_accuracy,
        )

    def analyze_neuron_patterns(
        self,
        layer: int,
        top_neurons: list[NeuronProfile],
        train_data: list[tuple[str, bool]],
    ):
        """
        Analyze what patterns top neurons respond to.
        """
        print("\n" + "=" * 60)
        print(f"NEURON PATTERN ANALYSIS: L{layer}")
        print("=" * 60)

        # For top 10 neurons, find maximally activating prompts
        print("\n  What activates top neurons?")

        for profile in top_neurons[:10]:
            neuron_idx = profile.neuron_idx

            # Get activation for each prompt
            prompt_activations = []
            for prompt, label in train_data:
                tokens = self.tokenizer.encode(prompt)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()

                try:
                    mlp_act = self.get_mlp_activations(tokens, layer)
                    act_val = float(mlp_act[0, -1, neuron_idx])
                    prompt_activations.append((prompt, label, act_val))
                except:
                    continue

            # Sort by activation
            prompt_activations.sort(key=lambda x: -x[2])

            print(f"\n  Neuron {neuron_idx} (sep={profile.separation_score:.2f}):")
            print("    Top activating prompts:")
            for prompt, label, act in prompt_activations[:3]:
                label_str = "TOOL" if label else "NO_TOOL"
                print(f"      [{label_str}] {act:+.2f}: {prompt[:50]}")

            print("    Bottom activating prompts:")
            for prompt, label, act in prompt_activations[-3:]:
                label_str = "TOOL" if label else "NO_TOOL"
                print(f"      [{label_str}] {act:+.2f}: {prompt[:50]}")

    def run_experiment(self):
        """Run the full experiment."""
        print("=" * 60)
        print("FEATURE EMERGENCE + NEURON PROFILING")
        print("=" * 60)
        print("\nPhase 3.3 + 4.1: When does needs_tool emerge?")
        print("Which neurons in L11-12 are responsible?")

        self.load_model()

        # Prepare data
        train_data, test_data = self.prepare_dataset()
        print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")

        # Part 1: Feature emergence
        emergence_result = self.probe_feature_emergence(train_data, test_data)

        # Part 2: Neuron profiling for key layers
        neuron_results = {}
        for layer in [10, 11, 12, 13]:  # Focus on decision region
            result = self.profile_neurons(layer, train_data)
            if result:
                neuron_results[layer] = result

        # Part 3: Analyze top neurons
        if 11 in neuron_results:
            self.analyze_neuron_patterns(11, neuron_results[11].top_neurons, train_data)

        if 12 in neuron_results:
            self.analyze_neuron_patterns(12, neuron_results[12].top_neurons, train_data)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\nFeature Emergence:")
        print(f"  needs_tool emerges at: L{emergence_result.emergence_layer}")
        print(
            f"  Peak accuracy at: L{emergence_result.peak_layer} ({emergence_result.peak_accuracy:.1%})"
        )

        print("\nEmergence curve:")
        for layer in range(self.num_layers):
            acc = emergence_result.layer_accuracies[layer]
            bar = "█" * int(acc * 40)
            marker = " ← EMERGE" if layer == emergence_result.emergence_layer else ""
            marker = " ← PEAK" if layer == emergence_result.peak_layer else marker
            print(f"  L{layer:2d}: {bar} {acc:.1%}{marker}")

        print("\nNeuron Analysis:")
        for layer, result in neuron_results.items():
            if result.cumulative_accuracy:
                # Find minimum neurons for 90% accuracy
                for k, acc in result.cumulative_accuracy:
                    if acc >= 0.9:
                        print(f"  L{layer}: {k} neurons needed for 90% accuracy")
                        break
                else:
                    print(f"  L{layer}: Did not reach 90% accuracy")

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
Key findings:
1. If needs_tool emerges suddenly at L11-12 → confirms decision layer
2. If it's gradual L0→L17 → distributed processing
3. If few neurons (<50) needed → sparse, interpretable circuit
4. If many neurons (>200) needed → distributed representation

Next steps:
- Ablate top neurons and verify they're causal
- Patch neuron activations to flip decisions
- Name neurons based on what prompts activate them
""")

        # Save results
        results = {
            "emergence": {
                "feature": emergence_result.feature_name,
                "emergence_layer": emergence_result.emergence_layer,
                "peak_layer": emergence_result.peak_layer,
                "peak_accuracy": emergence_result.peak_accuracy,
                "layer_accuracies": emergence_result.layer_accuracies,
            },
            "neurons": {
                layer: {
                    "total_neurons": result.total_neurons,
                    "cumulative_accuracy": result.cumulative_accuracy,
                    "top_10_neurons": [
                        {
                            "idx": p.neuron_idx,
                            "separation": p.separation_score,
                            "auc": p.auc,
                        }
                        for p in result.top_neurons[:10]
                    ],
                }
                for layer, result in neuron_results.items()
            },
        }

        output_path = Path("feature_emergence_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return emergence_result, neuron_results


def main():
    experiment = FeatureEmergenceExperiment()
    experiment.run_experiment()


if __name__ == "__main__":
    main()
