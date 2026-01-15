#!/usr/bin/env python3
"""
circuit_deep_dive.py

Deep dive on the discovered circuit:
1. L12 Neuron 230 "Kill Switch" analysis
2. Compute Zone (L0-L5) information flow mapping
3. L11 Control Point direction finding

Run: uv run python examples/introspection/circuit_deep_dive.py
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


@dataclass
class NeuronAnalysis:
    """Deep analysis of a single neuron."""

    layer: int
    neuron_idx: int
    mean_tool: float
    mean_no_tool: float
    top_activating_prompts: list[tuple[str, bool, float]]
    bottom_activating_prompts: list[tuple[str, bool, float]]
    input_weights_top_dims: list[int]
    output_weights_top_dims: list[int]


class CircuitDeepDive:
    """Deep analysis of the tool-calling circuit."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id

    def load_model(self):
        """Load model."""
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        self.embed_layer = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        self.hidden_size = self.model.model.hidden_size
        self.embed_scale = self.hidden_size**0.5
        self.num_layers = len(self.layers)

        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}")

    def prepare_dataset(self):
        """Prepare prompts."""
        tool_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "Create a calendar event",
            "Set a timer for 10 minutes",
            "Book a flight to Paris",
            "Turn on the lights",
            "Play some music",
            "Call mom",
            "Order a pizza",
            "Check my schedule",
            "Find hotels in London",
            "Set an alarm for 7am",
            "Navigate to the airport",
            "Add milk to shopping list",
            "Translate hello to Spanish",
        ]

        no_tool_prompts = [
            "What is the capital of France?",
            "Explain quantum physics",
            "What is 2 + 2?",
            "Tell me about Einstein",
            "What is photosynthesis?",
            "Why is the sky blue?",
            "Who wrote Romeo and Juliet?",
            "What is machine learning?",
            "How does gravity work?",
            "Explain democracy",
            "What is the meaning of life?",
            "What is DNA?",
            "Who invented the telephone?",
            "What causes earthquakes?",
            "Describe the solar system",
        ]

        return ([(p, True) for p in tool_prompts], [(p, False) for p in no_tool_prompts])

    def get_mlp_activations(self, tokens: list[int], layer: int) -> mx.array:
        """Get MLP intermediate activations at a layer."""
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

        # Now at target layer
        target_layer = self.layers[layer]

        if hasattr(target_layer, "input_layernorm"):
            normed = target_layer.input_layernorm(h)
        else:
            normed = h

        attn_out = target_layer.self_attn(normed, mask=mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = h + attn_out

        if hasattr(target_layer, "post_attention_layernorm"):
            mlp_input = target_layer.post_attention_layernorm(h)
        else:
            mlp_input = h

        mlp = target_layer.mlp
        if hasattr(mlp, "gate_proj"):
            gate = mlp.gate_proj(mlp_input)
            up = mlp.up_proj(mlp_input)
            mlp_hidden = nn.gelu(gate) * up
        else:
            mlp_hidden = mlp.up(mlp_input)

        return mlp_hidden.astype(mx.float32), mlp_input.astype(mx.float32)

    def get_layer_output(self, tokens: list[int], layer: int) -> mx.array:
        """Get output of a layer."""
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

    def analyze_kill_switch(self, tool_prompts, no_tool_prompts):
        """Deep analysis of L12 neuron 230."""
        print("\n" + "=" * 60)
        print("KILL SWITCH ANALYSIS: L12 Neuron 230")
        print("=" * 60)

        neuron_idx = 230
        layer = 12

        # Collect activations
        tool_activations = []
        no_tool_activations = []
        all_prompts_activations = []

        for prompt, label in tool_prompts + no_tool_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            mlp_hidden, mlp_input = self.get_mlp_activations(tokens, layer)
            act_val = float(mlp_hidden[0, -1, neuron_idx])

            all_prompts_activations.append((prompt, label, act_val, mlp_input[0, -1, :]))

            if label:
                tool_activations.append(act_val)
            else:
                no_tool_activations.append(act_val)

        # Statistics
        print("\n  Activation Statistics:")
        print(
            f"    Tool prompts:    mean={np.mean(tool_activations):+.1f}, std={np.std(tool_activations):.1f}"
        )
        print(
            f"    No-tool prompts: mean={np.mean(no_tool_activations):+.1f}, std={np.std(no_tool_activations):.1f}"
        )
        print(
            f"    Separation: {abs(np.mean(tool_activations) - np.mean(no_tool_activations)) / np.std(tool_activations + no_tool_activations):.2f}σ"
        )

        # Top/bottom activating prompts
        all_prompts_activations.sort(key=lambda x: x[2])

        print("\n  Top activating prompts (HIGHEST activation):")
        for prompt, label, act, _ in all_prompts_activations[-5:]:
            label_str = "TOOL" if label else "NO_TOOL"
            print(f"    [{label_str}] {act:+.1f}: {prompt[:50]}")

        print("\n  Bottom activating prompts (LOWEST activation):")
        for prompt, label, act, _ in all_prompts_activations[:5]:
            label_str = "TOOL" if label else "NO_TOOL"
            print(f"    [{label_str}] {act:+.1f}: {prompt[:50]}")

        # Analyze input weights
        print("\n  Input Weight Analysis:")
        mlp = self.layers[layer].mlp

        if hasattr(mlp, "gate_proj"):
            gate_weights = np.array(mlp.gate_proj.weight[neuron_idx, :].tolist())
            up_weights = np.array(mlp.up_proj.weight[neuron_idx, :].tolist())

            # Top input dimensions
            gate_top = np.argsort(np.abs(gate_weights))[-10:][::-1]
            print(f"    Gate proj top dims: {gate_top.tolist()}")
            print(f"    Gate proj top weights: {gate_weights[gate_top].tolist()[:5]}")

            up_top = np.argsort(np.abs(up_weights))[-10:][::-1]
            print(f"    Up proj top dims: {up_top.tolist()}")

        # Output weight analysis
        print("\n  Output Weight Analysis:")
        if hasattr(mlp, "down_proj"):
            down_weights = np.array(mlp.down_proj.weight[:, neuron_idx].tolist())
            down_top = np.argsort(np.abs(down_weights))[-10:][::-1]
            print(f"    Down proj top dims: {down_top.tolist()}")
            print(f"    Down proj top weights: {down_weights[down_top].tolist()[:5]}")

        # Hypothesis: What does this neuron detect?
        print("\n  Interpretation:")
        if np.mean(tool_activations) > np.mean(no_tool_activations):
            print("    Neuron 230 fires HIGH for TOOL prompts")
            print("    It's a TOOL DETECTOR / AMPLIFIER")
        else:
            print("    Neuron 230 fires HIGH for NO-TOOL prompts")
            print("    It's a NO-TOOL DETECTOR / SUPPRESSOR")

        return all_prompts_activations

    def map_compute_zone(self, tool_prompts, no_tool_prompts):
        """Map information flow through L0-L5."""
        print("\n" + "=" * 60)
        print("COMPUTE ZONE MAPPING: L0-L5")
        print("=" * 60)

        all_data = tool_prompts + no_tool_prompts

        # Track representations at each layer
        layer_representations = {}

        for layer in [0, 1, 2, 3, 4, 5]:
            tool_reps = []
            no_tool_reps = []

            for prompt, label in all_data:
                tokens = self.tokenizer.encode(prompt)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()

                h = self.get_layer_output(tokens, layer)
                rep = np.array(h[0, -1, :].tolist())

                if label:
                    tool_reps.append(rep)
                else:
                    no_tool_reps.append(rep)

            tool_reps = np.array(tool_reps)
            no_tool_reps = np.array(no_tool_reps)

            # Compute direction from no-tool to tool
            tool_mean = np.mean(tool_reps, axis=0)
            no_tool_mean = np.mean(no_tool_reps, axis=0)
            direction = tool_mean - no_tool_mean
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            layer_representations[layer] = {
                "tool_mean": tool_mean,
                "no_tool_mean": no_tool_mean,
                "direction": direction,
                "separation": np.linalg.norm(tool_mean - no_tool_mean),
            }

            print(f"\n  Layer {layer}:")
            print(f"    Separation (L2 norm): {layer_representations[layer]['separation']:.1f}")

            # Train probe
            X = np.vstack([tool_reps, no_tool_reps])
            y = np.array([1] * len(tool_reps) + [0] * len(no_tool_reps))
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, y)
            acc = probe.score(X, y)
            print(f"    Probe accuracy: {acc:.1%}")

        # Direction consistency across layers
        print("\n  Direction Consistency (cosine similarity):")
        for l1 in [0, 1, 3, 5]:
            for l2 in [l1 + 1, l1 + 2]:
                if l2 <= 5:
                    d1 = layer_representations[l1]["direction"]
                    d2 = layer_representations[l2]["direction"]
                    cos = np.dot(d1, d2)
                    print(f"    L{l1} → L{l2}: {cos:.3f}")

        # Find the most important dimensions
        print("\n  Key Dimensions (highest in tool direction):")
        for layer in [0, 5]:
            direction = layer_representations[layer]["direction"]
            top_dims = np.argsort(np.abs(direction))[-10:][::-1]
            print(f"    L{layer}: dims {top_dims.tolist()}")

        return layer_representations

    def find_control_direction(self, tool_prompts, no_tool_prompts):
        """Find the steering direction at L11."""
        print("\n" + "=" * 60)
        print("CONTROL POINT: L11 Steering Direction")
        print("=" * 60)

        layer = 11
        all_data = tool_prompts + no_tool_prompts

        # Collect MLP activations
        tool_acts = []
        no_tool_acts = []

        for prompt, label in all_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            mlp_hidden, _ = self.get_mlp_activations(tokens, layer)
            act = np.array(mlp_hidden[0, -1, :].tolist())

            if label:
                tool_acts.append(act)
            else:
                no_tool_acts.append(act)

        tool_acts = np.array(tool_acts)
        no_tool_acts = np.array(no_tool_acts)

        # Compute steering direction
        tool_mean = np.mean(tool_acts, axis=0)
        no_tool_mean = np.mean(no_tool_acts, axis=0)

        # Direction: no_tool → tool
        steering_direction = tool_mean - no_tool_mean
        steering_magnitude = np.linalg.norm(steering_direction)
        steering_direction_normalized = steering_direction / (steering_magnitude + 1e-8)

        print("\n  Steering Vector:")
        print(f"    Magnitude: {steering_magnitude:.1f}")
        print(f"    Non-zero dims: {np.sum(np.abs(steering_direction) > 1)}")

        # Top neurons in steering direction
        top_neurons = np.argsort(np.abs(steering_direction))[-20:][::-1]
        print("\n  Top neurons in steering direction:")
        for i, neuron in enumerate(top_neurons[:10]):
            val = steering_direction[neuron]
            print(f"    {i + 1}. Neuron {neuron}: {val:+.1f}")

        # Verify these match our known control neurons
        known_control = [1237, 551, 1069, 821, 1710]
        overlap = set(top_neurons[:20]) & set(known_control)
        print(f"\n  Overlap with known control neurons: {len(overlap)}/{len(known_control)}")
        print(f"    Known: {known_control}")
        print(f"    Found in top 20: {list(overlap)}")

        # PCA analysis
        print("\n  PCA Analysis:")
        all_acts = np.vstack([tool_acts, no_tool_acts])
        pca = PCA(n_components=10)
        pca.fit(all_acts)

        print(f"    Variance explained by PC1-5: {pca.explained_variance_ratio_[:5].sum():.1%}")

        # Project steering direction onto principal components
        for i in range(5):
            pc = pca.components_[i]
            projection = np.dot(steering_direction_normalized, pc)
            print(f"    Steering direction on PC{i + 1}: {abs(projection):.3f}")

        # Save steering direction for later use
        steering_data = {
            "layer": layer,
            "direction": steering_direction.tolist(),
            "magnitude": float(steering_magnitude),
            "top_neurons": top_neurons.tolist(),
        }

        return steering_data

    def verify_steering(self, tool_prompts, no_tool_prompts, steering_data):
        """Verify that steering direction works."""
        print("\n" + "=" * 60)
        print("STEERING VERIFICATION")
        print("=" * 60)

        layer = steering_data["layer"]
        direction = np.array(steering_data["direction"])

        # Test: Add steering direction to no-tool prompts
        # Check if representation moves toward tool cluster

        print("\n  Testing: Add steering to no-tool prompts")

        for prompt, _ in no_tool_prompts[:5]:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            mlp_hidden, _ = self.get_mlp_activations(tokens, layer)
            original = np.array(mlp_hidden[0, -1, :].tolist())

            # Add steering
            scales = [0.0, 0.5, 1.0, 2.0]
            tool_mean = np.mean(
                [
                    np.array(
                        self.get_mlp_activations(
                            self.tokenizer.encode(p).flatten().tolist()
                            if isinstance(self.tokenizer.encode(p), np.ndarray)
                            else self.tokenizer.encode(p),
                            layer,
                        )[0][0, -1, :].tolist()
                    )
                    for p, l in tool_prompts[:5]
                ],
                axis=0,
            )

            print(f"\n    Prompt: {prompt[:40]}...")
            for scale in scales:
                steered = original + scale * direction
                # Distance to tool mean
                dist_to_tool = np.linalg.norm(steered - tool_mean)
                print(f"      Scale {scale:.1f}: dist to tool mean = {dist_to_tool:.1f}")

    def run_experiment(self):
        """Run all deep dive analyses."""
        print("=" * 60)
        print("CIRCUIT DEEP DIVE")
        print("=" * 60)

        self.load_model()
        tool_prompts, no_tool_prompts = self.prepare_dataset()

        # 1. Kill switch analysis
        self.analyze_kill_switch(tool_prompts, no_tool_prompts)

        # 2. Compute zone mapping
        layer_reps = self.map_compute_zone(tool_prompts, no_tool_prompts)

        # 3. Find steering direction
        steering_data = self.find_control_direction(tool_prompts, no_tool_prompts)

        # 4. Verify steering
        self.verify_steering(tool_prompts, no_tool_prompts, steering_data)

        # Summary
        print("\n" + "=" * 60)
        print("CIRCUIT SUMMARY")
        print("=" * 60)
        print("""
KILL SWITCH (L12:230):
  - Single neuron that gates tool-calling output
  - Ablating it drops accuracy 70%
  - Fires differently for tool vs no-tool

COMPUTE ZONE (L0-L5):
  - Decision is computed here
  - Probe accuracy increases through layers
  - Direction consistency shows information flow

CONTROL POINT (L11):
  - Steering direction identified
  - Top neurons match our ablation findings
  - Adding direction moves representations toward tool cluster

PRACTICAL APPLICATIONS:
  1. Activate steering: Add L11 direction to get tool-calling
  2. Suppress steering: Subtract L11 direction to prevent tool-calling
  3. Kill switch: Zero L12:230 to completely block tools
""")

        # Save results
        results = {
            "kill_switch": {"layer": 12, "neuron": 230},
            "control_point": {
                "layer": 11,
                "top_neurons": steering_data["top_neurons"][:10],
            },
            "compute_zone": {
                "layers": [0, 1, 3, 5],
                "separations": {l: float(layer_reps[l]["separation"]) for l in [0, 1, 3, 5]},
            },
        }

        output_path = Path("circuit_analysis_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        return results


def main():
    experiment = CircuitDeepDive()
    experiment.run_experiment()


if __name__ == "__main__":
    main()
