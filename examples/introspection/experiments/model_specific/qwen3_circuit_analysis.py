#!/usr/bin/env python3
"""
qwen3_circuit_analysis.py

Run circuit analysis on Qwen3-0.6B to compare with FunctionGemma findings.

Key questions:
1. Is there a similar compute/stable/control zone structure?
2. Are decisions made early (L0-L5) or late?
3. Are there kill switch neurons?
4. Is the decision boundary nearly 1D?

Run: uv run python examples/introspection/qwen3_circuit_analysis.py
"""

import json
import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class Qwen3CircuitAnalysis:
    """Circuit analysis for Qwen3 models."""

    def __init__(self, model_id: str = "lmstudio-community/Qwen3-0.6B-MLX-bf16"):
        self.model_id = model_id

    def load_model(self):
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer
        self.config = study.adapter.config

        self.embed_layer = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.hidden_size = self.config.hidden_size
        self.num_layers = len(self.layers)

        print(f"Loaded: {self.num_layers} layers, hidden size {self.hidden_size}")

    def get_layer_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get activations at a specific layer for the last token."""
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])

        # Forward through embedding
        h = self.embed_layer(input_ids)

        # Create attention mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Forward through layers
        for i, layer in enumerate(self.layers):
            output = layer(h, mask=mask, cache=None)
            if isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, "hidden_states"):
                h = output.hidden_states
            else:
                h = output

            if i == layer_idx:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    def get_mlp_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP intermediate activations at a specific layer."""
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])

        # Forward through embedding
        h = self.embed_layer(input_ids)

        # Create attention mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Forward through layers
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                # Get MLP activations specifically
                if hasattr(layer, "input_layernorm"):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                # Attention
                attn_out = layer.self_attn(normed, mask=mask, cache=None)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                # MLP input
                if hasattr(layer, "post_attention_layernorm"):
                    mlp_input = layer.post_attention_layernorm(h)
                else:
                    mlp_input = h

                # Get MLP intermediate
                mlp = layer.mlp
                if hasattr(mlp, "gate_proj"):
                    gate = mlp.gate_proj(mlp_input)
                    up = mlp.up_proj(mlp_input)
                    # SwiGLU: silu(gate) * up
                    mlp_hidden = nn.silu(gate) * up
                    return np.array(mlp_hidden[0, -1, :].tolist())
                else:
                    # Fallback
                    return np.array(mlp_input[0, -1, :].tolist())

            # Normal forward
            output = layer(h, mask=mask, cache=None)
            if isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, "hidden_states"):
                h = output.hidden_states
            else:
                h = output

        return np.array(h[0, -1, :].tolist())

    def run_probe_analysis(self):
        """Test when tool/no-tool becomes decodable."""
        print("\n" + "=" * 60)
        print("PROBE ANALYSIS: When does task type become decodable?")
        print("=" * 60)

        # For Qwen3, we'll use a general task classification
        # (since it's not specifically a function-calling model)

        # Task type 1: Action/command prompts
        action_prompts = [
            "Calculate 25 * 4",
            "Translate 'hello' to French",
            "Write a haiku about spring",
            "Sort these numbers: 5, 2, 8, 1",
            "Convert 100 celsius to fahrenheit",
            "Summarize this text: The quick brown fox",
            "Generate a random password",
            "Count the words in: Hello world today",
        ]

        # Task type 2: Question/factual prompts
        question_prompts = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?",
            "When did World War 2 end?",
            "What is the speed of light?",
            "Who invented the telephone?",
            "What is the largest planet?",
            "What causes earthquakes?",
        ]

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        print("\n1. Collecting activations for each layer...")

        layer_results = {}

        for layer_idx in range(0, self.num_layers, 2):  # Sample every 2 layers
            X = []
            y = []

            for prompt in action_prompts:
                formatted = format_prompt(prompt)
                act = self.get_layer_activations(formatted, layer_idx)
                X.append(act)
                y.append(1)  # Action

            for prompt in question_prompts:
                formatted = format_prompt(prompt)
                act = self.get_layer_activations(formatted, layer_idx)
                X.append(act)
                y.append(0)  # Question

            X = np.array(X)
            y = np.array(y)

            # Train probe
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, y)
            acc = probe.score(X, y)

            # Compute mean separation
            action_mean = X[y == 1].mean(axis=0)
            question_mean = X[y == 0].mean(axis=0)
            separation = np.linalg.norm(action_mean - question_mean)

            layer_results[layer_idx] = {
                "accuracy": acc,
                "separation": separation,
            }

            print(f"   L{layer_idx:2d}: Probe accuracy = {acc:.1%}, Separation = {separation:.1f}")

        return layer_results

    def run_mlp_ablation(self):
        """Test which MLP layers are critical."""
        print("\n" + "=" * 60)
        print("MLP ABLATION: Which layers are critical?")
        print("=" * 60)

        # Use a simple task: math vs factual
        test_prompts = [
            ("Calculate 5 + 3", "8"),
            ("What is 10 * 2", "20"),
            ("What is the capital of Japan?", "Tokyo"),
            ("Who painted the Mona Lisa?", "Leonardo"),
        ]

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        print("\n1. Getting baseline predictions...")

        baseline_correct = 0
        for query, expected in test_prompts:
            formatted = format_prompt(query)
            tokens = self.tokenizer.encode(formatted)
            input_ids = mx.array([tokens])

            output = self.model(input_ids)
            logits = output.logits[0, -1, :]
            top_idx = int(mx.argmax(logits))
            top_token = self.tokenizer.decode([top_idx])

            # Simple check - does output contain expected
            if expected.lower() in top_token.lower() or top_token.strip() in expected:
                baseline_correct += 1

        baseline_acc = baseline_correct / len(test_prompts)
        print(f"   Baseline accuracy: {baseline_acc:.1%}")

        # For Qwen3, just report probe results since ablation requires modifying weights
        print("\nNote: Full ablation requires weight modification. See probe results above.")

    def run_neuron_analysis(self):
        """Analyze individual neuron profiles."""
        print("\n" + "=" * 60)
        print("NEURON ANALYSIS: Finding discriminative neurons")
        print("=" * 60)

        action_prompts = [
            "Calculate 25 * 4",
            "Write a poem",
            "Sort these numbers",
            "Translate to Spanish",
        ]

        question_prompts = [
            "What is gravity?",
            "Who is Einstein?",
            "When did WWII end?",
            "What causes rain?",
        ]

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        # Analyze middle layer (equivalent to L11 in 18-layer model would be ~L17-18 in 28-layer)
        target_layer = self.num_layers * 11 // 18  # Scale to equivalent position

        print(f"\n1. Collecting MLP activations at layer {target_layer}...")

        action_acts = []
        question_acts = []

        for prompt in action_prompts:
            formatted = format_prompt(prompt)
            act = self.get_mlp_activations(formatted, target_layer)
            action_acts.append(act)

        for prompt in question_prompts:
            formatted = format_prompt(prompt)
            act = self.get_mlp_activations(formatted, target_layer)
            question_acts.append(act)

        action_acts = np.array(action_acts)
        question_acts = np.array(question_acts)

        # Compute mean difference for each neuron
        action_mean = action_acts.mean(axis=0)
        question_mean = question_acts.mean(axis=0)
        diff = action_mean - question_mean

        # Find most discriminative neurons
        top_action = np.argsort(diff)[-10:][::-1]
        top_question = np.argsort(diff)[:10]

        print(f"\n2. Top action-promoting neurons (layer {target_layer}):")
        for n in top_action[:5]:
            print(f"   Neuron {n}: diff = {diff[n]:+.1f}")

        print(f"\n3. Top question-promoting neurons (layer {target_layer}):")
        for n in top_question[:5]:
            print(f"   Neuron {n}: diff = {diff[n]:+.1f}")

        # PCA analysis
        all_acts = np.vstack([action_acts, question_acts])
        all_labels = np.array([1] * len(action_acts) + [0] * len(question_acts))

        pca = PCA(n_components=3)
        pca.fit(all_acts)

        print("\n4. PCA variance explained:")
        for i, var in enumerate(pca.explained_variance_ratio_[:3]):
            print(f"   PC{i + 1}: {var:.1%}")

        return {
            "target_layer": target_layer,
            "action_neurons": top_action.tolist(),
            "question_neurons": top_question.tolist(),
            "pc1_variance": float(pca.explained_variance_ratio_[0]),
        }

    def run_comparison(self):
        """Run full analysis and compare to FunctionGemma."""
        print("\n" + "=" * 60)
        print("QWEN3-0.6B CIRCUIT ANALYSIS")
        print("=" * 60)
        print("\nComparing to FunctionGemma 270M findings:")
        print("  - FunctionGemma: 18 layers, hidden=640, intermediate=2048")
        print("  - Qwen3-0.6B: 28 layers, hidden=1024, intermediate=2816")

        self.load_model()

        probe_results = self.run_probe_analysis()
        neuron_results = self.run_neuron_analysis()

        # Summary comparison
        print("\n" + "=" * 60)
        print("COMPARISON WITH FUNCTIONGEMMA CIRCUIT")
        print("=" * 60)

        print("""
FunctionGemma 270M:
  - Decision computed: L0-L5 (early)
  - Stable zone: L6-L10 (0% drop)
  - Control point: L11 (100% flip rate)
  - Gate (kill switch): L12
  - PC1 variance at L11: 86%

Qwen3-0.6B (this analysis):""")

        # Find early high accuracy
        early_acc = [probe_results[l]["accuracy"] for l in sorted(probe_results.keys())[:5]]
        print(f"  - Early layer accuracy: {max(early_acc):.1%}")

        # Find when accuracy stabilizes
        accs = [(l, probe_results[l]["accuracy"]) for l in sorted(probe_results.keys())]
        print(f"  - Layer probe accuracies: {[(l, f'{a:.1%}') for l, a in accs]}")

        print(f"  - Target layer (scaled L11): {neuron_results['target_layer']}")
        print(f"  - PC1 variance: {neuron_results['pc1_variance']:.1%}")
        print(f"  - Top action neurons: {neuron_results['action_neurons'][:5]}")
        print(f"  - Top question neurons: {neuron_results['question_neurons'][:5]}")

        # Save results
        results = {
            "model": self.model_id,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "probe_results": {str(k): v for k, v in probe_results.items()},
            "neuron_analysis": neuron_results,
        }

        with open("qwen3_circuit_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to qwen3_circuit_results.json")

        return results


def main():
    analyzer = Qwen3CircuitAnalysis()
    analyzer.run_comparison()


if __name__ == "__main__":
    main()
