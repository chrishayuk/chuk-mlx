#!/usr/bin/env python3
"""
qwen3_detailed_analysis.py

More detailed circuit analysis for Qwen3-0.6B.

Since probe accuracy is 100% at all layers, we need different metrics:
1. Ablation impact (which layers break the model)
2. Activation patching (which layers can flip decisions)
3. Representation change rate (where does information get written)

Run: uv run python examples/introspection/qwen3_detailed_analysis.py
"""

import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


class Qwen3DetailedAnalysis:
    """Detailed circuit analysis for Qwen3."""

    def __init__(self, model_id: str = "lmstudio-community/Qwen3-0.6B-MLX-bf16"):
        self.model_id = model_id

    def load_model(self):
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer
        self.config = study.adapter.config

        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)

        print(f"Loaded: {self.num_layers} layers")

    def get_all_layer_activations(self, prompt: str) -> list[np.ndarray]:
        """Get activations at all layers."""
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])

        # Forward through embedding
        h = self.model.model.embed_tokens(input_ids)

        # Create attention mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        activations = [np.array(h[0, -1, :].tolist())]

        # Forward through layers
        for layer in self.layers:
            output = layer(h, mask=mask, cache=None)
            if isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, 'hidden_states'):
                h = output.hidden_states
            else:
                h = output

            activations.append(np.array(h[0, -1, :].tolist()))

        return activations

    def analyze_representation_change(self):
        """Analyze where representations change most."""
        print("\n" + "=" * 60)
        print("REPRESENTATION CHANGE ANALYSIS")
        print("=" * 60)
        print("\nMeasuring how much representations change at each layer...")

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        test_prompts = [
            format_prompt("Calculate 25 + 17"),
            format_prompt("What is the capital of Spain?"),
            format_prompt("Write a poem about the moon"),
            format_prompt("Who invented electricity?"),
        ]

        # Collect activations
        all_layer_changes = np.zeros(self.num_layers)

        for prompt in test_prompts:
            acts = self.get_all_layer_activations(prompt)

            for i in range(self.num_layers):
                # Change from layer i to layer i+1
                prev = acts[i]
                curr = acts[i + 1]
                change = np.linalg.norm(curr - prev)
                all_layer_changes[i] += change

        all_layer_changes /= len(test_prompts)

        # Find peak change layers
        peak_layers = np.argsort(all_layer_changes)[-5:][::-1]

        print("\n  Layer change magnitudes:")
        for i in range(0, self.num_layers, 2):
            bar = "█" * int(all_layer_changes[i] / 5)
            print(f"  L{i:2d}: {all_layer_changes[i]:6.1f} {bar}")

        print(f"\n  Peak change layers: {peak_layers.tolist()}")

        return all_layer_changes

    def analyze_task_separation(self):
        """Analyze separation between task types at each layer."""
        print("\n" + "=" * 60)
        print("TASK SEPARATION ANALYSIS")
        print("=" * 60)
        print("\nMeasuring separation between action vs question prompts...")

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        action_prompts = [
            format_prompt("Calculate 25 * 4"),
            format_prompt("Write a haiku"),
            format_prompt("Sort: 5, 2, 8, 1"),
            format_prompt("Translate hello to French"),
        ]

        question_prompts = [
            format_prompt("What is gravity?"),
            format_prompt("Who is Shakespeare?"),
            format_prompt("When did WWII end?"),
            format_prompt("What causes thunder?"),
        ]

        # Collect activations at each layer
        separations = []

        for layer_idx in range(self.num_layers + 1):  # +1 for embedding
            action_acts = []
            question_acts = []

            for prompt in action_prompts:
                acts = self.get_all_layer_activations(prompt)
                action_acts.append(acts[layer_idx])

            for prompt in question_prompts:
                acts = self.get_all_layer_activations(prompt)
                question_acts.append(acts[layer_idx])

            action_mean = np.mean(action_acts, axis=0)
            question_mean = np.mean(question_acts, axis=0)
            sep = np.linalg.norm(action_mean - question_mean)
            separations.append(sep)

        # Find where separation increases most
        sep_increases = [separations[i+1] - separations[i] for i in range(len(separations)-1)]
        peak_increase_layers = np.argsort(sep_increases)[-5:][::-1]

        print("\n  Layer separations:")
        for i in range(0, len(separations), 2):
            bar = "█" * int(separations[i] / 10)
            print(f"  L{i:2d}: {separations[i]:6.1f} {bar}")

        print(f"\n  Peak increase layers: {peak_increase_layers.tolist()}")

        return separations, sep_increases

    def analyze_output_formation(self):
        """Analyze where output tokens are decided."""
        print("\n" + "=" * 60)
        print("OUTPUT FORMATION ANALYSIS")
        print("=" * 60)
        print("\nSkipping logit analysis (requires tied embeddings handling).")
        print("See representation change analysis above for layer importance.")

        return []

    def compare_to_gemma(self):
        """Generate comparison summary."""
        print("\n" + "=" * 60)
        print("COMPARISON: QWEN3 vs FUNCTIONGEMMA")
        print("=" * 60)

        print("""
┌─────────────────────────────────────────────────────────────────┐
│                     FUNCTIONGEMMA 270M                          │
├─────────────────────────────────────────────────────────────────┤
│  L0-L5:  COMPUTE ZONE (decision made, 50% drop if ablated)     │
│  L6-L10: STABLE ZONE (0% drop, decision locked)                │
│  L11:    CONTROL POINT (100% flip rate, 86% on PC1)            │
│  L12:    GATE (kill switch neuron 230)                         │
│  L13-17: FORMATION (output formatting)                         │
└─────────────────────────────────────────────────────────────────┘

Key Finding: Decision is made EARLY (L0-L5), not late.
             L11-12 are for reading/formatting, not computing.

┌─────────────────────────────────────────────────────────────────┐
│                     QWEN3 0.6B (28 layers)                      │
├─────────────────────────────────────────────────────────────────┤
│  100% probe accuracy from L0 - task info present immediately    │
│  Separation grows monotonically (no U-shape)                    │
│  PC1 variance: 34.9% (vs 86% in Gemma) - more distributed      │
│  Peak change layers identify compute zones                      │
└─────────────────────────────────────────────────────────────────┘
""")

    def run_all(self):
        """Run all analyses."""
        self.load_model()

        change_results = self.analyze_representation_change()
        sep_results, sep_increases = self.analyze_task_separation()
        logit_results = self.analyze_output_formation()

        self.compare_to_gemma()

        # Save results
        results = {
            'model': self.model_id,
            'num_layers': self.num_layers,
            'representation_changes': change_results.tolist(),
            'task_separations': sep_results,
            'separation_increases': sep_increases,
            'logit_changes': logit_results,
        }

        with open('qwen3_detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to qwen3_detailed_results.json")


def main():
    analyzer = Qwen3DetailedAnalysis()
    analyzer.run_all()


if __name__ == "__main__":
    main()
