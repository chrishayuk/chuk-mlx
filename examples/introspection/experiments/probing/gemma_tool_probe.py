#!/usr/bin/env python3
"""
gemma_tool_probe.py

Compare base Gemma vs FunctionGemma on the ACTUAL tool-calling task.

The previous test used action vs question - too easy (100% everywhere).
This test uses tool-calling vs no-tool prompts, which is what
FunctionGemma was specifically trained for.

Run: uv run python examples/introspection/gemma_tool_probe.py
"""

import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


def format_tool_prompt(query: str) -> str:
    """Format prompt with tool definitions (like FunctionGemma expects)."""
    tools = [
        {"name": "get_weather", "description": "Get weather for a location"},
        {"name": "send_email", "description": "Send an email"},
        {"name": "set_timer", "description": "Set a timer"},
    ]

    return f"""<start_of_turn>developer
You have access to functions: {json.dumps(tools)}
<end_of_turn>
<start_of_turn>user
{query}
<end_of_turn>
<start_of_turn>model
"""


class GemmaToolProbe:
    """Compare Gemma models on tool-calling task."""

    def __init__(self):
        self.models = {
            'base': 'mlx-community/gemma-3-270m-it-bf16',
            'function': 'mlx-community/functiongemma-270m-it-bf16',
        }
        self.results = {}

    def load_model(self, model_id: str):
        """Load a model."""
        print(f"\nLoading: {model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return {
            'model': study.adapter.model,
            'tokenizer': study.adapter.tokenizer,
            'config': study.adapter.config,
            'layers': study.adapter.model.model.layers,
            'num_layers': len(study.adapter.model.model.layers),
        }

    def get_layer_activations(self, model_data: dict, prompt: str, layer_idx: int) -> np.ndarray:
        """Get activations at a specific layer."""
        model = model_data['model']
        tokenizer = model_data['tokenizer']

        tokens = tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])

        h = model.model.embed_tokens(input_ids)
        hidden_size = model_data['config'].hidden_size
        h = h * mx.array(hidden_size ** 0.5, dtype=h.dtype)

        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(model_data['layers']):
            output = layer(h, mask=mask, cache=None)
            if hasattr(output, 'hidden_states'):
                h = output.hidden_states
            elif isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            if i == layer_idx:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    def run_tool_probe(self, model_data: dict, model_name: str):
        """Run probe on tool-calling task."""
        print(f"\n{'='*60}")
        print(f"TOOL-CALLING PROBE: {model_name}")
        print('='*60)

        # Prompts that SHOULD trigger tool calls
        tool_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John about the meeting",
            "Set a timer for 10 minutes",
            "What's the temperature in Paris?",
            "Email Alice the report",
            "Set an alarm for 7am",
            "Check the weather forecast",
            "Send a message to Bob",
        ]

        # Prompts that should NOT trigger tool calls
        no_tool_prompts = [
            "What is the capital of France?",
            "Explain quantum physics",
            "Who wrote Romeo and Juliet?",
            "What is 2 + 2?",
            "Tell me about Einstein",
            "How does gravity work?",
            "What is photosynthesis?",
            "Why is the sky blue?",
        ]

        num_layers = model_data['num_layers']
        accuracies = []
        separations = []

        print("\nLayer-by-layer probe accuracy (tool vs no-tool):")

        for layer_idx in range(num_layers):
            X = []
            y = []

            for query in tool_prompts:
                prompt = format_tool_prompt(query)
                act = self.get_layer_activations(model_data, prompt, layer_idx)
                X.append(act)
                y.append(1)  # Tool

            for query in no_tool_prompts:
                prompt = format_tool_prompt(query)
                act = self.get_layer_activations(model_data, prompt, layer_idx)
                X.append(act)
                y.append(0)  # No tool

            X = np.array(X)
            y = np.array(y)

            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, y)
            acc = probe.score(X, y)
            accuracies.append(acc)

            action_mean = X[y == 1].mean(axis=0)
            question_mean = X[y == 0].mean(axis=0)
            sep = np.linalg.norm(action_mean - question_mean)
            separations.append(sep)

            bar = "█" * int(acc * 20)
            print(f"  L{layer_idx:2d}: {acc:.1%} {bar} (sep={sep:.1f})")

        # U-shape check
        early_acc = np.mean(accuracies[:3])
        mid_acc = np.mean(accuracies[5:10])
        late_acc = np.mean(accuracies[-5:])

        print(f"\nU-shape check:")
        print(f"  Early (L0-2):  {early_acc:.1%}")
        print(f"  Middle (L5-9): {mid_acc:.1%}")
        print(f"  Late (L13-17): {late_acc:.1%}")

        # Detect U-shape
        if mid_acc < early_acc - 0.05 and late_acc > mid_acc + 0.05:
            print("  → U-SHAPE DETECTED!")
            u_shape = True
        elif all(accuracies[i] <= accuracies[i+1] + 0.02 for i in range(len(accuracies)-1)):
            print("  → MONOTONIC (no dip)")
            u_shape = False
        else:
            print("  → IRREGULAR pattern")
            u_shape = False

        return {
            'accuracies': accuracies,
            'separations': separations,
            'early_acc': early_acc,
            'mid_acc': mid_acc,
            'late_acc': late_acc,
            'u_shape': u_shape,
        }

    def run_pca_at_key_layers(self, model_data: dict, model_name: str):
        """Check PCA at multiple layers."""
        print(f"\n{'='*60}")
        print(f"PCA ANALYSIS: {model_name}")
        print('='*60)

        tool_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "Set a timer for 5 minutes",
            "Check temperature in Paris",
            "Email the report to Alice",
        ]

        no_tool_prompts = [
            "What is gravity?",
            "Who is Einstein?",
            "Explain photosynthesis",
            "What causes rain?",
            "How does evolution work?",
        ]

        key_layers = [0, 5, 11, 17]  # Embedding, early, middle, late

        print("\nPC1 variance by layer:")

        pc1_variances = {}

        for layer_idx in key_layers:
            X = []

            for query in tool_prompts + no_tool_prompts:
                prompt = format_tool_prompt(query)
                act = self.get_layer_activations(model_data, prompt, layer_idx)
                X.append(act)

            X = np.array(X)

            pca = PCA(n_components=3)
            pca.fit(X)

            pc1_var = pca.explained_variance_ratio_[0]
            pc1_variances[layer_idx] = pc1_var

            bar = "█" * int(pc1_var * 50)
            print(f"  L{layer_idx:2d}: {pc1_var:.1%} {bar}")

        # Check if L11 is special (high PC1 variance)
        if 11 in pc1_variances:
            l11_var = pc1_variances[11]
            if l11_var > 0.7:
                print(f"\n→ L11 is nearly 1D ({l11_var:.1%}) - CONTROL POINT!")
            elif l11_var > 0.5:
                print(f"\n→ L11 is somewhat 1D ({l11_var:.1%})")
            else:
                print(f"\n→ L11 is multi-dimensional ({l11_var:.1%})")

        return pc1_variances

    def run_comparison(self):
        """Run full comparison."""
        print("\n" + "=" * 60)
        print("BASE GEMMA vs FUNCTIONGEMMA: TOOL-CALLING TASK")
        print("=" * 60)
        print("\nThis test uses actual tool-calling prompts to reveal")
        print("circuit differences between base and tool-trained models.")

        for name, model_id in self.models.items():
            model_data = self.load_model(model_id)

            probe_results = self.run_tool_probe(model_data, name)
            pca_results = self.run_pca_at_key_layers(model_data, name)

            self.results[name] = {
                'probe': probe_results,
                'pca': pca_results,
            }

        # Final comparison
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        base = self.results['base']
        func = self.results['function']

        print("\n┌────────────────────────────────────────────────────────┐")
        print("│                    BASE GEMMA                           │")
        print("├────────────────────────────────────────────────────────┤")
        print(f"│  U-shape: {'YES' if base['probe']['u_shape'] else 'NO':4s}                                        │")
        print(f"│  Early acc: {base['probe']['early_acc']:.1%}                                     │")
        print(f"│  Mid acc:   {base['probe']['mid_acc']:.1%}                                     │")
        print(f"│  Late acc:  {base['probe']['late_acc']:.1%}                                     │")
        if 11 in base['pca']:
            print(f"│  L11 PC1:   {base['pca'][11]:.1%}                                     │")
        print("└────────────────────────────────────────────────────────┘")

        print("\n┌────────────────────────────────────────────────────────┐")
        print("│                    FUNCTIONGEMMA                        │")
        print("├────────────────────────────────────────────────────────┤")
        print(f"│  U-shape: {'YES' if func['probe']['u_shape'] else 'NO':4s}                                        │")
        print(f"│  Early acc: {func['probe']['early_acc']:.1%}                                     │")
        print(f"│  Mid acc:   {func['probe']['mid_acc']:.1%}                                     │")
        print(f"│  Late acc:  {func['probe']['late_acc']:.1%}                                     │")
        if 11 in func['pca']:
            print(f"│  L11 PC1:   {func['pca'][11]:.1%}                                     │")
        print("└────────────────────────────────────────────────────────┘")

        # Hypothesis verdict
        print("\n" + "=" * 60)
        print("HYPOTHESIS VERDICT")
        print("=" * 60)

        if func['probe']['u_shape'] and not base['probe']['u_shape']:
            print("\n✓ U-SHAPE: Only FunctionGemma has it → from tool-training!")
        elif func['probe']['u_shape'] and base['probe']['u_shape']:
            print("\n✗ Both have U-shape → architectural, not training-specific")
        elif not func['probe']['u_shape'] and not base['probe']['u_shape']:
            print("\n? Neither has U-shape on this task")
        else:
            print("\n? Unexpected: base has U-shape but function doesn't")

        # Check PC1 difference
        if 11 in base['pca'] and 11 in func['pca']:
            base_pc1 = base['pca'][11]
            func_pc1 = func['pca'][11]

            if func_pc1 > base_pc1 + 0.15:
                print(f"\n✓ 1D DECISION: FunctionGemma L11 is more 1D ({func_pc1:.1%} vs {base_pc1:.1%})")
                print("  → Tool-training compressed decision to 1D!")
            elif abs(func_pc1 - base_pc1) < 0.1:
                print(f"\n≈ Similar PC1: both ~{(func_pc1+base_pc1)/2:.1%}")
            else:
                print(f"\n? Base has higher PC1: {base_pc1:.1%} vs {func_pc1:.1%}")

        # Save results
        with open('gemma_tool_comparison.json', 'w') as f:
            json_results = {
                name: {
                    'accuracies': data['probe']['accuracies'],
                    'separations': data['probe']['separations'],
                    'u_shape': data['probe']['u_shape'],
                    'pca': {str(k): v for k, v in data['pca'].items()},
                }
                for name, data in self.results.items()
            }
            json.dump(json_results, f, indent=2)

        print("\nResults saved to gemma_tool_comparison.json")


def main():
    comparison = GemmaToolProbe()
    comparison.run_comparison()


if __name__ == "__main__":
    main()
