#!/usr/bin/env python3
"""
gemma_base_vs_function.py

Compare base Gemma 3 270M vs FunctionGemma to test:
1. Does the U-shape come from tool-training?
2. Does the kill switch exist in base model?
3. Is the 1D decision space (86% PC1) from specialization?

Run: uv run python examples/introspection/gemma_base_vs_function.py
"""

import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


class GemmaComparison:
    """Compare base Gemma vs FunctionGemma circuits."""

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

        # Forward through embedding
        h = model.model.embed_tokens(input_ids)

        # Scale embeddings (Gemma-specific)
        hidden_size = model_data['config'].hidden_size
        h = h * mx.array(hidden_size ** 0.5, dtype=h.dtype)

        # Create attention mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Forward through layers
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

    def get_final_hidden_with_ablation(
        self,
        model_data: dict,
        prompt: str,
        ablate_layer: int = -1,
    ) -> np.ndarray:
        """Get final hidden state with optional MLP ablation."""
        model = model_data['model']
        tokenizer = model_data['tokenizer']

        tokens = tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])

        # Forward through embedding
        h = model.model.embed_tokens(input_ids)

        # Scale embeddings (Gemma-specific)
        hidden_size = model_data['config'].hidden_size
        h = h * mx.array(hidden_size ** 0.5, dtype=h.dtype)

        # Create attention mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Forward through layers
        for layer_idx, layer in enumerate(model_data['layers']):
            if layer_idx == ablate_layer:
                # Ablate MLP
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                attn_out = layer.self_attn(normed, mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                # Skip MLP (zero output)
                if hasattr(layer, 'post_attention_layernorm'):
                    mlp_input = layer.post_attention_layernorm(h)
                else:
                    mlp_input = h

                # Just add zeros instead of MLP output
                # h = h + 0  (no change needed)
            else:
                output = layer(h, mask=mask, cache=None)
                if hasattr(output, 'hidden_states'):
                    h = output.hidden_states
                elif isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output

        # Final norm
        h = model.model.norm(h)
        return np.array(h[0, -1, :].tolist())

    def run_probe_analysis(self, model_data: dict, model_name: str):
        """Run probe analysis to find accuracy curve."""
        print(f"\n{'='*50}")
        print(f"PROBE ANALYSIS: {model_name}")
        print('='*50)

        # Action vs Question prompts (works for both base and function models)
        action_prompts = [
            "Calculate 25 * 4",
            "Write a haiku about spring",
            "Sort these numbers: 5, 2, 8, 1",
            "Translate 'hello' to French",
            "Generate a random password",
            "Summarize: The quick brown fox",
        ]

        question_prompts = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?",
            "When did World War 2 end?",
            "What is the speed of light?",
            "Who invented the telephone?",
        ]

        num_layers = model_data['num_layers']
        accuracies = []
        separations = []

        print("\nLayer-by-layer probe accuracy:")

        for layer_idx in range(num_layers):
            X = []
            y = []

            for prompt in action_prompts:
                act = self.get_layer_activations(model_data, prompt, layer_idx)
                X.append(act)
                y.append(1)

            for prompt in question_prompts:
                act = self.get_layer_activations(model_data, prompt, layer_idx)
                X.append(act)
                y.append(0)

            X = np.array(X)
            y = np.array(y)

            # Train probe
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X, y)
            acc = probe.score(X, y)
            accuracies.append(acc)

            # Compute separation
            action_mean = X[y == 1].mean(axis=0)
            question_mean = X[y == 0].mean(axis=0)
            sep = np.linalg.norm(action_mean - question_mean)
            separations.append(sep)

            # Print with visual bar
            bar = "█" * int(acc * 20)
            print(f"  L{layer_idx:2d}: {acc:.1%} {bar}")

        # Check for U-shape
        early_acc = np.mean(accuracies[:3])
        mid_acc = np.mean(accuracies[5:10])
        late_acc = np.mean(accuracies[-5:])

        print(f"\nU-shape check:")
        print(f"  Early (L0-2):  {early_acc:.1%}")
        print(f"  Middle (L5-9): {mid_acc:.1%}")
        print(f"  Late (L13-17): {late_acc:.1%}")

        if mid_acc < early_acc - 0.1 and late_acc > mid_acc + 0.1:
            print("  → U-SHAPE DETECTED")
        else:
            print("  → NO U-shape (monotonic or flat)")

        return {
            'accuracies': accuracies,
            'separations': separations,
            'early_acc': early_acc,
            'mid_acc': mid_acc,
            'late_acc': late_acc,
        }

    def run_ablation_analysis(self, model_data: dict, model_name: str):
        """Run ablation to find critical layers."""
        print(f"\n{'='*50}")
        print(f"ABLATION ANALYSIS: {model_name}")
        print('='*50)

        # Test prompts
        action_prompts = [
            "Calculate 5 + 3",
            "Write a poem",
            "Sort: 3, 1, 2",
        ]

        question_prompts = [
            "What is gravity?",
            "Who is Einstein?",
            "What causes rain?",
        ]

        # Train probe on normal activations
        X_train = []
        y_train = []

        for prompt in action_prompts:
            act = self.get_final_hidden_with_ablation(model_data, prompt, ablate_layer=-1)
            X_train.append(act)
            y_train.append(1)

        for prompt in question_prompts:
            act = self.get_final_hidden_with_ablation(model_data, prompt, ablate_layer=-1)
            X_train.append(act)
            y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        baseline_acc = probe.score(X_train, y_train)

        print(f"\nBaseline accuracy: {baseline_acc:.1%}")
        print("\nMLP ablation by layer:")

        drops = []
        critical_layers = []

        for layer_idx in range(model_data['num_layers']):
            X_ablated = []

            for prompt in action_prompts + question_prompts:
                act = self.get_final_hidden_with_ablation(model_data, prompt, ablate_layer=layer_idx)
                X_ablated.append(act)

            X_ablated = np.array(X_ablated)
            ablated_acc = probe.score(X_ablated, y_train)
            drop = baseline_acc - ablated_acc
            drops.append(drop)

            status = ""
            if drop > 0.3:
                status = "CRITICAL"
                critical_layers.append(layer_idx)
            elif drop > 0.1:
                status = "important"

            bar = "█" * int(max(0, drop) * 20)
            print(f"  L{layer_idx:2d}: drop={drop:+.1%} {bar} {status}")

        print(f"\nCritical layers (>30% drop): {critical_layers}")

        return {
            'drops': drops,
            'critical_layers': critical_layers,
            'baseline_acc': baseline_acc,
        }

    def run_pca_analysis(self, model_data: dict, model_name: str, target_layer: int = 11):
        """Check if decision space is 1D at target layer."""
        print(f"\n{'='*50}")
        print(f"PCA ANALYSIS: {model_name} (L{target_layer})")
        print('='*50)

        action_prompts = [
            "Calculate 25 * 4",
            "Write a haiku",
            "Sort: 5, 2, 8",
            "Translate hello",
            "Generate password",
        ]

        question_prompts = [
            "What is gravity?",
            "Who is Shakespeare?",
            "What is photosynthesis?",
            "When did WWII end?",
            "What causes thunder?",
        ]

        X = []
        y = []

        for prompt in action_prompts:
            act = self.get_layer_activations(model_data, prompt, target_layer)
            X.append(act)
            y.append(1)

        for prompt in question_prompts:
            act = self.get_layer_activations(model_data, prompt, target_layer)
            X.append(act)
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        pca = PCA(n_components=5)
        pca.fit(X)

        print("\nVariance explained by top PCs:")
        for i, var in enumerate(pca.explained_variance_ratio_[:5]):
            bar = "█" * int(var * 50)
            print(f"  PC{i+1}: {var:.1%} {bar}")

        pc1_var = pca.explained_variance_ratio_[0]

        if pc1_var > 0.7:
            print(f"\n→ NEARLY 1D: {pc1_var:.1%} on PC1 (like FunctionGemma)")
        elif pc1_var > 0.5:
            print(f"\n→ SOMEWHAT 1D: {pc1_var:.1%} on PC1")
        else:
            print(f"\n→ MULTI-DIMENSIONAL: only {pc1_var:.1%} on PC1")

        return {
            'pc_variances': pca.explained_variance_ratio_.tolist(),
            'pc1_variance': pc1_var,
        }

    def run_comparison(self):
        """Run full comparison."""
        print("\n" + "=" * 60)
        print("BASE GEMMA vs FUNCTIONGEMMA COMPARISON")
        print("=" * 60)
        print("\nHypothesis: U-shape and kill switch come from tool-training")

        for name, model_id in self.models.items():
            model_data = self.load_model(model_id)

            probe_results = self.run_probe_analysis(model_data, name)
            ablation_results = self.run_ablation_analysis(model_data, name)
            pca_results = self.run_pca_analysis(model_data, name, target_layer=11)

            self.results[name] = {
                'probe': probe_results,
                'ablation': ablation_results,
                'pca': pca_results,
            }

        # Final comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)

        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│                    BASE GEMMA 3 270M                     │")
        print("├─────────────────────────────────────────────────────────┤")

        base = self.results['base']
        print(f"│  Early accuracy:  {base['probe']['early_acc']:.1%}                              │")
        print(f"│  Middle accuracy: {base['probe']['mid_acc']:.1%}                              │")
        print(f"│  Late accuracy:   {base['probe']['late_acc']:.1%}                              │")
        print(f"│  PC1 variance:    {base['pca']['pc1_variance']:.1%}                              │")
        print(f"│  Critical layers: {base['ablation']['critical_layers']}                         │")
        print("└─────────────────────────────────────────────────────────┘")

        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│                    FUNCTIONGEMMA 270M                    │")
        print("├─────────────────────────────────────────────────────────┤")

        func = self.results['function']
        print(f"│  Early accuracy:  {func['probe']['early_acc']:.1%}                              │")
        print(f"│  Middle accuracy: {func['probe']['mid_acc']:.1%}                              │")
        print(f"│  Late accuracy:   {func['probe']['late_acc']:.1%}                              │")
        print(f"│  PC1 variance:    {func['pca']['pc1_variance']:.1%}                              │")
        print(f"│  Critical layers: {func['ablation']['critical_layers']}                         │")
        print("└─────────────────────────────────────────────────────────┘")

        # Hypothesis test
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST")
        print("=" * 60)

        base_has_ushape = base['probe']['mid_acc'] < base['probe']['early_acc'] - 0.1
        func_has_ushape = func['probe']['mid_acc'] < func['probe']['early_acc'] - 0.1

        print(f"\nU-shape in base Gemma:     {'YES' if base_has_ushape else 'NO'}")
        print(f"U-shape in FunctionGemma:  {'YES' if func_has_ushape else 'NO'}")

        if func_has_ushape and not base_has_ushape:
            print("\n→ HYPOTHESIS CONFIRMED: U-shape comes from tool-training!")
        elif func_has_ushape and base_has_ushape:
            print("\n→ Both have U-shape: architectural feature, not training")
        else:
            print("\n→ Need more investigation")

        base_1d = base['pca']['pc1_variance'] > 0.7
        func_1d = func['pca']['pc1_variance'] > 0.7

        print(f"\n1D decision in base Gemma:     {'YES' if base_1d else 'NO'} ({base['pca']['pc1_variance']:.1%})")
        print(f"1D decision in FunctionGemma:  {'YES' if func_1d else 'NO'} ({func['pca']['pc1_variance']:.1%})")

        if func_1d and not base_1d:
            print("\n→ HYPOTHESIS CONFIRMED: 1D structure from tool-training!")

        # Save results
        with open('gemma_comparison_results.json', 'w') as f:
            # Convert numpy to lists for JSON
            json_results = {}
            for name, data in self.results.items():
                json_results[name] = {
                    'probe': {
                        'accuracies': data['probe']['accuracies'],
                        'separations': data['probe']['separations'],
                        'early_acc': data['probe']['early_acc'],
                        'mid_acc': data['probe']['mid_acc'],
                        'late_acc': data['probe']['late_acc'],
                    },
                    'ablation': {
                        'drops': data['ablation']['drops'],
                        'critical_layers': data['ablation']['critical_layers'],
                    },
                    'pca': data['pca'],
                }
            json.dump(json_results, f, indent=2)

        print("\nResults saved to gemma_comparison_results.json")


def main():
    comparison = GemmaComparison()
    comparison.run_comparison()


if __name__ == "__main__":
    main()
