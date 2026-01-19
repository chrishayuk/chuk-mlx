#!/usr/bin/env python3
"""
neuron_causal_v2.py

Improved causal analysis with:
1. Probe-based prediction (not heuristic)
2. Full MLP ablation comparison
3. More aggressive patching (top 20 neurons)
4. L10 vs L11 vs L12 comparison

Run: uv run python examples/introspection/neuron_causal_v2.py
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class NeuronCausalV2:
    """Improved causal analysis."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id
        self.probe = None  # Will train a probe for classification

        # Top neurons from previous experiment
        self.top_neurons = {
            10: [
                1709,
                1436,
                1390,
                1586,
                1106,
                1023,
                732,
                977,
                1872,
                1048,
                1148,
                1877,
                257,
                1270,
                893,
                1341,
                1134,
                997,
                72,
                1817,
            ],
            11: [
                1237,
                551,
                1069,
                821,
                1710,
                1873,
                1653,
                1292,
                1712,
                712,
                543,
                1077,
                1161,
                217,
                324,
                1323,
                1110,
                612,
                417,
                537,
            ],
            12: [
                230,
                1361,
                901,
                1545,
                1173,
                1412,
                41,
                1767,
                1693,
                266,
                1111,
                1643,
                1533,
                950,
                1297,
                1039,
                467,
                755,
                166,
                891,
            ],
        }

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
        """Prepare train and test prompts."""
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

        train = [(p, True) for p in tool_prompts[:10]] + [(p, False) for p in no_tool_prompts[:10]]
        test = [(p, True) for p in tool_prompts[10:]] + [(p, False) for p in no_tool_prompts[10:]]

        return train, test

    def get_final_hidden(self, tokens: list[int]) -> mx.array:
        """Get final layer hidden state (before lm_head)."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer in self.layers:
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

        h = self.final_norm(h)
        return h.astype(mx.float32)

    def get_final_hidden_with_ablation(
        self,
        tokens: list[int],
        ablate_layer: int,
        ablate_neurons: list[int] | None = None,
        ablate_full_mlp: bool = False,
    ) -> mx.array:
        """Get final hidden state with ablation."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            if layer_idx == ablate_layer:
                # Attention
                if hasattr(layer, "input_layernorm"):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                attn_out = layer.self_attn(normed, mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                if ablate_full_mlp:
                    # Skip MLP entirely
                    pass
                else:
                    # MLP with neuron ablation
                    if hasattr(layer, "post_attention_layernorm"):
                        mlp_input = layer.post_attention_layernorm(h)
                    else:
                        mlp_input = h

                    mlp = layer.mlp
                    if hasattr(mlp, "gate_proj"):
                        gate = mlp.gate_proj(mlp_input)
                        up = mlp.up_proj(mlp_input)
                        mlp_hidden = nn.gelu(gate) * up

                        if ablate_neurons:
                            # Create ablation mask
                            mask_arr = mx.ones(mlp_hidden.shape[-1])
                            for neuron_idx in ablate_neurons:
                                if neuron_idx < mlp_hidden.shape[-1]:
                                    mask_arr = mx.concatenate(
                                        [
                                            mask_arr[:neuron_idx],
                                            mx.array([0.0]),
                                            mask_arr[neuron_idx + 1 :],
                                        ]
                                    )
                            mlp_hidden = mlp_hidden * mask_arr

                        mlp_out = mlp.down_proj(mlp_hidden)
                    else:
                        mlp_out = mlp(mlp_input)

                    h = h + mlp_out
            else:
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

        h = self.final_norm(h)
        return h.astype(mx.float32)

    def train_probe(self, train_data: list[tuple[str, bool]]):
        """Train a probe on final hidden states."""
        print("\n  Training probe on final hidden states...")

        X = []
        y = []
        for prompt, label in train_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            h = self.get_final_hidden(tokens)
            X.append(np.array(h[0, -1, :].tolist()))
            y.append(int(label))

        X = np.array(X)
        y = np.array(y)

        self.probe = LogisticRegression(max_iter=1000)
        self.probe.fit(X, y)

        train_acc = self.probe.score(X, y)
        print(f"  Probe train accuracy: {train_acc:.1%}")

    def predict_with_probe(self, h: mx.array) -> tuple[bool, float]:
        """Predict using trained probe."""
        x = np.array(h[0, -1, :].tolist()).reshape(1, -1)
        pred = bool(self.probe.predict(x)[0])
        prob = float(self.probe.predict_proba(x)[0, 1])
        return pred, prob

    def run_full_mlp_ablation(self, test_data: list[tuple[str, bool]]):
        """Compare full MLP ablation across layers."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: FULL MLP ABLATION")
        print("=" * 60)
        print("\nQuestion: Which layer's MLP is most important?")

        # Baseline accuracy
        baseline_correct = 0
        for prompt, label in test_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            h = self.get_final_hidden(tokens)
            pred, _ = self.predict_with_probe(h)
            if pred == label:
                baseline_correct += 1
        baseline_acc = baseline_correct / len(test_data)
        print(f"\n  Baseline accuracy: {baseline_acc:.1%}")

        print("\n  Ablating full MLP at each layer:")
        print(f"  {'Layer':<8} {'Accuracy':>10} {'Drop':>10}")
        print("  " + "-" * 30)

        for layer in range(self.num_layers):
            correct = 0
            for prompt, label in test_data:
                tokens = self.tokenizer.encode(prompt)
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.flatten().tolist()
                h = self.get_final_hidden_with_ablation(tokens, layer, ablate_full_mlp=True)
                pred, _ = self.predict_with_probe(h)
                if pred == label:
                    correct += 1
            acc = correct / len(test_data)
            drop = baseline_acc - acc
            marker = " ← CRITICAL" if drop > 0.2 else ""
            print(f"  L{layer:<7} {acc:>9.1%} {drop:>+9.1%}{marker}")

    def run_neuron_ablation(self, test_data: list[tuple[str, bool]]):
        """Test neuron ablation with probe-based accuracy."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: TOP NEURON ABLATION")
        print("=" * 60)
        print("\nQuestion: Are top discriminative neurons causally necessary?")

        # Baseline
        baseline_correct = 0
        for prompt, label in test_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            h = self.get_final_hidden(tokens)
            pred, _ = self.predict_with_probe(h)
            if pred == label:
                baseline_correct += 1
        baseline_acc = baseline_correct / len(test_data)

        for layer in [10, 11, 12]:
            print(f"\n  --- Layer {layer} ---")
            neurons = self.top_neurons[layer]

            for k in [1, 5, 10, 20]:
                neurons_to_ablate = neurons[:k]

                correct = 0
                for prompt, label in test_data:
                    tokens = self.tokenizer.encode(prompt)
                    if isinstance(tokens, np.ndarray):
                        tokens = tokens.flatten().tolist()
                    h = self.get_final_hidden_with_ablation(
                        tokens, layer, ablate_neurons=neurons_to_ablate
                    )
                    pred, _ = self.predict_with_probe(h)
                    if pred == label:
                        correct += 1

                acc = correct / len(test_data)
                drop = baseline_acc - acc
                print(f"    Ablate top {k:2d}: acc={acc:.1%}, drop={drop:+.1%}")

    def run_patching_experiment(self, test_data: list[tuple[str, bool]]):
        """Patch activations from tool prompts into no-tool prompts."""
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: ACTIVATION PATCHING")
        print("=" * 60)
        print("\nQuestion: Can we flip predictions by patching neurons?")

        tool_prompts = [(p, l) for p, l in test_data if l]
        no_tool_prompts = [(p, l) for p, l in test_data if not l]

        for layer in [10, 11, 12]:
            print(f"\n  --- Layer {layer} ---")
            neurons = self.top_neurons[layer]

            for k in [5, 10, 20]:
                neurons_to_patch = neurons[:k]
                flips = 0
                total = 0

                for source_prompt, _ in tool_prompts:
                    for target_prompt, _ in no_tool_prompts:
                        source_tokens = self.tokenizer.encode(source_prompt)
                        target_tokens = self.tokenizer.encode(target_prompt)

                        if isinstance(source_tokens, np.ndarray):
                            source_tokens = source_tokens.flatten().tolist()
                        if isinstance(target_tokens, np.ndarray):
                            target_tokens = target_tokens.flatten().tolist()

                        # Original prediction on target
                        h_original = self.get_final_hidden(target_tokens)
                        original_pred, original_prob = self.predict_with_probe(h_original)

                        # Patched prediction
                        h_patched = self.get_final_hidden_with_patch(
                            target_tokens, source_tokens, layer, neurons_to_patch
                        )
                        patched_pred, patched_prob = self.predict_with_probe(h_patched)

                        if original_pred != patched_pred:
                            flips += 1
                        total += 1

                flip_rate = flips / total if total > 0 else 0
                print(f"    Patch top {k:2d}: {flips}/{total} flipped ({flip_rate:.1%})")

    def get_final_hidden_with_patch(
        self,
        target_tokens: list[int],
        source_tokens: list[int],
        patch_layer: int,
        patch_neurons: list[int],
    ) -> mx.array:
        """Get final hidden with patched neurons from source."""
        # First get source MLP activations
        source_input = mx.array([source_tokens])
        source_emb = self.embed_layer(source_input) * self.embed_scale
        source_h = source_emb
        source_seq_len = source_h.shape[1]
        source_mask = nn.MultiHeadAttention.create_additive_causal_mask(source_seq_len)
        source_mask = source_mask.astype(source_h.dtype)

        source_mlp_activations = None

        for layer_idx in range(patch_layer + 1):
            layer = self.layers[layer_idx]

            if layer_idx == patch_layer:
                if hasattr(layer, "input_layernorm"):
                    normed = layer.input_layernorm(source_h)
                else:
                    normed = source_h

                attn_out = layer.self_attn(normed, mask=source_mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                source_h = source_h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    mlp_input = layer.post_attention_layernorm(source_h)
                else:
                    mlp_input = source_h

                mlp = layer.mlp
                if hasattr(mlp, "gate_proj"):
                    gate = mlp.gate_proj(mlp_input)
                    up = mlp.up_proj(mlp_input)
                    source_mlp_activations = nn.gelu(gate) * up
            else:
                try:
                    layer_out = layer(source_h, mask=source_mask)
                except TypeError:
                    layer_out = layer(source_h)

                if hasattr(layer_out, "hidden_states"):
                    source_h = layer_out.hidden_states
                elif isinstance(layer_out, tuple):
                    source_h = layer_out[0]
                else:
                    source_h = layer_out

        # Now run target with patching
        target_input = mx.array([target_tokens])
        target_emb = self.embed_layer(target_input) * self.embed_scale
        h = target_emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            if layer_idx == patch_layer:
                if hasattr(layer, "input_layernorm"):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                attn_out = layer.self_attn(normed, mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                if hasattr(layer, "post_attention_layernorm"):
                    mlp_input = layer.post_attention_layernorm(h)
                else:
                    mlp_input = h

                mlp = layer.mlp
                if hasattr(mlp, "gate_proj"):
                    gate = mlp.gate_proj(mlp_input)
                    up = mlp.up_proj(mlp_input)
                    mlp_hidden = nn.gelu(gate) * up

                    # PATCH: blend source activations into target
                    # For last token, replace specified neurons with source values
                    target_last = mlp_hidden[0, -1, :].astype(mx.float32)
                    source_last = source_mlp_activations[0, -1, :].astype(mx.float32)

                    # Create blend mask
                    patch_mask = mx.zeros(mlp_hidden.shape[-1])
                    for neuron_idx in patch_neurons:
                        if neuron_idx < mlp_hidden.shape[-1]:
                            patch_mask = mx.concatenate(
                                [
                                    patch_mask[:neuron_idx],
                                    mx.array([1.0]),
                                    patch_mask[neuron_idx + 1 :],
                                ]
                            )

                    patched_last = source_last * patch_mask + target_last * (1 - patch_mask)

                    # Reconstruct
                    if seq_len > 1:
                        mlp_hidden = mx.concatenate(
                            [
                                mlp_hidden[:, :-1, :],
                                patched_last.reshape(1, 1, -1).astype(mlp_hidden.dtype),
                            ],
                            axis=1,
                        )
                    else:
                        mlp_hidden = patched_last.reshape(1, 1, -1).astype(mlp_hidden.dtype)

                    mlp_out = mlp.down_proj(mlp_hidden)
                else:
                    mlp_out = mlp(mlp_input)

                h = h + mlp_out
            else:
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

        h = self.final_norm(h)
        return h.astype(mx.float32)

    def run_experiment(self):
        """Run all experiments."""
        print("=" * 60)
        print("NEURON CAUSAL ANALYSIS V2")
        print("=" * 60)
        print("\nUsing probe-based classification for accurate metrics.")

        self.load_model()
        train_data, test_data = self.prepare_dataset()

        print(f"\nDataset: {len(train_data)} train, {len(test_data)} test")

        # Train probe
        self.train_probe(train_data)

        # Test probe accuracy
        test_correct = 0
        for prompt, label in test_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            h = self.get_final_hidden(tokens)
            pred, _ = self.predict_with_probe(h)
            if pred == label:
                test_correct += 1
        print(f"  Probe test accuracy: {test_correct / len(test_data):.1%}")

        # Run experiments
        self.run_full_mlp_ablation(test_data)
        self.run_neuron_ablation(test_data)
        self.run_patching_experiment(test_data)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("""
Key questions answered:
1. Which layer's MLP is most critical? (Full ablation)
2. Are top neurons causally necessary? (Neuron ablation)
3. Can patching flip decisions? (Activation patching)

If L10 ablation hurts more than L11/L12:
  → L10 is the DECISION layer
  → L11-12 are COMMITMENT/FORMATTING layers

If patching flips predictions:
  → Those neurons are SUFFICIENT for the decision
""")


def main():
    experiment = NeuronCausalV2()
    experiment.run_experiment()


if __name__ == "__main__":
    main()
