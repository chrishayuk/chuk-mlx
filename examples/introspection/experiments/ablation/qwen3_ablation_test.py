#!/usr/bin/env python3
"""
qwen3_ablation_test.py

Quick ablation test on Qwen3 to find critical layers.

The probe showed 100% accuracy everywhere, which is suspicious.
Ablation will reveal which layers actually matter.

Run: uv run python examples/introspection/qwen3_ablation_test.py
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


class Qwen3AblationTest:
    """Quick ablation test for Qwen3."""

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
        self.hidden_size = self.config.hidden_size

        print(f"Loaded: {self.num_layers} layers, hidden={self.hidden_size}")

    def get_final_hidden_with_ablation(
        self,
        prompt: str,
        ablate_layer: int = -1,
        ablate_neurons: list[int] = None,
    ) -> np.ndarray:
        """Get final hidden state, optionally ablating a layer's MLP."""
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

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == ablate_layer:
                # Ablate this layer's MLP
                h = self._forward_with_ablated_mlp(h, mask, layer, ablate_neurons)
            else:
                output = layer(h, mask=mask, cache=None)
                if isinstance(output, tuple):
                    h = output[0]
                elif hasattr(output, 'hidden_states'):
                    h = output.hidden_states
                else:
                    h = output

        # Final norm
        h = self.model.model.norm(h)
        return np.array(h[0, -1, :].tolist())

    def _forward_with_ablated_mlp(
        self,
        h: mx.array,
        mask: mx.array,
        layer,
        ablate_neurons: list[int] = None,
    ) -> mx.array:
        """Forward through layer with ablated MLP."""
        # Pre-attention norm
        if hasattr(layer, 'input_layernorm'):
            normed = layer.input_layernorm(h)
        else:
            normed = h

        # Attention
        attn_out = layer.self_attn(normed, mask=mask, cache=None)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = h + attn_out

        # Post-attention norm
        if hasattr(layer, 'post_attention_layernorm'):
            mlp_input = layer.post_attention_layernorm(h)
        else:
            mlp_input = h

        # MLP with ablation
        mlp = layer.mlp
        if hasattr(mlp, 'gate_proj'):
            gate = mlp.gate_proj(mlp_input)
            up = mlp.up_proj(mlp_input)
            mlp_hidden = nn.silu(gate) * up

            if ablate_neurons is None:
                # Zero entire MLP output
                mlp_out = mx.zeros_like(mlp.down_proj(mlp_hidden))
            else:
                # Zero specific neurons
                hidden_np = np.array(mlp_hidden.tolist())
                for n in ablate_neurons:
                    if n < hidden_np.shape[-1]:
                        hidden_np[0, -1, n] = 0
                mlp_hidden = mx.array(hidden_np).astype(mlp_hidden.dtype)
                mlp_out = mlp.down_proj(mlp_hidden)
        else:
            mlp_out = mx.zeros_like(mlp(mlp_input))

        return h + mlp_out

    def run_ablation_test(self):
        """Test which layers are critical by ablating MLPs."""
        print("\n" + "=" * 60)
        print("MLP ABLATION TEST")
        print("=" * 60)

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        # Use train/test split
        train_action = [
            "Calculate 25 * 4",
            "Write a haiku",
            "Sort: 5, 2, 8",
            "Translate to French",
            "Generate a password",
            "Count words in: hello world",
        ]

        train_question = [
            "What is gravity?",
            "Who is Einstein?",
            "When did WWII end?",
            "What causes rain?",
            "Where is Paris?",
            "Why is sky blue?",
        ]

        test_action = [
            "Multiply 7 by 8",
            "Write a limerick",
            "Convert 100F to C",
        ]

        test_question = [
            "What is photosynthesis?",
            "Who wrote Hamlet?",
            "What is the speed of light?",
        ]

        print("\n1. Training probe on normal activations...")

        # Collect training data
        X_train = []
        y_train = []

        for prompt in train_action:
            formatted = format_prompt(prompt)
            act = self.get_final_hidden_with_ablation(formatted, ablate_layer=-1)
            X_train.append(act)
            y_train.append(1)

        for prompt in train_question:
            formatted = format_prompt(prompt)
            act = self.get_final_hidden_with_ablation(formatted, ablate_layer=-1)
            X_train.append(act)
            y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)

        # Baseline test accuracy
        X_test = []
        y_test = []

        for prompt in test_action:
            formatted = format_prompt(prompt)
            act = self.get_final_hidden_with_ablation(formatted, ablate_layer=-1)
            X_test.append(act)
            y_test.append(1)

        for prompt in test_question:
            formatted = format_prompt(prompt)
            act = self.get_final_hidden_with_ablation(formatted, ablate_layer=-1)
            X_test.append(act)
            y_test.append(0)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        baseline_acc = probe.score(X_test, y_test)
        print(f"   Baseline test accuracy: {baseline_acc:.1%}")

        print("\n2. Testing MLP ablation at each layer...")
        print(f"\n   {'Layer':<8} {'Accuracy':>10} {'Drop':>10} {'Status':<15}")
        print("   " + "-" * 45)

        critical_layers = []

        for layer_idx in range(self.num_layers):
            # Collect test activations with ablation
            X_ablated = []

            for prompt in test_action + test_question:
                formatted = format_prompt(prompt)
                act = self.get_final_hidden_with_ablation(formatted, ablate_layer=layer_idx)
                X_ablated.append(act)

            X_ablated = np.array(X_ablated)
            ablated_acc = probe.score(X_ablated, y_test)
            drop = baseline_acc - ablated_acc

            status = ""
            if drop > 0.3:
                status = "CRITICAL"
                critical_layers.append(layer_idx)
            elif drop > 0.1:
                status = "Important"
            elif drop < -0.1:
                status = "(improved?)"

            print(f"   L{layer_idx:<6} {ablated_acc:>9.1%} {drop:>+9.1%} {status:<15}")

        print(f"\n   Critical layers (>30% drop): {critical_layers}")

        return critical_layers

    def run_neuron_ablation(self, target_layers: list[int] = None):
        """Find critical neurons in important layers."""
        print("\n" + "=" * 60)
        print("NEURON ABLATION TEST")
        print("=" * 60)

        if target_layers is None:
            # Default to late layers where most change happens
            target_layers = [20, 22, 24, 26]

        def format_prompt(query):
            return f"""<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        # Simple test set
        test_prompts = [
            (format_prompt("Calculate 5 + 3"), 1),
            (format_prompt("What is gravity?"), 0),
            (format_prompt("Write a poem"), 1),
            (format_prompt("Who is Einstein?"), 0),
        ]

        print(f"\nTesting top-10 neuron ablation in layers: {target_layers}")

        for layer_idx in target_layers:
            print(f"\n   Layer {layer_idx}:")

            # First, find discriminative neurons
            action_acts = []
            question_acts = []

            for prompt, label in test_prompts:
                act = self.get_final_hidden_with_ablation(prompt, ablate_layer=-1)
                if label == 1:
                    action_acts.append(act)
                else:
                    question_acts.append(act)

            # This is a simplification - we'd need MLP activations, not final hidden
            # For now, just test ablating random top neurons
            print(f"     Testing ablation of top neurons...")

            # Test ablating neurons 0-9, 100-109, etc.
            neuron_groups = [
                list(range(0, 10)),
                list(range(100, 110)),
                list(range(500, 510)),
                list(range(1000, 1010)),
            ]

            for neurons in neuron_groups:
                X_ablated = []
                y_test = []

                for prompt, label in test_prompts:
                    act = self.get_final_hidden_with_ablation(
                        prompt,
                        ablate_layer=layer_idx,
                        ablate_neurons=neurons,
                    )
                    X_ablated.append(act)
                    y_test.append(label)

                # Simple accuracy check
                X_ablated = np.array(X_ablated)
                y_test = np.array(y_test)

                # Just measure if ablation changes the mean difference
                action_mean = X_ablated[np.array(y_test) == 1].mean(axis=0)
                question_mean = X_ablated[np.array(y_test) == 0].mean(axis=0)
                sep = np.linalg.norm(action_mean - question_mean)

                print(f"     Neurons {neurons[0]}-{neurons[-1]}: separation = {sep:.1f}")

    def run_all(self):
        """Run all ablation tests."""
        self.load_model()

        critical_layers = self.run_ablation_test()

        # Only run neuron ablation if we found critical layers
        if critical_layers:
            self.run_neuron_ablation(critical_layers[:4])
        else:
            print("\nNo critical layers found. Testing late layers...")
            self.run_neuron_ablation([20, 22, 24, 26])

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("""
Unlike FunctionGemma which has:
  - L5: 50% drop (most critical)
  - L11: 100% flip rate (control point)
  - L12: 70% drop from single neuron (kill switch)

Qwen3's critical layers (if any) indicate where
the model actually computes task distinctions.

If no layers show >30% drop:
  - Either probe task is too easy
  - Or Qwen3's representations are highly redundant
  - Or task info is distributed (no single control point)
""")


def main():
    tester = Qwen3AblationTest()
    tester.run_all()


if __name__ == "__main__":
    main()
