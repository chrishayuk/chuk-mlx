#!/usr/bin/env python3
"""
neuron_causal_analysis.py

Phase 4.2-4.3: Causal Neuron Analysis

Three experiments:
1. Neuron Ablation: Zero out top discriminative neurons, measure accuracy drop
2. Activation Patching: Inject tool-neuron activations into no-tool prompts
3. L11 Investigation: Why doesn't L11 reach 90% accuracy?

Run: uv run python examples/introspection/neuron_causal_analysis.py
"""

import warnings
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class AblationResult:
    """Result of ablating neurons."""

    layer: int
    neurons_ablated: list[int]
    original_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float


@dataclass
class PatchingResult:
    """Result of patching activations."""

    source_prompt: str
    target_prompt: str
    source_label: bool
    target_label: bool
    original_prediction: bool
    patched_prediction: bool
    flipped: bool


class NeuronCausalAnalysis:
    """Causal analysis of tool-calling neurons."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id

        # Top neurons from previous experiment
        self.top_neurons = {
            10: [1709, 1436, 1390, 1586, 1106, 1023, 732, 977, 1872, 1048],
            11: [1237, 551, 1069, 821, 1710, 1873, 1653, 1292, 1712, 712],
            12: [230, 1361, 901, 1545, 1173, 1412, 41, 1767, 1693, 266],
            13: [1106, 707, 1089, 1301, 1216, 496, 57, 1807, 74, 1801],
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

    def prepare_dataset(self) -> list[tuple[str, bool]]:
        """Prepare test prompts."""
        prompts = [
            # Tool
            ("What is the weather in Tokyo?", True),
            ("Send an email to John", True),
            ("Create a calendar event", True),
            ("Set a timer for 10 minutes", True),
            ("Book a flight to Paris", True),
            ("Turn on the lights", True),
            ("Play some music", True),
            ("Call mom", True),
            ("Order a pizza", True),
            ("Check my schedule", True),
            # No tool
            ("What is the capital of France?", False),
            ("Explain quantum physics", False),
            ("What is 2 + 2?", False),
            ("Tell me about Einstein", False),
            ("What is photosynthesis?", False),
            ("Why is the sky blue?", False),
            ("Who wrote Romeo and Juliet?", False),
            ("What is machine learning?", False),
            ("How does gravity work?", False),
            ("Explain democracy", False),
        ]
        return prompts

    def forward_with_neuron_ablation(
        self,
        tokens: list[int],
        ablate_layer: int,
        ablate_neurons: list[int],
    ) -> mx.array:
        """
        Forward pass with specific MLP neurons zeroed out.
        """
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            if layer_idx == ablate_layer:
                # Manual forward with ablation
                # Attention
                if hasattr(layer, "input_layernorm"):
                    normed = layer.input_layernorm(h)
                else:
                    normed = h

                attn_out = layer.self_attn(normed, mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h = h + attn_out

                # MLP with ablation
                if hasattr(layer, "post_attention_layernorm"):
                    mlp_input = layer.post_attention_layernorm(h)
                else:
                    mlp_input = h

                mlp = layer.mlp

                if hasattr(mlp, "gate_proj"):
                    gate = mlp.gate_proj(mlp_input)
                    up = mlp.up_proj(mlp_input)
                    gate_activated = nn.gelu(gate)
                    mlp_hidden = gate_activated * up

                    # ABLATE: zero out specific neurons using mask
                    # Create a mask that zeros out specific neurons
                    mask_arr = mx.ones(mlp_hidden.shape[-1])
                    for neuron_idx in ablate_neurons:
                        if neuron_idx < mlp_hidden.shape[-1]:
                            mask_arr = mx.concatenate(
                                [mask_arr[:neuron_idx], mx.array([0.0]), mask_arr[neuron_idx + 1 :]]
                            )
                    mlp_hidden = mlp_hidden * mask_arr

                    mlp_out = mlp.down_proj(mlp_hidden)
                else:
                    mlp_out = mlp(mlp_input)

                h = h + mlp_out
            else:
                # Normal forward
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
        logits = self.lm_head(h)
        return logits

    def forward_normal(self, tokens: list[int]) -> mx.array:
        """Normal forward pass."""
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
        logits = self.lm_head(h)
        return logits

    def forward_with_activation_patch(
        self,
        target_tokens: list[int],
        source_tokens: list[int],
        patch_layer: int,
        patch_neurons: list[int],
    ) -> mx.array:
        """
        Forward pass with specific neuron activations patched from source to target.
        """
        # First get source activations
        source_input = mx.array([source_tokens])
        source_emb = self.embed_layer(source_input) * self.embed_scale
        source_h = source_emb
        source_seq_len = source_h.shape[1]
        source_mask = nn.MultiHeadAttention.create_additive_causal_mask(source_seq_len)
        source_mask = source_mask.astype(source_h.dtype)

        source_mlp_hidden = None

        for layer_idx in range(patch_layer + 1):
            layer = self.layers[layer_idx]

            if layer_idx == patch_layer:
                # Get MLP activations
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
                    source_mlp_hidden = nn.gelu(gate) * up
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

        # Now run target with patched activations
        target_input = mx.array([target_tokens])
        target_emb = self.embed_layer(target_input) * self.embed_scale
        h = target_emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            if layer_idx == patch_layer:
                # Patch activations
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

                    # PATCH: copy source activations to target for specific neurons
                    # Create patched version by blending source and target activations
                    # Only patch last token
                    patched_last = mlp_hidden[0, -1, :].astype(mx.float32)
                    source_last = source_mlp_hidden[0, -1, :].astype(mx.float32)

                    # Create mask for neurons to patch
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

                    # Blend: patched = source * mask + target * (1 - mask)
                    patched_last = source_last * patch_mask + patched_last * (1 - patch_mask)

                    # Reconstruct mlp_hidden with patched last token
                    mlp_hidden = mx.concatenate(
                        [
                            mlp_hidden[:, :-1, :],
                            patched_last.reshape(1, 1, -1).astype(mlp_hidden.dtype),
                        ],
                        axis=1,
                    )

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
        logits = self.lm_head(h)
        return logits

    def predict_tool_calling(self, logits: mx.array) -> bool:
        """
        Simple heuristic: check if top predicted token suggests tool-calling.
        Tool-calling outputs often start with specific tokens.
        """
        # Get top token
        top_token = int(mx.argmax(logits[0, -1, :]))

        # Decode to check
        try:
            decoded = self.tokenizer.decode([top_token])
            # Heuristic: tool calls often start with [, {, or action-like tokens
            tool_indicators = ["[", "{", "<", "function", "tool", "call", "action"]
            return any(ind in decoded.lower() for ind in tool_indicators)
        except:
            return False

    def run_ablation_experiment(self, test_data: list[tuple[str, bool]]):
        """
        Experiment 1: Ablate top neurons and measure accuracy drop.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: NEURON ABLATION")
        print("=" * 60)
        print("\nQuestion: Are top discriminative neurons causally important?")

        results = []

        for layer in [10, 11, 12]:
            print(f"\n--- Layer {layer} ---")
            neurons = self.top_neurons[layer]

            for k in [1, 5, 10]:
                neurons_to_ablate = neurons[:k]

                # Measure accuracy with and without ablation
                correct_normal = 0
                correct_ablated = 0

                for prompt, label in test_data:
                    tokens = self.tokenizer.encode(prompt)
                    if isinstance(tokens, np.ndarray):
                        tokens = tokens.flatten().tolist()

                    # Normal prediction
                    normal_logits = self.forward_normal(tokens)
                    normal_pred = self.predict_tool_calling(normal_logits)

                    # Ablated prediction
                    ablated_logits = self.forward_with_neuron_ablation(
                        tokens, layer, neurons_to_ablate
                    )
                    ablated_pred = self.predict_tool_calling(ablated_logits)

                    # For this analysis, we care about consistency, not ground truth
                    # We want to see if ablation changes behavior
                    if normal_pred == label:
                        correct_normal += 1
                    if ablated_pred == label:
                        correct_ablated += 1

                normal_acc = correct_normal / len(test_data)
                ablated_acc = correct_ablated / len(test_data)
                drop = normal_acc - ablated_acc

                print(
                    f"  Ablate top {k:2d}: normal={normal_acc:.1%}, ablated={ablated_acc:.1%}, drop={drop:+.1%}"
                )

                results.append(
                    AblationResult(
                        layer=layer,
                        neurons_ablated=neurons_to_ablate,
                        original_accuracy=normal_acc,
                        ablated_accuracy=ablated_acc,
                        accuracy_drop=drop,
                    )
                )

        return results

    def run_patching_experiment(self, test_data: list[tuple[str, bool]]):
        """
        Experiment 2: Patch tool neuron activations into no-tool prompts.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: ACTIVATION PATCHING")
        print("=" * 60)
        print("\nQuestion: Can we flip predictions by patching neuron activations?")

        tool_prompts = [(p, l) for p, l in test_data if l]
        no_tool_prompts = [(p, l) for p, l in test_data if not l]

        results = []

        for layer in [11, 12]:
            print(f"\n--- Layer {layer} ---")
            neurons = self.top_neurons[layer][:5]  # Top 5 neurons

            flips = 0
            total = 0

            for source_prompt, source_label in tool_prompts[:5]:
                for target_prompt, target_label in no_tool_prompts[:5]:
                    source_tokens = self.tokenizer.encode(source_prompt)
                    target_tokens = self.tokenizer.encode(target_prompt)

                    if isinstance(source_tokens, np.ndarray):
                        source_tokens = source_tokens.flatten().tolist()
                    if isinstance(target_tokens, np.ndarray):
                        target_tokens = target_tokens.flatten().tolist()

                    # Original prediction on target
                    original_logits = self.forward_normal(target_tokens)
                    original_pred = self.predict_tool_calling(original_logits)

                    # Patched prediction (inject tool neurons from source)
                    patched_logits = self.forward_with_activation_patch(
                        target_tokens, source_tokens, layer, neurons
                    )
                    patched_pred = self.predict_tool_calling(patched_logits)

                    flipped = original_pred != patched_pred
                    if flipped:
                        flips += 1
                    total += 1

                    results.append(
                        PatchingResult(
                            source_prompt=source_prompt,
                            target_prompt=target_prompt,
                            source_label=source_label,
                            target_label=target_label,
                            original_prediction=original_pred,
                            patched_prediction=patched_pred,
                            flipped=flipped,
                        )
                    )

            flip_rate = flips / total if total > 0 else 0
            print(f"  Patch top 5 neurons: {flips}/{total} flipped ({flip_rate:.1%})")

        return results

    def run_l11_investigation(self, test_data: list[tuple[str, bool]]):
        """
        Experiment 3: Why doesn't L11 reach 90% accuracy with neurons?
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: L11 INVESTIGATION")
        print("=" * 60)
        print("\nQuestion: Why is L11 different from L10/L12/L13?")

        # Hypothesis: L11 is doing transformation, not just classification
        # Check activation statistics

        tool_acts = []
        no_tool_acts = []

        for prompt, label in test_data:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Get L11 MLP activations
            input_ids = mx.array([tokens])
            emb = self.embed_layer(input_ids) * self.embed_scale
            h = emb
            seq_len = h.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for layer_idx in range(11):
                layer = self.layers[layer_idx]
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

            # L11 MLP
            layer = self.layers[11]
            if hasattr(layer, "input_layernorm"):
                normed = layer.input_layernorm(h)
            else:
                normed = h

            attn_out = layer.self_attn(normed, mask=mask)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h_post_attn = h + attn_out

            if hasattr(layer, "post_attention_layernorm"):
                mlp_input = layer.post_attention_layernorm(h_post_attn)
            else:
                mlp_input = h_post_attn

            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                gate = mlp.gate_proj(mlp_input)
                up = mlp.up_proj(mlp_input)
                mlp_hidden = nn.gelu(gate) * up

            # Get last token activations
            act = np.array(mlp_hidden[0, -1, :].tolist())

            if label:
                tool_acts.append(act)
            else:
                no_tool_acts.append(act)

        tool_acts = np.array(tool_acts)
        no_tool_acts = np.array(no_tool_acts)

        # Analysis 1: Sparsity
        tool_sparsity = np.mean(np.abs(tool_acts) < 1e-3)
        no_tool_sparsity = np.mean(np.abs(no_tool_acts) < 1e-3)
        print("\n  Sparsity (|x| < 1e-3):")
        print(f"    Tool prompts: {tool_sparsity:.1%}")
        print(f"    No-tool prompts: {no_tool_sparsity:.1%}")

        # Analysis 2: Activation magnitude
        tool_norm = np.mean(np.linalg.norm(tool_acts, axis=1))
        no_tool_norm = np.mean(np.linalg.norm(no_tool_acts, axis=1))
        print("\n  Activation L2 norm:")
        print(f"    Tool prompts: {tool_norm:.1f}")
        print(f"    No-tool prompts: {no_tool_norm:.1f}")

        # Analysis 3: Variance per neuron
        tool_var = np.var(tool_acts, axis=0)
        no_tool_var = np.var(no_tool_acts, axis=0)
        pooled_var = (tool_var + no_tool_var) / 2

        high_var_neurons = np.sum(pooled_var > np.percentile(pooled_var, 90))
        low_var_neurons = np.sum(pooled_var < np.percentile(pooled_var, 10))
        print("\n  Neuron variance:")
        print(f"    High variance (top 10%): {high_var_neurons} neurons")
        print(f"    Low variance (bottom 10%): {low_var_neurons} neurons")

        # Analysis 4: Compare to L10 and L12
        print("\n  Hypothesis: L11 has more distributed activations (harder to classify)")

        # Get separation scores
        mean_diff = np.abs(np.mean(tool_acts, axis=0) - np.mean(no_tool_acts, axis=0))
        separation = mean_diff / (np.sqrt(pooled_var) + 1e-6)

        print(f"    Max separation score: {np.max(separation):.2f}")
        print(f"    Mean separation score: {np.mean(separation):.4f}")
        print(f"    Neurons with sep > 1.0: {np.sum(separation > 1.0)}")
        print(f"    Neurons with sep > 0.5: {np.sum(separation > 0.5)}")

    def run_experiment(self):
        """Run all causal analysis experiments."""
        print("=" * 60)
        print("NEURON CAUSAL ANALYSIS")
        print("=" * 60)

        self.load_model()
        test_data = self.prepare_dataset()

        # Experiment 1: Ablation
        ablation_results = self.run_ablation_experiment(test_data)

        # Experiment 2: Patching
        patching_results = self.run_patching_experiment(test_data)

        # Experiment 3: L11 Investigation
        self.run_l11_investigation(test_data)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\n1. ABLATION RESULTS:")
        print("   If ablating neurons drops accuracy → they're causally important")
        for r in ablation_results:
            if len(r.neurons_ablated) == 10:
                print(f"   L{r.layer}: ablate 10 → drop {r.accuracy_drop:+.1%}")

        print("\n2. PATCHING RESULTS:")
        print("   If patching flips predictions → neurons carry decision signal")
        l11_flips = sum(1 for r in patching_results if r.flipped and "L11" in str(r))
        l12_flips = sum(1 for r in patching_results if r.flipped)
        print(
            f"   Flip rate: {l12_flips}/{len(patching_results)} = {l12_flips / len(patching_results):.1%}"
        )

        print("\n3. L11 INVESTIGATION:")
        print("   L11 may be a 'transformation' layer rather than 'decision' layer")
        print("   Lower separation scores = more distributed representation")

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
If ablation causes accuracy drop:
  → Top neurons are CAUSALLY NECESSARY for tool-calling

If patching flips predictions:
  → Neurons are CAUSALLY SUFFICIENT for the decision

If L11 has lower separation but L12 has higher:
  → L11 transforms, L12 decides

This confirms the circuit:
  L0-L1: Tool info encoded in embeddings
  L2-L10: Gradual processing
  L11: Transformation/compression
  L12: Final decision amplification
  L13-L17: Output formatting
""")

        return ablation_results, patching_results


def main():
    experiment = NeuronCausalAnalysis()
    experiment.run_experiment()


if __name__ == "__main__":
    main()
