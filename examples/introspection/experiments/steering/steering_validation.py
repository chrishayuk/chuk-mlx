#!/usr/bin/env python3
"""
steering_validation.py

Validate that the steering tool actually moves representations
in the expected direction, even if token outputs are noisy.

The key test: Does steering move the L11 activations toward/away
from the tool cluster?

Run: uv run python examples/introspection/steering_validation.py
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


class SteeringValidation:
    """Validate steering effectiveness using probe-based metrics."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id

    def load_model(self):
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        self.embed_layer = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.hidden_size = self.model.model.hidden_size
        self.embed_scale = self.hidden_size**0.5
        self.num_layers = len(self.layers)

    def get_final_activations_with_steering(
        self,
        tokens: list[int],
        mode: str = "normal",
        boost_scale: float = 5000.0,
    ) -> np.ndarray:
        """Get final layer activations with optional steering."""

        # Key neurons from circuit analysis
        TOOL_PROMOTERS = [803, 2036, 831, 436]
        TOOL_SUPPRESSORS = [1347, 1237, 821, 217]
        CONTROL_LAYER = 11

        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            if layer_idx == CONTROL_LAYER and mode != "normal":
                # Apply steering
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

                    # Apply steering
                    hidden_np = np.array(mlp_hidden.tolist())

                    if mode == "force_tool":
                        for n in TOOL_PROMOTERS:
                            hidden_np[0, -1, n] += boost_scale
                        for n in TOOL_SUPPRESSORS:
                            hidden_np[0, -1, n] -= boost_scale * 0.5
                    elif mode == "prevent_tool":
                        for n in TOOL_PROMOTERS:
                            hidden_np[0, -1, n] -= boost_scale
                        for n in TOOL_SUPPRESSORS:
                            hidden_np[0, -1, n] += boost_scale * 0.5

                    mlp_hidden = mx.array(hidden_np).astype(mlp_hidden.dtype)
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
        return np.array(h[0, -1, :].tolist())

    def run_validation(self):
        print("=" * 60)
        print("STEERING VALIDATION")
        print("=" * 60)
        print("\nValidating that steering moves representations correctly.")

        self.load_model()

        # Train/test data
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
        ]

        # Step 1: Train a probe on normal activations
        print("\n1. Training probe on normal activations...")

        X_train = []
        y_train = []

        for prompt in tool_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            act = self.get_final_activations_with_steering(tokens, mode="normal")
            X_train.append(act)
            y_train.append(1)

        for prompt in no_tool_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            act = self.get_final_activations_with_steering(tokens, mode="normal")
            X_train.append(act)
            y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        train_acc = probe.score(X_train, y_train)
        print(f"   Probe train accuracy: {train_acc:.1%}")

        # Step 2: Test steering effect
        print("\n2. Testing steering effect...")
        print("\n   NO-TOOL prompts with steering:")
        print(f"   {'Prompt':<35} {'Normal':>10} {'Force':>10} {'Prevent':>10}")
        print("   " + "-" * 70)

        for prompt in no_tool_prompts[:5]:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Normal
            act_normal = self.get_final_activations_with_steering(tokens, "normal")
            prob_normal = probe.predict_proba([act_normal])[0, 1]

            # Force tool
            act_force = self.get_final_activations_with_steering(tokens, "force_tool")
            prob_force = probe.predict_proba([act_force])[0, 1]

            # Prevent tool
            act_prevent = self.get_final_activations_with_steering(tokens, "prevent_tool")
            prob_prevent = probe.predict_proba([act_prevent])[0, 1]

            prompt_short = prompt[:33] + ".." if len(prompt) > 35 else prompt
            print(
                f"   {prompt_short:<35} {prob_normal:>9.1%} {prob_force:>9.1%} {prob_prevent:>9.1%}"
            )

        print("\n   TOOL prompts with steering:")
        print(f"   {'Prompt':<35} {'Normal':>10} {'Force':>10} {'Prevent':>10}")
        print("   " + "-" * 70)

        for prompt in tool_prompts[:5]:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            act_normal = self.get_final_activations_with_steering(tokens, "normal")
            prob_normal = probe.predict_proba([act_normal])[0, 1]

            act_force = self.get_final_activations_with_steering(tokens, "force_tool")
            prob_force = probe.predict_proba([act_force])[0, 1]

            act_prevent = self.get_final_activations_with_steering(tokens, "prevent_tool")
            prob_prevent = probe.predict_proba([act_prevent])[0, 1]

            prompt_short = prompt[:33] + ".." if len(prompt) > 35 else prompt
            print(
                f"   {prompt_short:<35} {prob_normal:>9.1%} {prob_force:>9.1%} {prob_prevent:>9.1%}"
            )

        # Step 3: Quantify steering effect
        print("\n3. Quantifying steering effect...")

        force_increases = []
        prevent_decreases = []

        for prompt in no_tool_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            act_normal = self.get_final_activations_with_steering(tokens, "normal")
            act_force = self.get_final_activations_with_steering(tokens, "force_tool")
            act_prevent = self.get_final_activations_with_steering(tokens, "prevent_tool")

            prob_normal = probe.predict_proba([act_normal])[0, 1]
            prob_force = probe.predict_proba([act_force])[0, 1]
            prob_prevent = probe.predict_proba([act_prevent])[0, 1]

            force_increases.append(prob_force - prob_normal)
            prevent_decreases.append(prob_normal - prob_prevent)

        print("   On NO-TOOL prompts:")
        print(f"     Force tool increases P(tool) by: {np.mean(force_increases):+.1%} (avg)")
        print(f"     Prevent tool decreases P(tool) by: {np.mean(prevent_decreases):+.1%} (avg)")

        force_increases = []
        prevent_decreases = []

        for prompt in tool_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            act_normal = self.get_final_activations_with_steering(tokens, "normal")
            act_force = self.get_final_activations_with_steering(tokens, "force_tool")
            act_prevent = self.get_final_activations_with_steering(tokens, "prevent_tool")

            prob_normal = probe.predict_proba([act_normal])[0, 1]
            prob_force = probe.predict_proba([act_force])[0, 1]
            prob_prevent = probe.predict_proba([act_prevent])[0, 1]

            force_increases.append(prob_force - prob_normal)
            prevent_decreases.append(prob_normal - prob_prevent)

        print("\n   On TOOL prompts:")
        print(f"     Force tool increases P(tool) by: {np.mean(force_increases):+.1%} (avg)")
        print(f"     Prevent tool decreases P(tool) by: {np.mean(prevent_decreases):+.1%} (avg)")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("""
If steering is working:
- "Force tool" should INCREASE P(tool) for all prompts
- "Prevent tool" should DECREASE P(tool) for all prompts

This validates that the L11 neurons we identified:
- Neuron 803, 2036, 831: Promote tool-calling
- Neuron 1237, 821, 1347: Suppress tool-calling

Are causally connected to the model's internal representation
of "tool vs no-tool".
""")


def main():
    validator = SteeringValidation()
    validator.run_validation()


if __name__ == "__main__":
    main()
