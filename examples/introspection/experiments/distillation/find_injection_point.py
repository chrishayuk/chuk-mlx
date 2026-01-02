#!/usr/bin/env python3
"""
find_injection_point.py

Empirically determine where transformed embeddings should rejoin
the model for optimal tool-calling accuracy.

The question: At which layer can we inject transformed embeddings
and still get correct outputs?

Run: uv run python examples/introspection/find_injection_point.py
"""

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InjectionResult:
    """Result of injecting at a specific layer."""
    layer: int
    accuracy: float
    similarity_to_normal: float
    tool_accuracy: float
    no_tool_accuracy: float


class SimpleAttention(nn.Module):
    """Simple self-attention for context mixing."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = nn.RoPE(dims=self.head_dim, base=10000.0)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        scale = self.head_dim ** -0.5
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class SimpleMLP(nn.Module):
    """Simple MLP for feature mixing."""

    def __init__(self, hidden_size: int, intermediate_size: int = 1024):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


class EmbeddingTransform(nn.Module):
    """
    Transform raw embeddings to match what a specific layer expects.

    This is the "context encoder" that replaces early layers.
    """

    def __init__(
        self,
        hidden_size: int = 640,
        num_attention_layers: int = 1,
        intermediate_size: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-attention norm
        self.input_norm = nn.RMSNorm(hidden_size)

        # Attention layers for context
        self.attention_layers = []
        self.mlp_layers = []
        self.norms = []

        for _ in range(num_attention_layers):
            self.attention_layers.append(SimpleAttention(hidden_size))
            self.mlp_layers.append(SimpleMLP(hidden_size, intermediate_size))
            self.norms.append(nn.RMSNorm(hidden_size))

    def __call__(self, embeddings: mx.array, mask: mx.array | None = None) -> mx.array:
        # Normalize first
        x = self.input_norm(embeddings)

        # Apply attention layers
        for attn, mlp, norm in zip(self.attention_layers, self.mlp_layers, self.norms):
            # Attention with residual
            x = x + attn(x, mask)
            # MLP with residual
            x = x + mlp(norm(x))

        return x


class InjectionPointExperiment:
    """
    Find optimal injection point for transformed embeddings.
    """

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.embeddings = None

    def load_model(self):
        """Load the full model."""
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        # Get embedding layer
        if hasattr(self.model, "model"):
            self.embed_layer = self.model.model.embed_tokens
            self.layers = self.model.model.layers
            self.final_norm = self.model.model.norm
            self.hidden_size = self.model.model.hidden_size
            self.embed_scale = self.hidden_size ** 0.5
        else:
            self.embed_layer = self.model.embed_tokens
            self.layers = self.model.layers
            self.final_norm = self.model.norm
            self.hidden_size = 640
            self.embed_scale = self.hidden_size ** 0.5

        self.embeddings = self.embed_layer.weight.astype(mx.float32)
        self.num_layers = len(self.layers)

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Embedding scale: {self.embed_scale:.1f}")

    def get_normal_activation(self, prompt: str, layer: int) -> mx.array:
        """Get normal activation at a layer (for comparison)."""
        from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        hooks = ModelHooks(self.model)
        hooks.configure(CaptureConfig(
            layers=[layer],
            capture_hidden_states=True,
            positions=PositionSelection.ALL,
        ))

        input_ids = mx.array([tokens])
        hooks.forward(input_ids)

        if layer in hooks.state.hidden_states:
            h = hooks.state.hidden_states[layer]
            return h.astype(mx.float32)
        return None

    def get_raw_embeddings(self, prompt: str) -> mx.array:
        """Get raw embeddings (before any layer)."""
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()

        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids)

        # Apply Gemma's embedding scale
        emb = emb * self.embed_scale

        return emb.astype(mx.float32)

    def run_from_layer(
        self,
        hidden_states: mx.array,
        start_layer: int,
    ) -> mx.array:
        """Run model from a specific layer using injected hidden states."""

        h = hidden_states

        # Create mask
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Run remaining layers
        for layer_idx in range(start_layer, self.num_layers):
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

        # Final norm
        h = self.final_norm(h)

        return h

    def compute_similarity(self, a: mx.array, b: mx.array) -> float:
        """Compute cosine similarity between activations."""
        # Use last token
        a_last = a[0, -1, :].astype(mx.float32)
        b_last = b[0, -1, :].astype(mx.float32)

        a_norm = mx.sqrt(mx.sum(a_last * a_last))
        b_norm = mx.sqrt(mx.sum(b_last * b_last))

        if a_norm > 0 and b_norm > 0:
            sim = mx.sum(a_last * b_last) / (a_norm * b_norm)
            return float(sim)
        return 0.0

    def test_injection_point(
        self,
        inject_layer: int,
        transform: EmbeddingTransform | None,
        test_prompts: list[tuple[str, bool]],
    ) -> InjectionResult:
        """Test accuracy when injecting at a specific layer."""

        correct_tool = 0
        correct_no_tool = 0
        total_tool = 0
        total_no_tool = 0
        similarities = []

        for prompt, should_call_tool in test_prompts:
            # Get raw embeddings
            emb = self.get_raw_embeddings(prompt)

            # Apply transform if provided
            if transform is not None:
                seq_len = emb.shape[1]
                mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
                mask = mask.astype(emb.dtype)
                emb = transform(emb, mask)

            # Scale to match expected norm at injection layer
            # (We analyzed that L0 has ~500x norm increase)
            if inject_layer == 0:
                # No scaling needed, will go through all layers
                pass
            else:
                # Get target norm from a sample
                sample_act = self.get_normal_activation(prompt, inject_layer - 1)
                if sample_act is not None:
                    target_norm = float(mx.mean(mx.sqrt(mx.sum(sample_act * sample_act, axis=-1))))
                    current_norm = float(mx.mean(mx.sqrt(mx.sum(emb * emb, axis=-1))))
                    if current_norm > 0:
                        emb = emb * (target_norm / current_norm)

            # Run from injection point
            output = self.run_from_layer(emb, start_layer=inject_layer)

            # Get normal output for comparison
            normal_act = self.get_normal_activation(prompt, self.num_layers - 1)
            if normal_act is not None:
                sim = self.compute_similarity(output, normal_act)
                similarities.append(sim)

            # Simple heuristic: check if output looks like tool-calling
            # (In a real setup, we'd use the LM head to generate and check)
            output_last = output[0, -1, :].astype(mx.float32)
            output_norm = float(mx.sqrt(mx.sum(output_last * output_last)))

            # Use norm as proxy (tool-calling outputs tend to have different norms)
            # This is a simplified heuristic
            if should_call_tool:
                total_tool += 1
                if output_norm > 50:  # Arbitrary threshold
                    correct_tool += 1
            else:
                total_no_tool += 1
                if output_norm <= 50:
                    correct_no_tool += 1

        tool_acc = correct_tool / total_tool if total_tool > 0 else 0
        no_tool_acc = correct_no_tool / total_no_tool if total_no_tool > 0 else 0
        overall_acc = (correct_tool + correct_no_tool) / len(test_prompts)
        mean_sim = np.mean(similarities) if similarities else 0

        return InjectionResult(
            layer=inject_layer,
            accuracy=overall_acc,
            similarity_to_normal=mean_sim,
            tool_accuracy=tool_acc,
            no_tool_accuracy=no_tool_acc,
        )

    def analyze_layer_norms(self, sample_prompts: list[str]):
        """Analyze what norms each layer expects."""
        print("\n" + "=" * 60)
        print("LAYER NORM ANALYSIS")
        print("=" * 60)

        print("\nWhat norm does each layer's input have?")
        print("-" * 50)

        for layer in [0, 3, 6, 9, 11, 12, 14, 17]:
            if layer >= self.num_layers:
                continue

            norms = []
            for prompt in sample_prompts[:5]:
                if layer == 0:
                    act = self.get_raw_embeddings(prompt)
                else:
                    act = self.get_normal_activation(prompt, layer - 1)

                if act is not None:
                    norm = float(mx.mean(mx.sqrt(mx.sum(act * act, axis=-1))))
                    norms.append(norm)

            if norms:
                mean_norm = np.mean(norms)
                print(f"  L{layer:2d} input norm: {mean_norm:8.1f}")

    def run_experiment(self):
        """Run the full injection point experiment."""
        print("=" * 60)
        print("INJECTION POINT ANALYSIS")
        print("=" * 60)

        self.load_model()

        # Test prompts
        test_prompts = [
            # Tool-calling
            ("What is the weather in Tokyo?", True),
            ("Send an email to John", True),
            ("Create a calendar event", True),
            ("Search for restaurants nearby", True),
            ("Set a timer for 10 minutes", True),
            ("Get the stock price of Apple", True),
            ("Book a flight to Paris", True),
            ("Check my schedule", True),
            ("Find hotels in London", True),
            ("Calculate 25 times 4", True),
            # No tool
            ("What is the capital of France?", False),
            ("Explain quantum physics", False),
            ("Write a poem about the ocean", False),
            ("What is 2 + 2?", False),
            ("Tell me about Einstein", False),
            ("How do I learn Python?", False),
            ("What is the meaning of life?", False),
            ("Tell me a joke", False),
            ("What is photosynthesis?", False),
            ("Describe a rainbow", False),
        ]

        # Analyze layer norms first
        self.analyze_layer_norms([p for p, _ in test_prompts[:5]])

        # Test different injection points
        print("\n" + "=" * 60)
        print("INJECTION POINT RESULTS")
        print("=" * 60)

        injection_layers = [0, 3, 6, 9, 10, 11, 12, 14, self.num_layers - 1]
        injection_layers = [l for l in injection_layers if l < self.num_layers]

        results = []

        for inject_at in injection_layers:
            # Create transform (more layers for earlier injection)
            if inject_at <= 3:
                num_attn_layers = 2
            elif inject_at <= 6:
                num_attn_layers = 1
            else:
                num_attn_layers = 0

            if num_attn_layers > 0:
                transform = EmbeddingTransform(
                    hidden_size=self.hidden_size,
                    num_attention_layers=num_attn_layers,
                )
            else:
                transform = None

            result = self.test_injection_point(inject_at, transform, test_prompts)
            results.append(result)

            transform_desc = f"+ {num_attn_layers} attn" if num_attn_layers > 0 else "direct"
            print(f"\nL{inject_at:2d} ({transform_desc}):")
            print(f"  Accuracy: {result.accuracy:.1%}")
            print(f"  Similarity to normal: {result.similarity_to_normal:.3f}")
            print(f"  Tool accuracy: {result.tool_accuracy:.1%}")
            print(f"  No-tool accuracy: {result.no_tool_accuracy:.1%}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n{'Layer':<8} {'Transform':<12} {'Accuracy':>10} {'Similarity':>12}")
        print("-" * 44)

        for result in results:
            layer = result.layer
            if layer <= 3:
                transform = "+ 2 attn"
            elif layer <= 6:
                transform = "+ 1 attn"
            else:
                transform = "direct"

            print(f"L{layer:<7} {transform:<12} {result.accuracy:>9.1%} {result.similarity_to_normal:>11.3f}")

        # Find optimal
        best = max(results, key=lambda r: r.similarity_to_normal)
        print(f"\nOptimal injection point: L{best.layer}")
        print(f"  Similarity: {best.similarity_to_normal:.3f}")
        print(f"  Accuracy: {best.accuracy:.1%}")

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
Higher similarity = transformed embeddings match what that layer
normally sees, so the remaining layers work correctly.

Optimal injection point should have:
1. High similarity (> 0.9) - matches expected input
2. Low layer number - skips most computation
3. Minimal transform needed - efficient

For tool-calling classification:
- If L6 works well → can skip 6 layers (33% speedup)
- If L11 works well → can skip 11 layers (61% speedup)
""")

        return results


def main():
    experiment = InjectionPointExperiment()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
