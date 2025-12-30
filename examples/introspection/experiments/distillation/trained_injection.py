#!/usr/bin/env python3
"""
trained_injection.py

Train transforms to match actual layer activations, then test injection.

The key insight: untrained transforms can't match layer activations.
We need to TRAIN the transform to produce matching activations.

Run: uv run python examples/introspection/trained_injection.py
"""

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainedInjectionResult:
    """Result of trained injection at a specific layer."""
    layer: int
    train_loss: float
    final_similarity: float
    tool_accuracy: float
    no_tool_accuracy: float
    overall_accuracy: float
    speedup_factor: float


class LightTransformBlock(nn.Module):
    """Lightweight transformer block for embedding transformation."""

    def __init__(self, hidden_size: int, num_heads: int = 4, intermediate_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention
        self.input_norm = nn.RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = nn.RoPE(dims=self.head_dim, base=10000.0)

        # MLP
        self.post_attn_norm = nn.RMSNorm(hidden_size)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # Attention with residual
        normed = self.input_norm(x)
        q = self.q_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        scale = self.head_dim ** -0.5
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        x = x + self.o_proj(attn_out)

        # MLP with residual
        normed = self.post_attn_norm(x)
        x = x + self.down(nn.gelu(self.up(normed)))

        return x


class EmbeddingToLayerTransform(nn.Module):
    """
    Transform embeddings to match what a specific target layer expects.

    Trained to minimize MSE between:
    - transform(embeddings)
    - actual_layer_input (from full model forward pass)
    """

    def __init__(
        self,
        hidden_size: int = 640,
        num_layers: int = 2,
        intermediate_size: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Stack of light blocks
        self.layers = [
            LightTransformBlock(hidden_size, intermediate_size=intermediate_size)
            for _ in range(num_layers)
        ]

        # Final norm to match layer input
        self.output_norm = nn.RMSNorm(hidden_size)

        # Learnable scale factor (layers have different norms)
        self.scale = mx.array([1.0])

    def __call__(self, embeddings: mx.array, mask: mx.array | None = None) -> mx.array:
        x = embeddings
        for layer in self.layers:
            x = layer(x, mask)
        x = self.output_norm(x)
        x = x * self.scale
        return x


class TrainedInjectionExperiment:
    """
    Train transforms to match layer activations, then test injection.
    """

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the full model."""
        print(f"Loading model: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        # Get model components
        if hasattr(self.model, "model"):
            self.embed_layer = self.model.model.embed_tokens
            self.layers = self.model.model.layers
            self.final_norm = self.model.model.norm
            self.hidden_size = self.model.model.hidden_size
            self.embed_scale = self.hidden_size ** 0.5
            self.lm_head = self.model.lm_head
        else:
            self.embed_layer = self.model.embed_tokens
            self.layers = self.model.layers
            self.final_norm = self.model.norm
            self.hidden_size = 640
            self.embed_scale = self.hidden_size ** 0.5
            self.lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None

        self.num_layers = len(self.layers)
        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def get_embeddings(self, tokens: list[int]) -> mx.array:
        """Get scaled embeddings for tokens."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids)
        return (emb * self.embed_scale).astype(mx.float32)

    def get_layer_input(self, tokens: list[int], target_layer: int) -> mx.array:
        """Get actual activation that enters target_layer (output of layer target_layer-1)."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Run layers up to target_layer
        for layer_idx in range(target_layer):
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

        return h.astype(mx.float32)

    def run_from_layer(self, hidden_states: mx.array, start_layer: int) -> mx.array:
        """Run model from a specific layer."""
        h = hidden_states

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

        h = self.final_norm(h)
        return h

    def get_full_output(self, tokens: list[int]) -> mx.array:
        """Get full model output for comparison."""
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

    def train_transform(
        self,
        target_layer: int,
        train_prompts: list[str],
        num_epochs: int = 50,
        lr: float = 1e-3,
    ) -> EmbeddingToLayerTransform:
        """Train a transform to match target layer's expected input."""

        print(f"\n  Training transform for L{target_layer}...")

        # Determine number of transform layers based on target
        if target_layer <= 3:
            num_transform_layers = 3
        elif target_layer <= 6:
            num_transform_layers = 2
        else:
            num_transform_layers = 1

        transform = EmbeddingToLayerTransform(
            hidden_size=self.hidden_size,
            num_layers=num_transform_layers,
        )

        # Prepare training data
        train_data = []
        for prompt in train_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            emb = self.get_embeddings(tokens)
            target = self.get_layer_input(tokens, target_layer)
            train_data.append((emb, target))

        # Training
        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(model, emb, target):
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            pred = model(emb, mask)

            # MSE loss on last token (most important for classification)
            mse = mx.mean((pred[:, -1, :] - target[:, -1, :]) ** 2)

            # Also cosine similarity loss
            pred_last = pred[:, -1, :]
            target_last = target[:, -1, :]
            pred_norm = mx.sqrt(mx.sum(pred_last * pred_last, axis=-1, keepdims=True) + 1e-8)
            target_norm = mx.sqrt(mx.sum(target_last * target_last, axis=-1, keepdims=True) + 1e-8)
            cosine = mx.sum((pred_last / pred_norm) * (target_last / target_norm), axis=-1)
            cosine_loss = 1.0 - mx.mean(cosine)

            return mse + cosine_loss * 0.1

        loss_and_grad = nn.value_and_grad(transform, loss_fn)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for emb, target in train_data:
                loss, grads = loss_and_grad(transform, emb, target)
                optimizer.update(transform, grads)
                mx.eval(transform.parameters())
                epoch_loss += float(loss)

            avg_loss = epoch_loss / len(train_data)
            losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch}: loss = {avg_loss:.4f}")

        return transform

    def test_injection(
        self,
        target_layer: int,
        transform: EmbeddingToLayerTransform,
        test_prompts: list[tuple[str, bool]],
    ) -> TrainedInjectionResult:
        """Test injection accuracy with trained transform."""

        correct_tool = 0
        correct_no_tool = 0
        total_tool = 0
        total_no_tool = 0
        similarities = []

        for prompt, should_call_tool in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Get embeddings and transform
            emb = self.get_embeddings(tokens)
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            transformed = transform(emb, mask)

            # Run from injection point
            injected_output = self.run_from_layer(transformed, target_layer)

            # Get full model output for comparison
            full_output = self.get_full_output(tokens)

            # Compute similarity
            inj_last = injected_output[0, -1, :].astype(mx.float32)
            full_last = full_output[0, -1, :].astype(mx.float32)

            inj_norm = mx.sqrt(mx.sum(inj_last * inj_last))
            full_norm = mx.sqrt(mx.sum(full_last * full_last))

            if inj_norm > 0 and full_norm > 0:
                sim = mx.sum(inj_last * full_last) / (inj_norm * full_norm)
                similarities.append(float(sim))

            # Get logits for classification
            if self.lm_head is not None:
                inj_logits = self.lm_head(injected_output[:, -1, :])
                full_logits = self.lm_head(full_output[:, -1, :])

                # Check if top predictions match
                inj_top = int(mx.argmax(inj_logits, axis=-1)[0])
                full_top = int(mx.argmax(full_logits, axis=-1)[0])

                # Heuristic: tool-calling often starts with specific tokens
                # For FunctionGemma, tool calls might start with [, {, etc.
                tool_tokens = set(self.tokenizer.encode("[") if hasattr(self.tokenizer, 'encode') else [])

                if should_call_tool:
                    total_tool += 1
                    if inj_top == full_top:
                        correct_tool += 1
                else:
                    total_no_tool += 1
                    if inj_top == full_top:
                        correct_no_tool += 1
            else:
                # Fallback to norm-based heuristic
                if should_call_tool:
                    total_tool += 1
                    correct_tool += 1 if float(inj_norm) > 50 else 0
                else:
                    total_no_tool += 1
                    correct_no_tool += 1 if float(inj_norm) <= 50 else 0

        tool_acc = correct_tool / total_tool if total_tool > 0 else 0
        no_tool_acc = correct_no_tool / total_no_tool if total_no_tool > 0 else 0
        overall_acc = (correct_tool + correct_no_tool) / len(test_prompts)
        mean_sim = np.mean(similarities) if similarities else 0

        # Speedup factor (skipping target_layer layers out of num_layers)
        speedup = self.num_layers / (self.num_layers - target_layer + 1)

        return TrainedInjectionResult(
            layer=target_layer,
            train_loss=0.0,  # Will be filled in
            final_similarity=mean_sim,
            tool_accuracy=tool_acc,
            no_tool_accuracy=no_tool_acc,
            overall_accuracy=overall_acc,
            speedup_factor=speedup,
        )

    def run_experiment(self):
        """Run the full trained injection experiment."""
        print("=" * 60)
        print("TRAINED INJECTION EXPERIMENT")
        print("=" * 60)

        self.load_model()

        # Training prompts (used to train transforms)
        train_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "What is the capital of France?",
            "Explain quantum physics",
            "Create a calendar event",
            "Write a poem about the ocean",
            "Search for restaurants nearby",
            "What is 2 + 2?",
            "Set a timer for 10 minutes",
            "Tell me about Einstein",
            "Get the stock price of Apple",
            "How do I learn Python?",
            "Book a flight to Paris",
            "What is the meaning of life?",
            "Check my schedule",
            "Tell me a joke",
        ]

        # Test prompts (separate from training)
        test_prompts = [
            ("Find hotels in London", True),
            ("Calculate 25 times 4", True),
            ("Order food delivery", True),
            ("Play some music", True),
            ("Turn off the lights", True),
            ("What is photosynthesis?", False),
            ("Describe a rainbow", False),
            ("Who wrote Romeo and Juliet?", False),
            ("What is the speed of light?", False),
            ("Explain democracy", False),
        ]

        # Test layers
        target_layers = [3, 6, 9, 12]

        results = []
        for target_layer in target_layers:
            print(f"\n{'='*50}")
            print(f"TARGET LAYER: L{target_layer}")
            print("=" * 50)

            # Train transform
            transform = self.train_transform(
                target_layer,
                train_prompts,
                num_epochs=30,
            )

            # Test
            result = self.test_injection(target_layer, transform, test_prompts)
            results.append(result)

            print(f"\n  Results for L{target_layer}:")
            print(f"    Similarity to full output: {result.final_similarity:.3f}")
            print(f"    Overall accuracy (top-1 match): {result.overall_accuracy:.1%}")
            print(f"    Speedup factor: {result.speedup_factor:.1f}x")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n{'Layer':<8} {'Similarity':>12} {'Accuracy':>10} {'Speedup':>10}")
        print("-" * 44)

        for result in results:
            print(f"L{result.layer:<7} {result.final_similarity:>11.3f} {result.overall_accuracy:>9.1%} {result.speedup_factor:>9.1f}x")

        # Find best
        best = max(results, key=lambda r: r.final_similarity)
        print(f"\nBest injection point: L{best.layer}")
        print(f"  Similarity: {best.final_similarity:.3f}")
        print(f"  Accuracy: {best.overall_accuracy:.1%}")
        print(f"  Speedup: {best.speedup_factor:.1f}x")

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
With TRAINED transforms, we can measure:
1. How well can we replicate layer inputs?
2. Does the remaining model produce correct outputs?

Key insight: If similarity > 0.9 and accuracy > 80%,
that layer is a viable injection point.

The tradeoff:
- Earlier layers → more speedup, harder to match
- Later layers → less speedup, easier to match
""")

        return results


def main():
    experiment = TrainedInjectionExperiment()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
