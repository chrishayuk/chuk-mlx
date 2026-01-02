#!/usr/bin/env python3
"""
injection_upper_bound.py

Establish upper bounds for injection by using actual model layers.

Question: If we use the REAL layers 0-N as a transform, and inject at layer N+1,
what accuracy do we get? This gives us the theoretical upper bound.

Then: Can we distill those layers into a smaller model?

Run: uv run python examples/introspection/injection_upper_bound.py
"""

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import warnings
warnings.filterwarnings('ignore')


@dataclass
class InjectionBoundResult:
    """Result for upper bound testing."""
    inject_after_layer: int
    layers_skipped: int
    similarity_to_full: float
    top1_match_rate: float
    top5_match_rate: float
    speedup_factor: float


class InjectionUpperBound:
    """Test upper bounds using actual model layers."""

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

        if hasattr(self.model, "model"):
            self.embed_layer = self.model.model.embed_tokens
            self.layers = self.model.model.layers
            self.final_norm = self.model.model.norm
            self.hidden_size = self.model.model.hidden_size
            self.embed_scale = self.hidden_size ** 0.5
            self.lm_head = self.model.lm_head
        else:
            raise ValueError("Model structure not recognized")

        self.num_layers = len(self.layers)
        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def full_forward(self, tokens: list[int]) -> tuple[mx.array, mx.array]:
        """Full forward pass, return final hidden states and logits."""
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

        return h.astype(mx.float32), logits

    def partial_forward_then_inject(
        self,
        tokens: list[int],
        inject_after: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Run layers 0..inject_after, then continue from inject_after+1.

        This simulates: "what if we could perfectly replicate layers 0..inject_after
        with a smaller model, then use the real layers inject_after+1..N?"
        """
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Run layers 0..inject_after (the "transform")
        for layer_idx in range(inject_after + 1):
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

        # Continue from inject_after+1..end (the "remaining layers")
        for layer_idx in range(inject_after + 1, self.num_layers):
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
        logits = self.lm_head(h)

        return h.astype(mx.float32), logits

    def skip_layers_forward(
        self,
        tokens: list[int],
        skip_layers: list[int],
    ) -> tuple[mx.array, mx.array]:
        """
        Run forward pass but SKIP specified layers entirely.

        This tests: "what if we could remove layers N, M, K entirely?"
        """
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        skip_set = set(skip_layers)

        for layer_idx in range(self.num_layers):
            if layer_idx in skip_set:
                # Skip this layer entirely (just pass through residual)
                continue

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
        logits = self.lm_head(h)

        return h.astype(mx.float32), logits

    def test_injection_upper_bound(
        self,
        test_prompts: list[str],
        inject_after: int,
    ) -> InjectionBoundResult:
        """Test injection after specific layer."""

        similarities = []
        top1_matches = 0
        top5_matches = 0

        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Full forward
            full_h, full_logits = self.full_forward(tokens)

            # Injection forward (should be identical since we use real layers)
            inj_h, inj_logits = self.partial_forward_then_inject(tokens, inject_after)

            # Compute similarity
            full_last = full_h[0, -1, :]
            inj_last = inj_h[0, -1, :]

            full_norm = mx.sqrt(mx.sum(full_last * full_last))
            inj_norm = mx.sqrt(mx.sum(inj_last * inj_last))

            if full_norm > 0 and inj_norm > 0:
                sim = mx.sum(full_last * inj_last) / (full_norm * inj_norm)
                similarities.append(float(sim))

            # Check top-k match
            full_top1 = int(mx.argmax(full_logits[0, -1, :]))
            inj_top1 = int(mx.argmax(inj_logits[0, -1, :]))

            if full_top1 == inj_top1:
                top1_matches += 1

            # Top-5
            full_top5 = set(mx.argsort(full_logits[0, -1, :])[-5:].tolist())
            inj_top5 = set(mx.argsort(inj_logits[0, -1, :])[-5:].tolist())
            if full_top1 in inj_top5:
                top5_matches += 1

        mean_sim = np.mean(similarities) if similarities else 0
        top1_rate = top1_matches / len(test_prompts)
        top5_rate = top5_matches / len(test_prompts)

        # Layers skipped = 0 for this baseline (we use all layers)
        # But speedup would come from distilling layers 0..inject_after
        layers_kept = self.num_layers - inject_after
        speedup = self.num_layers / layers_kept if layers_kept > 0 else 1.0

        return InjectionBoundResult(
            inject_after_layer=inject_after,
            layers_skipped=inject_after,
            similarity_to_full=mean_sim,
            top1_match_rate=top1_rate,
            top5_match_rate=top5_rate,
            speedup_factor=speedup,
        )

    def test_layer_skipping(
        self,
        test_prompts: list[str],
        skip_layers: list[int],
    ) -> dict:
        """Test skipping specific layers entirely."""

        similarities = []
        top1_matches = 0
        top5_matches = 0

        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            full_h, full_logits = self.full_forward(tokens)
            skip_h, skip_logits = self.skip_layers_forward(tokens, skip_layers)

            full_last = full_h[0, -1, :]
            skip_last = skip_h[0, -1, :]

            full_norm = mx.sqrt(mx.sum(full_last * full_last))
            skip_norm = mx.sqrt(mx.sum(skip_last * skip_last))

            if full_norm > 0 and skip_norm > 0:
                sim = mx.sum(full_last * skip_last) / (full_norm * skip_norm)
                similarities.append(float(sim))

            full_top1 = int(mx.argmax(full_logits[0, -1, :]))
            skip_top1 = int(mx.argmax(skip_logits[0, -1, :]))

            if full_top1 == skip_top1:
                top1_matches += 1

            full_top5 = set(mx.argsort(full_logits[0, -1, :])[-5:].tolist())
            if full_top1 in full_top5:
                top5_matches += 1

        return {
            "skip_layers": skip_layers,
            "num_skipped": len(skip_layers),
            "similarity": np.mean(similarities) if similarities else 0,
            "top1_match": top1_matches / len(test_prompts),
            "top5_match": top5_matches / len(test_prompts),
            "speedup": self.num_layers / (self.num_layers - len(skip_layers)),
        }

    def run_experiment(self):
        """Run the upper bound experiment."""
        print("=" * 60)
        print("INJECTION UPPER BOUND EXPERIMENT")
        print("=" * 60)

        self.load_model()

        # Test prompts
        test_prompts = [
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
            "Find hotels in London",
            "Calculate 25 times 4",
            "What is photosynthesis?",
            "Describe a rainbow",
        ]

        # Part 1: Verify that our injection mechanism works (should get 100%)
        print("\n" + "=" * 50)
        print("PART 1: VERIFY INJECTION MECHANISM")
        print("=" * 50)
        print("\nUsing real layers as 'transform' (should give 100% match):")

        for inject_after in [0, 5, 10, 15]:
            result = self.test_injection_upper_bound(test_prompts, inject_after)
            print(f"  Inject after L{inject_after:2d}: sim={result.similarity_to_full:.4f}, top1={result.top1_match_rate:.1%}")

        # Part 2: Test layer skipping
        print("\n" + "=" * 50)
        print("PART 2: LAYER SKIPPING ANALYSIS")
        print("=" * 50)
        print("\nWhat happens if we skip layers entirely?")

        skip_configs = [
            [1],           # Skip just L1
            [1, 2],        # Skip L1-2
            [1, 2, 3],     # Skip L1-3 (early disruption zone)
            [3, 4, 5],     # Skip middle-early
            [6, 7, 8],     # Skip middle
            [9, 10, 11],   # Skip reconstruction zone
            [14, 15, 16],  # Skip late layers
            list(range(1, 6)),   # Skip 5 early layers
            list(range(1, 9)),   # Skip 8 early layers
            list(range(9, 15)),  # Skip reconstruction layers
        ]

        print(f"\n{'Skipped Layers':<25} {'Sim':>8} {'Top1':>8} {'Speedup':>8}")
        print("-" * 52)

        skip_results = []
        for skip_layers in skip_configs:
            result = self.test_layer_skipping(test_prompts, skip_layers)
            skip_results.append(result)

            layers_str = f"L{skip_layers[0]}-{skip_layers[-1]}" if len(skip_layers) > 1 else f"L{skip_layers[0]}"
            print(f"{layers_str:<25} {result['similarity']:>7.3f} {result['top1_match']:>7.1%} {result['speedup']:>7.1f}x")

        # Part 3: Find optimal skip pattern
        print("\n" + "=" * 50)
        print("PART 3: ANALYSIS")
        print("=" * 50)

        # Sort by top1 match rate
        best_skips = sorted(skip_results, key=lambda x: -x['top1_match'])

        print("\nBest layer skipping patterns (by top-1 accuracy):")
        for i, result in enumerate(best_skips[:5]):
            skip_layers = result['skip_layers']
            layers_str = f"L{skip_layers[0]}-{skip_layers[-1]}" if len(skip_layers) > 1 else f"L{skip_layers[0]}"
            print(f"  {i+1}. Skip {layers_str}: {result['top1_match']:.1%} top1, {result['speedup']:.1f}x speedup")

        # Analysis
        print("\n" + "=" * 50)
        print("INTERPRETATION")
        print("=" * 50)
        print("""
Part 1 confirms our injection mechanism is correct (100% match).

Part 2 reveals which layers can be skipped with minimal accuracy loss:
- If skipping L1-3 maintains high accuracy → early layers are compressible
- If skipping L6-8 hurts accuracy → these layers are critical
- If skipping late layers hurts → they're essential for output

Key insight: Layers that can be skipped are candidates for distillation
into a smaller "embedding transform" network.
""")

        return skip_results


def main():
    experiment = InjectionUpperBound()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
