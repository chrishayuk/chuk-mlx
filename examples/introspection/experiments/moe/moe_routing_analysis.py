#!/usr/bin/env python3
"""
MoE Routing Analysis Demo

This script demonstrates MoE introspection capabilities:
1. Capture router decisions across layers
2. Analyze expert utilization (load balancing)
3. Measure router confidence (entropy)
4. Compare routing patterns across prompts

Supported models:
- GPT-OSS (32 experts, 4 active)
- Llama4 (shared + routed experts)
- Granite-Hybrid (MoE + Mamba)
- Mixtral (8 experts, 2 active)
- Any model with MoE layers

Usage:
    # With GPT-OSS (if you have it downloaded)
    python moe_routing_analysis.py --model openai/gpt-oss

    # With a local test model
    python moe_routing_analysis.py --test

    # Analyze specific prompts
    python moe_routing_analysis.py --prompts "Hello world" "def fibonacci(n):"
"""

import argparse

# Add parent to path for imports
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from chuk_lazarus.introspection.moe import (
    MoECaptureConfig,
    MoEHooks,
    detect_moe_architecture,
)


def create_test_moe_model(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 4,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
):
    """Create a simple MoE model for testing."""

    class SimpleMoERouter(nn.Module):
        def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
            super().__init__()
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_tok
            self.weight = mx.random.normal((num_experts, hidden_size)) * 0.02
            self.bias = mx.zeros((num_experts,))

        def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            logits = x @ self.weight.T + self.bias
            k = self.num_experts_per_tok
            indices = mx.argsort(logits, axis=-1)[:, -k:][:, ::-1]
            weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
            return weights, indices

    class SimpleMoE(nn.Module):
        def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_tok
            self.router = SimpleMoERouter(hidden_size, num_experts, num_experts_per_tok)
            self.experts = [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]

        def __call__(self, x: mx.array) -> mx.array:
            batch_size, seq_len, hidden_size = x.shape
            x_flat = x.reshape(-1, hidden_size)
            weights, indices = self.router(x_flat)
            output = mx.zeros_like(x_flat)
            for expert_idx, expert in enumerate(self.experts):
                mask = indices == expert_idx
                expert_weights = mx.sum(weights * mask.astype(weights.dtype), axis=-1)
                if mx.any(expert_weights > 0):
                    output = output + expert(x_flat) * expert_weights[:, None]
            return output.reshape(batch_size, seq_len, hidden_size)

    class SimpleMoELayer(nn.Module):
        def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(hidden_size)
            self.self_attn = nn.MultiHeadAttention(hidden_size, 4)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size)
            self.mlp = SimpleMoE(hidden_size, num_experts, num_experts_per_tok)

        def __call__(self, x: mx.array) -> tuple[mx.array, None]:
            residual = x
            x = self.input_layernorm(x)
            x = self.self_attn(x, x, x)
            x = residual + x
            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x
            return x, None

    class SimpleMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = [
                SimpleMoELayer(hidden_size, num_experts, num_experts_per_tok)
                for _ in range(num_layers)
            ]
            self.norm = nn.RMSNorm(hidden_size)

    class SimpleMoEForCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleMoEModel()
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def __call__(self, input_ids: mx.array) -> mx.array:
            h = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                h, _ = layer(h)
            h = self.model.norm(h)
            return self.lm_head(h)

    return SimpleMoEForCausalLM()


def analyze_routing_patterns(model, input_ids: mx.array, tokenizer=None):
    """Analyze MoE routing patterns in detail."""
    print("\n" + "=" * 70)
    print("MoE ROUTING ANALYSIS")
    print("=" * 70)

    # Detect architecture
    arch = detect_moe_architecture(model)
    print(f"\nArchitecture: {arch.value}")

    # Create hooks
    hooks = MoEHooks(model)
    hooks.configure(
        MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        )
    )

    # Run forward pass
    print(f"\nInput shape: {input_ids.shape}")
    logits = hooks.forward(input_ids)
    mx.eval(logits)

    # Print captured layers
    print(f"\nMoE layers: {hooks.moe_layer_indices}")
    print(f"Captured: {hooks.state.captured_layers}")

    # Analyze each captured layer
    for layer_idx in hooks.state.captured_layers:
        print(f"\n{'─' * 60}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 60}")

        # Expert utilization
        util = hooks.get_expert_utilization(layer_idx)
        if util:
            print("\n  Expert Utilization:")
            print(f"    Load balance score: {util.load_balance_score:.2%}")
            print(f"    Most used expert:   #{util.most_used_expert}")
            print(f"    Least used expert:  #{util.least_used_expert}")

            # Show distribution
            counts = util.token_counts.tolist()
            total = sum(counts)
            print(f"\n    Token distribution across {util.num_experts} experts:")
            for i, count in enumerate(counts):
                pct = count / total * 100 if total > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"      Expert {i:2d}: {bar} {pct:5.1f}% ({int(count)} tokens)")

        # Router entropy
        entropy = hooks.get_router_entropy(layer_idx)
        if entropy:
            print("\n  Router Confidence:")
            print(f"    Mean entropy:       {entropy.mean_entropy:.4f}")
            print(f"    Max entropy:        {entropy.max_entropy:.4f}")
            print(f"    Normalized entropy: {entropy.normalized_entropy:.2%}")
            confidence = 1 - entropy.normalized_entropy
            print(f"    Confidence level:   {confidence:.2%} ", end="")
            if confidence > 0.8:
                print("(very confident)")
            elif confidence > 0.6:
                print("(confident)")
            elif confidence > 0.4:
                print("(moderate)")
            else:
                print("(uncertain)")

        # Show routing for last token
        pattern = hooks.get_routing_pattern(layer_idx, position=-1)
        if pattern:
            print("\n  Last token routing:")
            print(f"    Selected experts: {pattern['selected_experts']}")
            print(f"    Routing weights:  {[f'{w:.3f}' for w in pattern['routing_weights']]}")

    print("\n" + "=" * 70)


def compare_prompts(model, prompts: list[str], tokenizer):
    """Compare routing patterns across different prompts."""
    print("\n" + "=" * 70)
    print("CROSS-PROMPT ROUTING COMPARISON")
    print("=" * 70)

    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig())

    results = []
    for prompt in prompts:
        input_ids = mx.array([tokenizer.encode(prompt)])
        hooks.forward(input_ids)
        mx.eval(hooks.state.router_weights)

        comparison = hooks.compare_routing_across_layers()
        results.append(
            {
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "stats": comparison,
            }
        )

    # Print comparison table
    if results and results[0]["stats"]:
        layers = sorted(results[0]["stats"].keys())

        print(f"\n{'Prompt':<55} | ", end="")
        for layer in layers:
            print(f"L{layer} Entropy | ", end="")
        print()
        print("-" * (55 + len(layers) * 13))

        for result in results:
            print(f"{result['prompt']:<55} | ", end="")
            for layer in layers:
                if layer in result["stats"]:
                    ent = result["stats"][layer]["normalized_entropy"]
                    print(f"   {ent:.2%}    | ", end="")
                else:
                    print("    N/A     | ", end="")
            print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MoE Routing Analysis")
    parser.add_argument("--model", type=str, help="Model path or HF model ID")
    parser.add_argument("--test", action="store_true", help="Use test model")
    parser.add_argument("--prompts", nargs="+", help="Prompts to analyze")
    args = parser.parse_args()

    if args.test or not args.model:
        print("Creating test MoE model...")
        model = create_test_moe_model(
            vocab_size=1000,
            hidden_size=64,
            num_layers=4,
            num_experts=8,
            num_experts_per_tok=2,
        )

        # Create simple tokenizer-like encoding
        input_ids = mx.array([[1, 42, 100, 500, 200, 300, 400, 50]])

        print("\n" + "=" * 70)
        print("TEST MODEL ANALYSIS")
        print("=" * 70)
        print("Model: Test MoE (8 experts, 2 active per token)")
        print("Layers: 4")
        print("Hidden size: 64")

        analyze_routing_patterns(model, input_ids)

    else:
        # Load real model
        print(f"Loading model: {args.model}")

        try:
            from mlx_lm import load
            from transformers import AutoTokenizer

            model, tokenizer = load(args.model)

            prompts = args.prompts or [
                "The capital of France is",
                "def fibonacci(n):",
                "Once upon a time",
            ]

            for prompt in prompts:
                input_ids = mx.array([tokenizer.encode(prompt)])
                analyze_routing_patterns(model, input_ids, tokenizer)

            if len(prompts) > 1:
                compare_prompts(model, prompts, tokenizer)

        except ImportError as e:
            print(f"Error: {e}")
            print("Install mlx-lm and transformers to load real models")
            print("Falling back to test model...")
            model = create_test_moe_model()
            input_ids = mx.array([[1, 42, 100, 500, 200, 300, 400, 50]])
            analyze_routing_patterns(model, input_ids)


if __name__ == "__main__":
    main()
