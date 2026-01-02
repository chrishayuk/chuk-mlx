#!/usr/bin/env python3
"""
Attention Pattern Analysis - Which Heads Drive Tool-Calling?

Compare attention patterns between a base model and fine-tuned model
to identify which attention heads are responsible for divergence.

Run: uv run python examples/introspection/attention_divergence.py
     uv run python examples/introspection/attention_divergence.py --base model1 --ft model2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from _loader import format_tool_prompt, load_chat_template, load_model

# Default model pairs
DEFAULT_PAIRS = {
    "gemma": {
        "base": "mlx-community/gemma-3-270m-it-bf16",
        "ft": "mlx-community/functiongemma-270m-it-bf16",
    },
}


@dataclass
class HeadDivergence:
    """Divergence metrics for a specific attention head."""
    layer: int
    head: int
    prompt_type: str
    js_divergence: float
    cosine_similarity: float


def compute_attention_weights(model, input_ids: mx.array, layer_idx: int) -> mx.array:
    """Compute attention weights for a specific layer."""
    config = model.config
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    # Get embeddings
    h = model.model.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size**0.5, dtype=h.dtype)

    # Process through layers up to target
    for i in range(layer_idx):
        block_out = model.model.layers[i](h, mask=None, cache=None)
        h = block_out.hidden_states

    # Apply input norm
    h_normed = layer.input_layernorm(h)
    batch_size, seq_len, _ = h_normed.shape

    # Project Q, K
    queries = attn.q_proj(h_normed)
    keys = attn.k_proj(h_normed)

    # Reshape
    queries = queries.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    keys = keys.transpose(0, 2, 1, 3)

    # Apply Q/K norms if available
    if hasattr(attn, 'q_norm'):
        queries = attn.q_norm(queries)
    if hasattr(attn, 'k_norm'):
        keys = attn.k_norm(keys)

    # Apply RoPE
    queries = attn.rope(queries)
    keys = attn.rope(keys)

    # Repeat KV heads for GQA
    if hasattr(attn, 'n_rep') and attn.n_rep > 1:
        keys = mx.repeat(keys, attn.n_rep, axis=1)

    # Compute attention scores
    scale = attn.scale
    scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    scores = scores + causal_mask.astype(scores.dtype)

    return mx.softmax(scores, axis=-1)


def js_divergence(p: mx.array, q: mx.array, eps: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence."""
    p = mx.clip(p, eps, 1.0)
    q = mx.clip(q, eps, 1.0)
    p = p / mx.sum(p, axis=-1, keepdims=True)
    q = q / mx.sum(q, axis=-1, keepdims=True)
    m = (p + q) / 2
    kl_pm = mx.sum(p * mx.log(p / m), axis=-1)
    kl_qm = mx.sum(q * mx.log(q / m), axis=-1)
    return float(mx.mean((kl_pm + kl_qm) / 2))


def compare_attention_heads(
    base_model,
    ft_model,
    tokenizer,
    prompt: str,
    prompt_type: str,
    layers_to_analyze: list[int],
) -> list[HeadDivergence]:
    """Compare attention patterns between models."""
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    divergences = []
    num_heads = base_model.config.num_attention_heads

    for layer_idx in layers_to_analyze:
        base_attn = compute_attention_weights(base_model, input_ids, layer_idx)
        ft_attn = compute_attention_weights(ft_model, input_ids, layer_idx)

        for head_idx in range(num_heads):
            base_head = base_attn[0, head_idx, -1, :]
            ft_head = ft_attn[0, head_idx, -1, :]

            js_div = js_divergence(base_head[None, :], ft_head[None, :])

            dot = float(mx.sum(base_head * ft_head))
            norm_b = float(mx.sqrt(mx.sum(base_head * base_head)))
            norm_f = float(mx.sqrt(mx.sum(ft_head * ft_head)))
            cos_sim = dot / (norm_b * norm_f + 1e-8)

            divergences.append(HeadDivergence(
                layer=layer_idx,
                head=head_idx,
                prompt_type=prompt_type,
                js_divergence=js_div,
                cosine_similarity=cos_sim,
            ))

    return divergences


def main():
    parser = argparse.ArgumentParser(description="Attention Pattern Analysis")
    parser.add_argument("--base", default=None, help="Base model ID")
    parser.add_argument("--ft", default=None, help="Fine-tuned model ID")
    parser.add_argument("--pair", choices=list(DEFAULT_PAIRS.keys()), default="gemma")
    args = parser.parse_args()

    if args.base and args.ft:
        base_id, ft_id = args.base, args.ft
    else:
        pair = DEFAULT_PAIRS[args.pair]
        base_id, ft_id = pair["base"], pair["ft"]

    print("=" * 80)
    print("Attention Pattern Analysis")
    print(f"Base: {base_id}")
    print(f"Fine-tuned: {ft_id}")
    print("=" * 80)

    base_model, _, base_config, _ = load_model(base_id)
    ft_model, ft_tokenizer, ft_config, _ = load_model(ft_id)
    template = load_chat_template(ft_id)

    num_layers = ft_config.num_hidden_layers
    layers_to_analyze = [0, int(num_layers * 0.5), int(num_layers * 0.6),
                         int(num_layers * 0.8), num_layers - 2, num_layers - 1]
    layers_to_analyze = sorted(set(layers_to_analyze))

    print(f"\nAnalyzing layers: {layers_to_analyze}")

    # Test prompts
    test_prompts = {
        "neutral": ["The capital of France is", "Once upon a time"],
        "tool_explicit": [],
    }

    if template:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            },
        }]
        for query in ["What is the weather in Tokyo?", "Tell me the weather in London"]:
            test_prompts["tool_explicit"].append(format_tool_prompt(template, query, tools))

    # Compute divergences
    all_divergences = []
    for prompt_type, prompts in test_prompts.items():
        print(f"\nProcessing {prompt_type} ({len(prompts)} prompts)...")
        for prompt in prompts:
            divs = compare_attention_heads(base_model, ft_model, ft_tokenizer, prompt, prompt_type, layers_to_analyze)
            all_divergences.extend(divs)

    # Summary
    print("\n" + "=" * 80)
    print("ATTENTION HEAD DIVERGENCE SUMMARY")
    print("=" * 80)

    for prompt_type in test_prompts.keys():
        pt_divs = [d for d in all_divergences if d.prompt_type == prompt_type]
        if not pt_divs:
            continue

        print(f"\n{prompt_type}:")
        print(f"{'Layer':<8} {'Head':<8} {'JS Div':>12} {'Cos Sim':>12}")
        print("-" * 45)

        # Group by layer
        for layer in layers_to_analyze:
            layer_divs = [d for d in pt_divs if d.layer == layer]
            for d in sorted(layer_divs, key=lambda x: x.js_divergence, reverse=True)[:3]:
                marker = " ***" if d.js_divergence > 0.1 else ""
                print(f"{d.layer:<8} {d.head:<8} {d.js_divergence:>12.6f} {d.cosine_similarity:>12.4f}{marker}")

    # Save results
    output_path = Path("attention_divergence_results.json")
    results = [{"layer": d.layer, "head": d.head, "prompt_type": d.prompt_type,
                "js_divergence": d.js_divergence, "cosine_similarity": d.cosine_similarity}
               for d in all_divergences]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    del base_model, ft_model
    mx.metal.clear_cache()


if __name__ == "__main__":
    main()
