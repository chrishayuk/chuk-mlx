#!/usr/bin/env python3
"""
Attention Head Analysis for Fact Retrieval

Research question: Which attention heads are responsible for key-value lookup
in multiplication fact retrieval?

We use ablation to find heads that, when removed, break retrieval
while preserving other model capabilities.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class HeadAblationResult:
    """Result of ablating a single attention head."""

    layer: int
    head: int
    original_answer: str
    original_prob: float
    ablated_answer: str
    ablated_prob: float
    impact: float  # How much the answer changed


def analyze_attention_heads(
    model_id: str = "openai/gpt-oss-20b",
    test_queries: list[str] | None = None,
    target_layers: list[int] | None = None,
) -> dict[str, list[HeadAblationResult]]:
    """
    Find which attention heads are causal for fact retrieval.

    Uses ablation: zero out each head's contribution and measure impact.
    """
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    print(f"  Layers: {num_layers}")
    print(f"  Heads per layer: {num_heads}")
    print(f"  Head dimension: {head_dim}")

    if test_queries is None:
        test_queries = ["7*8=", "3*4=", "6*7=", "9*9="]

    if target_layers is None:
        # Focus on retrieval layers (based on our earlier analysis)
        target_layers = [18, 19, 20, 21, 22, 23]

    def get_layers():
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return list(model.layers)

    def get_embed():
        if hasattr(model, "model"):
            return model.model.embed_tokens
        return model.embed_tokens

    def get_norm():
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        return model.norm

    def get_lm_head():
        return model.lm_head if hasattr(model, "lm_head") else None

    def get_scale():
        return getattr(config, "embedding_scale", None)

    def forward_with_head_ablation(
        prompt: str,
        ablate_layer: int | None = None,
        ablate_head: int | None = None,
    ) -> tuple[str, float]:
        """Forward pass with optional head ablation."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = get_layers()
        embed = get_embed()
        norm = get_norm()
        lm_head = get_lm_head()
        scale = get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            if idx == ablate_layer and ablate_head is not None:
                # Manual forward with head ablation
                # This is model-specific; for GPT-OSS style:
                attn = lyr.self_attn if hasattr(lyr, "self_attn") else lyr.attention

                # Get pre-norm hidden state
                if hasattr(lyr, "input_layernorm"):
                    h_norm = lyr.input_layernorm(h)
                elif hasattr(lyr, "ln_1"):
                    h_norm = lyr.ln_1(h)
                else:
                    h_norm = h

                # Compute attention with head ablation
                B, L, D = h_norm.shape

                # Get Q, K, V projections
                if hasattr(attn, "q_proj"):
                    q = attn.q_proj(h_norm)
                    k = attn.k_proj(h_norm)
                    v = attn.v_proj(h_norm)
                elif hasattr(attn, "qkv_proj"):
                    qkv = attn.qkv_proj(h_norm)
                    q, k, v = mx.split(qkv, 3, axis=-1)
                else:
                    # Fall back to normal forward
                    try:
                        out = lyr(h, mask=mask)
                    except TypeError:
                        out = lyr(h)
                    h = (
                        out.hidden_states
                        if hasattr(out, "hidden_states")
                        else (out[0] if isinstance(out, tuple) else out)
                    )
                    continue

                # Reshape for multi-head attention
                q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)

                # Compute attention scores
                scale_factor = head_dim**-0.5
                scores = (q @ k.transpose(0, 1, 3, 2)) * scale_factor

                # Apply causal mask
                if mask is not None:
                    scores = scores + mask

                attn_weights = mx.softmax(scores, axis=-1)

                # Apply attention
                attn_output = attn_weights @ v

                # ABLATE: Zero out the specified head
                attn_output = mx.array(attn_output)  # Copy
                # attn_output[:, ablate_head, :, :] = 0.0  # This doesn't work in MLX

                # Instead, create a mask
                head_mask = mx.ones((num_heads,))
                head_mask = mx.concatenate(
                    [head_mask[:ablate_head], mx.zeros((1,)), head_mask[ablate_head + 1 :]]
                )
                head_mask = head_mask.reshape(1, num_heads, 1, 1)
                attn_output = attn_output * head_mask

                # Reshape back
                attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, D)

                # Output projection
                if hasattr(attn, "o_proj"):
                    attn_output = attn.o_proj(attn_output)
                elif hasattr(attn, "out_proj"):
                    attn_output = attn.out_proj(attn_output)

                # Residual
                h = h + attn_output

                # MLP
                if hasattr(lyr, "post_attention_layernorm"):
                    h_mlp = lyr.post_attention_layernorm(h)
                elif hasattr(lyr, "ln_2"):
                    h_mlp = lyr.ln_2(h)
                else:
                    h_mlp = h

                if hasattr(lyr, "mlp"):
                    mlp_out = lyr.mlp(h_mlp)
                elif hasattr(lyr, "feed_forward"):
                    mlp_out = lyr.feed_forward(h_mlp)
                else:
                    mlp_out = h_mlp

                h = h + mlp_out
            else:
                # Normal forward
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )

        # Final prediction
        if norm is not None:
            h = norm(h)
        if lm_head is not None:
            outputs = lm_head(h)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            logits = h @ embed.weight.T

        probs = mx.softmax(logits[0, -1, :], axis=-1)
        top_idx = mx.argmax(probs).item()
        top_prob = float(probs[top_idx])
        top_token = tokenizer.decode([top_idx])

        return top_token, top_prob

    results = {}

    for query in test_queries:
        print(f"\nAnalyzing: {query}")

        # Get baseline
        baseline_token, baseline_prob = forward_with_head_ablation(query)
        print(f"  Baseline: '{baseline_token}' ({baseline_prob:.3f})")

        query_results = []

        for layer in target_layers:
            print(f"  Layer {layer}:", end=" ")
            layer_impacts = []

            for head in range(num_heads):
                ablated_token, ablated_prob = forward_with_head_ablation(
                    query, ablate_layer=layer, ablate_head=head
                )

                # Measure impact
                if ablated_token != baseline_token:
                    impact = 1.0  # Complete change
                else:
                    impact = baseline_prob - ablated_prob  # Confidence drop

                if impact > 0.1:  # Significant impact
                    layer_impacts.append((head, impact, ablated_token, ablated_prob))

                query_results.append(
                    HeadAblationResult(
                        layer=layer,
                        head=head,
                        original_answer=baseline_token,
                        original_prob=baseline_prob,
                        ablated_answer=ablated_token,
                        ablated_prob=ablated_prob,
                        impact=impact,
                    )
                )

            if layer_impacts:
                print(f"high-impact heads: {[(h, f'{i:.2f}') for h, i, _, _ in layer_impacts[:5]]}")
            else:
                print("no high-impact heads")

        results[query] = query_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Most Important Heads for Fact Retrieval")
    print("=" * 70)

    # Aggregate across queries
    head_importance = defaultdict(list)
    for query, query_results in results.items():
        for r in query_results:
            if r.impact > 0.1:
                head_importance[(r.layer, r.head)].append(r.impact)

    # Sort by average impact
    sorted_heads = sorted(head_importance.items(), key=lambda x: -np.mean(x[1]))[:20]

    print("\nTop 20 most important heads (across all test queries):")
    for (layer, head), impacts in sorted_heads:
        avg_impact = np.mean(impacts)
        count = len(impacts)
        print(
            f"  Layer {layer:2d}, Head {head:2d}: avg_impact={avg_impact:.3f}, affects {count}/{len(test_queries)} queries"
        )

    return results


def main():
    results = analyze_attention_heads(
        model_id="openai/gpt-oss-20b",
        test_queries=["7*8=", "3*4=", "6*7=", "9*9=", "2*5="],
        target_layers=[18, 19, 20, 21, 22, 23],
    )

    # Save results
    output = {}
    for query, query_results in results.items():
        output[query] = [
            {
                "layer": r.layer,
                "head": r.head,
                "original": r.original_answer,
                "original_prob": r.original_prob,
                "ablated": r.ablated_answer,
                "ablated_prob": r.ablated_prob,
                "impact": r.impact,
            }
            for r in query_results
            if r.impact > 0.05
        ]

    with open("attention_head_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to attention_head_results.json")


if __name__ == "__main__":
    main()
