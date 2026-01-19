"""
Attention Pattern Analysis for Structural Classification

Hypothesis: Attention patterns encode structural relationships that are
more vocabulary-agnostic than hidden states.

Previous MoE research showed:
- 89-98% of routing signal comes from attention at middle-to-late layers
- Same token routes differently based on context (78% sensitivity)
- Attention patterns correlate with expert selection

This experiment tests:
1. Do similar-structure programs (sum_even, collatz) have similar attention patterns?
2. Can we classify templates using attention features instead of hidden states?
3. Does this enable zero-shot generalization (no vocab transfer)?
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

import mlx.core as mx
import mlx.nn as nn

from templates import TemplateID, template_name, PROGRAM_TO_TEMPLATE


# =============================================================================
# ATTENTION EXTRACTION
# =============================================================================

def extract_attention_patterns(
    model, tokenizer, text: str, layers: List[int] = None
) -> Dict[int, mx.array]:
    """
    Extract attention patterns from specified layers.

    Returns: {layer_idx: attention_weights} where attention_weights is
    (num_heads, seq_len, seq_len)
    """
    if layers is None:
        layers = [10, 11, 12, 13, 14, 15]  # Middle-to-late layers

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    # Create causal mask (use None to let model handle it internally)
    # We'll compute attention manually without going through the full forward

    # Forward through embedding
    hidden = model.model.embed_tokens(input_ids)

    # Create mask in the right dtype
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(hidden.dtype)

    attention_patterns = {}

    for i, block in enumerate(model.model.layers):
        # Get attention weights if this is a layer we care about
        if i in layers:
            # Extract Q, K from the attention layer
            attn = block.self_attn

            # Compute Q, K (handle Grouped Query Attention)
            q = attn.q_proj(hidden)
            k = attn.k_proj(hidden)

            # Get dimensions
            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads
            head_dim = q.shape[-1] // num_heads

            # Reshape Q: (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
            q = q.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

            # Reshape K: (batch, seq, num_kv_heads * head_dim) -> (batch, num_kv_heads, seq, head_dim)
            kv_head_dim = k.shape[-1] // num_kv_heads
            k = k.reshape(1, seq_len, num_kv_heads, kv_head_dim).transpose(0, 2, 1, 3)

            # Repeat KV heads to match Q heads (GQA)
            # Each KV head serves (num_heads // num_kv_heads) Q heads
            repeat_factor = num_heads // num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)

            # Compute attention scores
            scale = math.sqrt(head_dim)
            scores = (q @ k.transpose(0, 1, 3, 2)) / scale

            # Apply causal mask
            if mask is not None:
                scores = scores + mask

            # Softmax to get attention weights
            attn_weights = mx.softmax(scores, axis=-1)

            attention_patterns[i] = attn_weights[0]  # Remove batch dim

        # Forward through the block
        hidden = block(hidden, mask=mask, cache=None)
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    return attention_patterns


def compute_attention_features(attn_patterns: Dict[int, mx.array]) -> mx.array:
    """
    Compute aggregate features from attention patterns.

    Features:
    - Mean attention entropy per head (how focused/distributed)
    - Attention to first token (BOS pattern)
    - Attention to last token (recent context)
    - Self-attention (diagonal)
    - Cross-position attention (off-diagonal patterns)
    """
    features = []

    for layer_idx, attn in sorted(attn_patterns.items()):
        num_heads, seq_len, _ = attn.shape

        # Skip if sequence too short
        if seq_len < 2:
            continue

        for head_idx in range(num_heads):
            head_attn = attn[head_idx]  # (seq_len, seq_len)

            # 1. Attention entropy (how spread out the attention is)
            # Higher entropy = more distributed attention
            # Lower entropy = more focused attention
            eps = 1e-10
            entropy = -mx.sum(head_attn * mx.log(head_attn + eps), axis=-1)
            mean_entropy = mx.mean(entropy).item()
            features.append(mean_entropy)

            # 2. Attention to first token (BOS/start pattern)
            attn_to_first = mx.mean(head_attn[:, 0]).item()
            features.append(attn_to_first)

            # 3. Attention to last token
            attn_to_last = mx.mean(head_attn[:, -1]).item()
            features.append(attn_to_last)

            # 4. Self-attention (diagonal average)
            diag_indices = min(seq_len, seq_len)
            self_attn = mx.mean(mx.array([head_attn[i, i].item() for i in range(diag_indices)])).item()
            features.append(self_attn)

            # 5. Local attention (attention to nearby tokens)
            local_attn = 0.0
            count = 0
            for i in range(seq_len):
                for j in range(max(0, i-2), min(seq_len, i+3)):
                    local_attn += head_attn[i, j].item()
                    count += 1
            local_attn = local_attn / count if count > 0 else 0
            features.append(local_attn)

    return mx.array(features)


# =============================================================================
# ANALYSIS
# =============================================================================

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two vectors."""
    dot = mx.sum(a * b).item()
    norm_a = mx.sqrt(mx.sum(a * a)).item()
    norm_b = mx.sqrt(mx.sum(b * b)).item()
    return dot / (norm_a * norm_b + 1e-10)


def analyze_attention_similarity(model, tokenizer, examples: List[dict]) -> dict:
    """
    Analyze attention pattern similarity across templates.
    """
    print("Extracting attention features...")

    # Group examples by template
    by_template = {}
    for ex in examples:
        tid = ex["template_id"]
        if tid not in by_template:
            by_template[tid] = []
        by_template[tid].append(ex)

    # Extract features for a sample of each template
    features_by_template = {}
    for tid, exs in by_template.items():
        sample = random.sample(exs, min(10, len(exs)))
        template_features = []

        for ex in sample:
            attn_patterns = extract_attention_patterns(model, tokenizer, ex["nl_input"])
            features = compute_attention_features(attn_patterns)
            template_features.append(features)

        features_by_template[tid] = template_features
        print(f"  Template {template_name(TemplateID(tid))}: {len(template_features)} examples, {len(template_features[0])} features")

    # Compute within-template and cross-template similarity
    print("\nComputing similarities...")

    results = {
        "within_template": {},
        "cross_template": {},
    }

    template_ids = sorted(features_by_template.keys())

    for tid in template_ids:
        features = features_by_template[tid]

        # Within-template similarity (should be high)
        within_sims = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = cosine_similarity(features[i], features[j])
                within_sims.append(sim)

        results["within_template"][tid] = {
            "mean": sum(within_sims) / len(within_sims) if within_sims else 0,
            "min": min(within_sims) if within_sims else 0,
            "max": max(within_sims) if within_sims else 0,
        }

    # Cross-template similarity
    for i, tid1 in enumerate(template_ids):
        for tid2 in template_ids[i+1:]:
            cross_sims = []
            for f1 in features_by_template[tid1]:
                for f2 in features_by_template[tid2]:
                    sim = cosine_similarity(f1, f2)
                    cross_sims.append(sim)

            key = f"{template_name(TemplateID(tid1))} vs {template_name(TemplateID(tid2))}"
            results["cross_template"][key] = {
                "mean": sum(cross_sims) / len(cross_sims) if cross_sims else 0,
                "min": min(cross_sims) if cross_sims else 0,
                "max": max(cross_sims) if cross_sims else 0,
            }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Attention Pattern Analysis for Structural Classification")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()
    mx.eval(model.parameters())
    print("   Model loaded.")

    # Load training data (has all templates)
    print("\n2. Loading training data...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")

    # Add some test examples for analysis
    all_examples = train_examples + test_examples[:20]
    print(f"   Loaded {len(all_examples)} examples")

    # Analyze attention similarity
    print("\n3. Analyzing attention patterns...")
    results = analyze_attention_similarity(model, tokenizer, all_examples)

    # Display results
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN SIMILARITY")
    print("=" * 70)

    print("\nWithin-Template Similarity (higher = more consistent):")
    print("-" * 50)
    for tid, stats in results["within_template"].items():
        name = template_name(TemplateID(tid))
        print(f"  {name:35} mean={stats['mean']:.3f} (min={stats['min']:.3f}, max={stats['max']:.3f})")

    print("\nCross-Template Similarity (lower = more distinguishable):")
    print("-" * 50)
    for key, stats in results["cross_template"].items():
        print(f"  {key:50} mean={stats['mean']:.3f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
If within-template similarity > cross-template similarity:
  → Attention patterns ARE structure-aware
  → Classification based on attention should work

If similarities are similar:
  → Attention patterns encode other features (vocabulary, position)
  → Need different approach for vocabulary-agnostic classification
""")

    # Quick test: sum_even vs collatz
    print("\n4. Specific comparison: sum_even vs collatz_length")
    print("-" * 50)

    sum_even_ex = [ex for ex in train_examples if ex["program_name"] == "sum_even"][:5]
    collatz_ex = test_examples[:5]

    print("   Extracting features...")
    sum_even_features = []
    for ex in sum_even_ex:
        attn = extract_attention_patterns(model, tokenizer, ex["nl_input"])
        features = compute_attention_features(attn)
        sum_even_features.append(features)

    collatz_features = []
    for ex in collatz_ex:
        attn = extract_attention_patterns(model, tokenizer, ex["nl_input"])
        features = compute_attention_features(attn)
        collatz_features.append(features)

    # Compare
    same_template_sims = []
    for i in range(len(sum_even_features)):
        for j in range(len(collatz_features)):
            sim = cosine_similarity(sum_even_features[i], collatz_features[j])
            same_template_sims.append(sim)

    print(f"\n   sum_even vs collatz_length (same template!):")
    print(f"   Mean similarity: {sum(same_template_sims) / len(same_template_sims):.3f}")
    print(f"   Range: [{min(same_template_sims):.3f}, {max(same_template_sims):.3f}]")

    # Save results
    results_path = results_dir / "attention_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Saved to: {results_path}")


if __name__ == "__main__":
    main()
