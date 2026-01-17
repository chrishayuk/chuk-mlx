"""
Probe which layer(s) encode comparison semantics.

For arithmetic, layer 12 (~55% depth) was the decision layer.
Comparisons might emerge at a different layer.

This script sweeps all layers and checks logit lens probabilities
for comparison-related tokens.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from chuk_lazarus.models_v2.loader import load_model


def probe_comparison_layers():
    """Sweep layers to find where comparison semantics emerge."""

    print("Loading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = result.model
    tokenizer = result.tokenizer
    config = result.config

    num_layers = config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Candidate tokens for comparison outputs
    # We'll check what the model "wants to say" at each layer
    candidate_tokens = {
        # Boolean-ish
        "true": tokenizer.encode("true")[-1],
        "false": tokenizer.encode("false")[-1],
        "yes": tokenizer.encode("yes")[-1],
        "no": tokenizer.encode("no")[-1],
        "True": tokenizer.encode("True")[-1],
        "False": tokenizer.encode("False")[-1],
        # Numeric boolean
        "1": tokenizer.encode("1")[-1],
        "0": tokenizer.encode("0")[-1],
        # Comparison words (might be intermediate representations)
        "greater": tokenizer.encode("greater")[-1],
        "less": tokenizer.encode("less")[-1],
        "equal": tokenizer.encode("equal")[-1],
        "same": tokenizer.encode("same")[-1],
        "different": tokenizer.encode("different")[-1],
    }

    print(f"\nCandidate token IDs: {candidate_tokens}")

    # Test inputs - canonical comparison format
    test_cases = [
        ("15 > 10 = ", True, "greater"),
        ("5 < 10 = ", True, "less"),
        ("7 == 7 = ", True, "equal"),
        ("8 != 3 = ", True, "not_equal"),
        ("3 > 10 = ", False, "greater"),
        ("15 < 10 = ", False, "less"),
        ("7 == 8 = ", False, "equal"),
        ("3 != 3 = ", False, "not_equal"),
    ]

    # Also test natural language variants
    nl_test_cases = [
        ("Is 15 greater than 10?", True, "greater_nl"),
        ("Is 5 less than 10?", True, "less_nl"),
        ("Does 7 equal 7?", True, "equal_nl"),
        ("Is 3 bigger than 10?", False, "greater_nl"),
    ]

    all_tests = test_cases + nl_test_cases

    backbone = model.model

    print("\n" + "=" * 80)
    print("LAYER SWEEP: Where do comparison semantics emerge?")
    print("=" * 80)

    # For each test case, sweep layers and record top predictions
    for test_input, expected_true, comp_type in all_tests:
        print(f"\n{'─' * 60}")
        print(f"Input: {test_input!r}")
        print(f"Expected: {'true' if expected_true else 'false'} ({comp_type})")
        print(f"{'─' * 60}")

        tokens = tokenizer.encode(test_input)
        input_ids = mx.array([tokens])

        # Forward through all layers, collecting hidden states
        h = backbone.embed_tokens(input_ids)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
        mask = mask.astype(h.dtype)

        layer_results = []

        for i, layer in enumerate(backbone.layers):
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output

            # Logit lens: project to vocab space
            h_normed = backbone.norm(h)
            logits = model.lm_head(h_normed)
            if hasattr(logits, "logits"):
                logits = logits.logits

            probs = mx.softmax(logits[0, -1, :])
            mx.eval(probs)

            # Get probabilities for candidate tokens
            token_probs = {}
            for name, tid in candidate_tokens.items():
                token_probs[name] = float(probs[tid].item())

            # Find top prediction among candidates
            top_token = max(token_probs, key=token_probs.get)
            top_prob = token_probs[top_token]

            # Also get global top-5
            top5_indices = mx.argsort(probs)[-5:][::-1].tolist()
            top5 = [(tokenizer.decode([idx]), float(probs[idx].item())) for idx in top5_indices]

            layer_results.append({
                "layer": i,
                "depth_pct": (i + 1) / num_layers,
                "top_candidate": top_token,
                "top_prob": top_prob,
                "true_prob": token_probs.get("true", 0) + token_probs.get("True", 0) + token_probs.get("1", 0),
                "false_prob": token_probs.get("false", 0) + token_probs.get("False", 0) + token_probs.get("0", 0),
                "top5_global": top5,
            })

        # Print summary - find layers where true/false separation is strongest
        print(f"\nLayer | Depth | Top Candidate | True-ish | False-ish | Separation")
        print("-" * 75)

        for r in layer_results:
            sep = r["true_prob"] - r["false_prob"] if expected_true else r["false_prob"] - r["true_prob"]
            correct = "+" if sep > 0 else "-"
            print(f"  {r['layer']:2d}  | {r['depth_pct']:4.0%}  | {r['top_candidate']:12s} | "
                  f"{r['true_prob']:6.3f}   | {r['false_prob']:6.3f}    | {sep:+6.3f} {correct}")

        # Find best layer for this comparison
        if expected_true:
            best_layer = max(layer_results, key=lambda r: r["true_prob"] - r["false_prob"])
        else:
            best_layer = max(layer_results, key=lambda r: r["false_prob"] - r["true_prob"])

        print(f"\nBest layer: {best_layer['layer']} ({best_layer['depth_pct']:.0%} depth)")
        print(f"Global top-5 at best layer: {best_layer['top5_global']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nLook for:")
    print("1. Consistent layer(s) where true/false separation is strongest")
    print("2. Whether canonical (15 > 10 =) vs NL (Is 15 greater than 10?) differ")
    print("3. Which tokens the model prefers (true/false vs 1/0 vs yes/no)")


if __name__ == "__main__":
    probe_comparison_layers()
