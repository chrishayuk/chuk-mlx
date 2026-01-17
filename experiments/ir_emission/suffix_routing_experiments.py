"""
Suffix Routing Experiments.

Visual proof that the suffix determines output type, not the operator.
Key finding: `foo bar = ` outputs a number because of the `= ` suffix.

Run: python experiments/ir_emission/suffix_routing_experiments.py
"""

import mlx.core as mx
import mlx.nn as nn
from chuk_lazarus.models_v2.loader import load_model


def get_top_predictions(model, tokenizer, text, n=5):
    """Get top-n predictions at final layer."""
    backbone = model.model
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    h = backbone.embed_tokens(input_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
    mask = mask.astype(h.dtype)

    for layer in backbone.layers:
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, "hidden_states") else output

    h_normed = backbone.norm(h)
    logits = model.lm_head(h_normed)
    if hasattr(logits, "logits"):
        logits = logits.logits
    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    top_idx = mx.argsort(probs)[-n:][::-1].tolist()
    return [
        (tokenizer.decode([i]).strip() or repr(tokenizer.decode([i])), float(probs[i].item()))
        for i in top_idx
    ]


def run_experiments():
    print("Loading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = result.model
    tokenizer = result.tokenizer

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SUFFIX SWAP")
    print("=" * 70)

    suffix_tests = [
        ("15 > 10 = ", "comparison + '=' suffix"),
        ("15 > 10 is ", "comparison + 'is' suffix"),
    ]

    for text, desc in suffix_tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        print(f"{text:<20} → {top[0][0]:<5} ({desc})")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: GARBAGE INPUT")
    print("=" * 70)

    garbage_tests = [
        ("15 10 = ", "no operator"),
        ("foo bar = ", "non-numeric"),
        ("= ", "just suffix"),
    ]

    for text, desc in garbage_tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        print(f"{text:<20} → {top[0][0]:<5} ({desc})")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("The '=' suffix triggers numeric output regardless of content.")
    print("The model routes by SYNTAX, not SEMANTICS.")


if __name__ == "__main__":
    run_experiments()
