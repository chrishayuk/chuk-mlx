"""
Probe GPT-OSS routing patterns.

GPT-OSS shows different suffix routing than TinyLlama:
- TinyLlama: 15 > 10 = → 5 (subtraction)
- GPT-OSS: 15 > 10 = → 0 (???)

Let's probe to understand what pattern GPT-OSS learned.
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


def get_top_predictions(model, tokenizer, text, n=5):
    """Get top-n predictions."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    output = model(input_ids)
    logits = output.logits if hasattr(output, "logits") else output
    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    top_idx = mx.argsort(probs)[-n:][::-1].tolist()
    return [(tokenizer.decode([i]).strip() or repr(tokenizer.decode([i])), float(probs[i].item())) for i in top_idx]


def main():
    print("Loading GPT-OSS 20B...")
    result = load_model("openai/gpt-oss-20b")
    model, tokenizer = result.model, result.tokenizer

    print("\n" + "=" * 60)
    print("  GPT-OSS Routing Analysis")
    print("=" * 60)

    # Test 1: Basic arithmetic patterns
    print("\n[1] Basic Arithmetic - What does GPT-OSS think '=' means?")
    print("-" * 55)

    tests = [
        "2 + 2 = ",
        "5 + 3 = ",
        "10 - 7 = ",
        "15 - 10 = ",
    ]

    for text in tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        print(f"{text:<15} → {top[0][0]:<5} (conf: {top[0][1]:.2%})")

    # Test 2: Comparison patterns
    print("\n[2] Comparisons - How does GPT-OSS interpret '>' with '='?")
    print("-" * 55)

    tests = [
        "15 > 10 = ",
        "10 > 15 = ",
        "5 > 5 = ",
        "True = ",
        "False = ",
    ]

    for text in tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        others = ", ".join([f"{t[0]}" for t in top[1:3]])
        print(f"{text:<15} → {top[0][0]:<5} (also: {others})")

    # Test 3: What does 'is' trigger?
    print("\n[3] Boolean suffix - 'is' patterns")
    print("-" * 55)

    tests = [
        "15 > 10 is ",
        "10 > 15 is ",
        "True is ",
        "False is ",
    ]

    for text in tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        others = ", ".join([f"{t[0]}" for t in top[1:3]])
        print(f"{text:<15} → {top[0][0]:<5} (also: {others})")

    # Test 4: Direct boolean
    print("\n[4] Direct boolean outputs")
    print("-" * 55)

    tests = [
        "Is 15 greater than 10? ",
        "Is 10 greater than 15? ",
        "15 > 10: ",
        "10 > 15: ",
    ]

    for text in tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        others = ", ".join([f"{t[0]}" for t in top[1:3]])
        print(f"{text:<30} → {top[0][0]:<5} (also: {others})")

    # Test 5: The garbage test
    print("\n[5] Garbage input - Does suffix alone control output?")
    print("-" * 55)

    tests = [
        "xyz = ",
        "??? = ",
        "abc def = ",
        "= ",
    ]

    for text in tests:
        top = get_top_predictions(model, tokenizer, text, 3)
        others = ", ".join([f"{t[0]}" for t in top[1:3]])
        print(f"{text:<15} → {top[0][0]:<5} (also: {others})")


if __name__ == "__main__":
    main()
