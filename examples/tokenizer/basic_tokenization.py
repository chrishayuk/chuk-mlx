"""
Basic Tokenization Example

Shows how to load a tokenizer and tokenize text.
"""

from chuk_lazarus.models import load_tokenizer


def main():
    # Load tokenizer from HuggingFace model
    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Tokenize text
    text = "Hello, how are you today?"
    tokens = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Decode back
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Batch tokenization
    texts = [
        "First sentence.",
        "Second sentence is longer.",
        "Third.",
    ]
    for t in texts:
        toks = tokenizer.encode(t)
        print(f"{t!r} -> {len(toks)} tokens")


if __name__ == "__main__":
    main()
