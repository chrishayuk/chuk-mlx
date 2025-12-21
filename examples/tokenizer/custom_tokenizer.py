"""
Custom Tokenizer Example

Shows how to create and use a custom character-level tokenizer.
"""

from chuk_lazarus.data.tokenizers import CustomTokenizer


def main():
    # Create custom tokenizer from text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is transforming the world.
    """

    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(sample_text, min_freq=1)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: pad={tokenizer.pad_token_id}, unk={tokenizer.unk_token_id}")

    # Tokenize
    text = "The fox is quick."
    tokens = tokenizer.encode(text)
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")

    # Decode
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Save and load
    tokenizer.save("./my_tokenizer")
    loaded = CustomTokenizer.load("./my_tokenizer")
    print(f"\nLoaded tokenizer vocab size: {loaded.vocab_size}")


if __name__ == "__main__":
    main()
