#!/usr/bin/env python3
"""
Tokenizer instrumentation demo.

Demonstrates the instrumentation module for analyzing tokenization behavior
without modifying the tokenizer itself. Useful for:
- Training decisions (batch sizes, sequence lengths)
- Comparing tokenizers before/after vocabulary changes
- Identifying tokenization inefficiencies
"""

from chuk_lazarus.data.tokenizers.instrumentation import (
    # Histogram tools
    compute_length_histogram,
    format_histogram_ascii,
    get_length_stats,
    # OOV analysis
    analyze_oov,
    find_rare_tokens,
    get_frequency_bands,
    # Waste metrics
    analyze_waste,
    analyze_padding_waste,
    analyze_truncation_loss,
    # Vocab comparison
    compare_vocab_impact,
    estimate_retokenization_cost,
)


class SimpleTokenizer:
    """Simple word-level tokenizer for demonstration."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self._next_id = 4

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.lower().split()
        ids = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self._next_id
                self._next_id += 1
            ids.append(self.vocab[word])

        if add_special_tokens:
            ids = [2] + ids + [3]  # <s> ... </s>
        return ids

    def decode(self, token_ids: list[int]) -> str:
        id_to_word = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_word.get(i, "<unk>") for i in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1


class CharTokenizer:
    """Character-level tokenizer for comparison."""

    def __init__(self):
        self.vocab = {chr(i): i - 32 for i in range(32, 127)}
        self.vocab["<pad>"] = 95
        self.vocab["<unk>"] = 96
        self.vocab["<s>"] = 97
        self.vocab["</s>"] = 98

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.vocab.get(c, 96) for c in text]  # 96 = <unk>
        if add_special_tokens:
            ids = [97] + ids + [98]  # <s> ... </s>
        return ids

    def decode(self, token_ids: list[int]) -> str:
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "?") for i in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def pad_token_id(self) -> int:
        return 95

    @property
    def unk_token_id(self) -> int:
        return 96


# Sample corpus for demonstration
SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models require lots of data",
    "Tokenization is the first step in NLP",
    "Natural language processing has many applications",
    "Deep learning revolutionized computer vision",
    "Transformers changed how we process text",
    "Attention is all you need",
    "BERT and GPT are popular language models",
    "Fine-tuning pretrained models is efficient",
    "Word embeddings capture semantic meaning",
    "hello",  # Short sample
    "This is a very long sentence that goes on and on with many words to demonstrate what happens when we have samples that are much longer than the typical sample in our corpus",  # Long sample
]


def demo_length_histograms():
    """Demonstrate token length histogram analysis."""
    print("\n" + "=" * 70)
    print("TOKEN LENGTH HISTOGRAM ANALYSIS")
    print("=" * 70)

    tokenizer = SimpleTokenizer()

    # Quick stats
    print("\n--- Quick Length Stats ---")
    stats = get_length_stats(SAMPLE_CORPUS, tokenizer)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Full histogram
    print("\n--- Full Histogram ---")
    histogram = compute_length_histogram(SAMPLE_CORPUS, tokenizer, num_bins=8)
    print(format_histogram_ascii(histogram, width=40))


def demo_oov_analysis():
    """Demonstrate OOV and rare token analysis."""
    print("\n" + "=" * 70)
    print("OOV AND RARE TOKEN ANALYSIS")
    print("=" * 70)

    tokenizer = SimpleTokenizer()

    # Frequency bands
    print("\n--- Token Frequency Bands ---")
    bands = get_frequency_bands(SAMPLE_CORPUS, tokenizer)
    for band, count in sorted(bands.items(), key=lambda x: x[0].value):
        print(f"  {band.value}: {count} tokens")

    # Find rare tokens
    print("\n--- Rare Tokens (appearing ≤2 times) ---")
    rare = find_rare_tokens(SAMPLE_CORPUS, tokenizer, max_frequency=2, top_k=10)
    for token_info in rare[:5]:
        print(f"  '{token_info.token_str}': {token_info.count}x ({token_info.band.value})")

    # Full OOV report
    print("\n--- OOV Report ---")
    report = analyze_oov(SAMPLE_CORPUS, tokenizer, vocab_size=1000)
    print(f"  Total tokens: {report.total_tokens}")
    print(f"  Unique tokens: {report.unique_tokens}")
    print(f"  Vocab utilization: {report.vocab_utilization:.1%}")
    print(f"  Singleton rate: {report.singleton_rate:.1%}")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")


def demo_waste_analysis():
    """Demonstrate padding and truncation waste analysis."""
    print("\n" + "=" * 70)
    print("TOKEN WASTE ANALYSIS")
    print("=" * 70)

    tokenizer = SimpleTokenizer()

    # Padding analysis
    print("\n--- Padding Analysis (max_length=32) ---")
    padding = analyze_padding_waste(SAMPLE_CORPUS, tokenizer, max_length=32)
    print(f"  Total positions: {padding.total_positions}")
    print(f"  Content tokens: {padding.total_content_tokens}")
    print(f"  Padding tokens: {padding.total_padding_tokens}")
    print(f"  Padding rate: {padding.padding_rate:.1%}")
    print(f"  Efficiency: {padding.efficiency:.1%}")

    # Truncation analysis
    print("\n--- Truncation Analysis (max_length=15) ---")
    truncation = analyze_truncation_loss(SAMPLE_CORPUS, tokenizer, max_length=15)
    print(f"  Truncated samples: {truncation.truncated_samples}/{truncation.total_samples}")
    print(f"  Truncation rate: {truncation.truncation_rate:.1%}")
    print(f"  Tokens lost: {truncation.total_tokens_lost}")
    print(f"  Content loss rate: {truncation.content_loss_rate:.1%}")
    print(f"  Severity - Minor: {truncation.minor_truncation}, Major: {truncation.major_truncation}, Severe: {truncation.severe_truncation}")

    # Combined waste report
    print("\n--- Combined Waste Report (max_length=20) ---")
    report = analyze_waste(SAMPLE_CORPUS, tokenizer, max_length=20)
    print(f"  Overall efficiency: {report.overall_efficiency:.1%}")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")


def demo_vocab_comparison():
    """Demonstrate before/after vocabulary comparison."""
    print("\n" + "=" * 70)
    print("VOCABULARY COMPARISON")
    print("=" * 70)

    word_tok = SimpleTokenizer()
    char_tok = CharTokenizer()

    # Compare tokenizers
    print("\n--- Word vs Character Tokenizer ---")
    report = compare_vocab_impact(
        SAMPLE_CORPUS,
        word_tok,
        char_tok,
        tokenizer1_name="WordLevel",
        tokenizer2_name="CharLevel",
    )

    print(f"  WordLevel vocab size: {report.tokenizer1_vocab_size}")
    print(f"  CharLevel vocab size: {report.tokenizer2_vocab_size}")
    print(f"  Total samples: {report.total_samples}")
    print(f"  ")
    print(f"  Tokens (WordLevel): {report.tokens1_total}")
    print(f"  Tokens (CharLevel): {report.tokens2_total}")
    print(f"  Token ratio: {report.token_count_ratio:.2f}x")
    print(f"  ")
    print(f"  Chars/token (WordLevel): {report.chars_per_token1:.2f}")
    print(f"  Chars/token (CharLevel): {report.chars_per_token2:.2f}")
    print(f"  ")
    print(f"  Training speedup with WordLevel: {1/report.token_count_ratio:.2f}x")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

    # Retokenization cost
    print("\n--- Retokenization Cost Estimate ---")
    cost = estimate_retokenization_cost(SAMPLE_CORPUS, word_tok, char_tok)
    print(f"  Vocab overlap: {cost['vocab_overlap']} tokens ({cost['vocab_overlap_rate']:.1%})")
    print(f"  New tokens: {cost['new_tokens']}")
    print(f"  Removed tokens: {cost['removed_tokens']}")
    print(f"  Embedding reuse rate: {cost['embedding_reuse_rate']:.1%}")


def main():
    """Run all instrumentation demos."""
    print("\n" + "#" * 70)
    print("#" + " TOKENIZER INSTRUMENTATION DEMO ".center(68) + "#")
    print("#" * 70)
    print("\nThis demo shows instrumentation tools for analyzing tokenization")
    print("behavior without modifying the tokenizer. These metrics help with:")
    print("  - Choosing optimal sequence lengths")
    print("  - Understanding tokenization efficiency")
    print("  - Comparing tokenizers for vocabulary swaps")
    print("  - Identifying rare/OOV token issues")

    demo_length_histograms()
    demo_oov_analysis()
    demo_waste_analysis()
    demo_vocab_comparison()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
