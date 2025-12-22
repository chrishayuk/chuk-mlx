"""
Tokenizer Analysis Example

Demonstrates the comprehensive token analysis tools:
- Coverage analysis (UNK rate, vocab utilization)
- Entropy analysis (distribution, perplexity)
- Fit score (tokenizer-dataset compatibility)
- Diff analysis (compare tokenizers on corpus)

Uses Pydantic models for all outputs.
"""

from chuk_lazarus.data.tokenizers.analyze import (
    CoverageReport,
    EntropyReport,
    FitScore,
    analyze_coverage,
    analyze_entropy,
    calculate_fit_score,
    compare_tokenizers_for_dataset,
    diff_corpus,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def demo_coverage_analysis():
    """Demonstrate coverage analysis."""
    print("=" * 60)
    print("Coverage Analysis")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample texts - mix of common and rare words
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms natural language processing.",
        "Quantum computing leverages superposition and entanglement.",
        "The implementation uses polymorphism and encapsulation.",
        "Cryptocurrency transactions require cryptographic signatures.",
    ]

    # Analyze coverage
    report: CoverageReport = analyze_coverage(texts, tokenizer, include_fragments=True)

    print(f"\nAnalyzed {len(texts)} texts")
    print(f"Total tokens:      {report.total_tokens:,}")
    print(f"Unique tokens:     {report.unique_tokens_used:,}")
    print(f"UNK rate:          {report.unk_rate:.2%}")
    print(f"Tokens per word:   {report.tokens_per_word:.2f}")
    print(f"Vocab utilization: {report.vocab_utilization:.2%}")

    if report.domain_warnings:
        print("\nWarnings:")
        for warning in report.domain_warnings:
            print(f"  - {warning}")

    if report.fragment_analysis:
        print(f"\nFragment analysis:")
        print(f"  Fragment ratio: {report.fragment_analysis.fragment_ratio:.2%}")
        if report.fragment_analysis.top_fragments:
            print(f"  Top fragments: {report.fragment_analysis.top_fragments[:5]}")


def demo_entropy_analysis():
    """Demonstrate entropy analysis."""
    print("\n" + "=" * 60)
    print("Entropy Analysis")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Varied texts for entropy analysis
    texts = [
        "The the the the the",  # Low entropy (repetitive)
        "Hello world how are you doing today",  # Medium entropy
        "Cryptographic hashing algorithms ensure data integrity.",  # Higher entropy
        "Machine learning models predict outcomes from data.",
    ]

    report: EntropyReport = analyze_entropy(texts, tokenizer, top_n=20)

    print(f"\nEntropy:           {report.entropy:.4f} bits")
    print(f"Perplexity:        {report.perplexity:.2f}")
    print(f"Normalized:        {report.normalized_entropy:.4f}")
    print(f"Uniformity:        {report.uniformity_score:.2%}")
    print(f"Concentration:     {report.concentration_ratio:.2%}")

    if report.distribution:
        print(f"\nToken distribution:")
        print(f"  Unique tokens:    {report.distribution.unique_tokens}")
        print(f"  Type-token ratio: {report.distribution.type_token_ratio:.4f}")
        print(f"  Singletons:       {report.distribution.singleton_count}")

        print(f"\nTop tokens:")
        for i, (token_id, decoded, count) in enumerate(report.distribution.top_tokens[:10]):
            print(f"  {i + 1:2}. {decoded!r:15} (id={token_id}) {count:4}")


def demo_fit_score():
    """Demonstrate fit score calculation."""
    print("\n" + "=" * 60)
    print("Fit Score Analysis")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample dataset
    texts = [
        "What is the capital of France?",
        "Paris is the capital and largest city of France.",
        "How do neural networks learn?",
        "Neural networks learn through backpropagation of errors.",
        "Explain quantum entanglement.",
        "Quantum entanglement is a phenomenon where particles are correlated.",
    ]

    score: FitScore = calculate_fit_score(texts, tokenizer)

    print(f"\nFit Score:   {score.overall_score:.2%}")
    print(f"Coverage:    {score.coverage_score:.2%}")
    print(f"Compression: {score.compression_score:.2%}")
    print(f"Entropy:     {score.entropy_score:.2%}")
    print(f"Vocab util:  {score.vocab_utilization_score:.2%}")

    print(f"\nRecommendation: {score.recommendation}")

    if score.details:
        print("\nDetails:")
        for key, value in score.details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def demo_tokenizer_comparison():
    """Demonstrate tokenizer comparison on a corpus."""
    print("\n" + "=" * 60)
    print("Tokenizer Comparison")
    print("=" * 60)

    # Note: In real usage, you'd compare different tokenizers
    # Here we use the same tokenizer for demonstration
    tok1 = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok2 = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    texts = [
        "Hello, how are you today?",
        "The weather is beautiful.",
        "Machine learning is transforming industries.",
        "Natural language processing enables AI to understand text.",
    ]

    # Compare using diff_corpus
    diff = diff_corpus(texts, tok1, tok2)

    print(f"\nTexts compared:       {diff.total_texts}")
    print(f"Tokenizer 1 total:    {diff.tokenizer1_total_tokens:,} tokens")
    print(f"Tokenizer 2 total:    {diff.tokenizer2_total_tokens:,} tokens")
    print(f"Avg length delta:     {diff.avg_length_delta:+.2f}")
    print(f"Compression change:   {diff.compression_improvement:.2%}")

    # Compare using fit scores
    comparison = compare_tokenizers_for_dataset(texts, tok1, tok2)

    print(f"\nFit Score Comparison:")
    print(f"  Tokenizer 1: {comparison.tokenizer1_score.overall_score:.2%}")
    print(f"  Tokenizer 2: {comparison.tokenizer2_score.overall_score:.2%}")
    print(f"  Winner:      {comparison.winner}")
    print(f"  Delta:       {comparison.score_delta:.2%}")
    if comparison.comparison_notes:
        print(f"  Notes:       {', '.join(comparison.comparison_notes)}")


def main():
    """Run all analysis demos."""
    print("Tokenizer Analysis Tools Demo")
    print("=" * 60)

    demo_coverage_analysis()
    demo_entropy_analysis()
    demo_fit_score()
    demo_tokenizer_comparison()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
