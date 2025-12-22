"""
Training Utilities Example

Demonstrates tokenizer training utilities:
- Sequence packing for efficient GPU utilization
- Throughput profiling for performance optimization

Uses Pydantic models for all data structures.
"""

from chuk_lazarus.data.tokenizers.training import (
    # Packer
    PackedSequence,
    PackingConfig,
    ThroughputMetrics,
    ThroughputProfiler,
    estimate_training_tokens,
    pack_sequences,
    profile_tokenization,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def demo_sequence_packing():
    """Demonstrate sequence packing for efficient training."""
    print("=" * 60)
    print("Sequence Packing")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Short sequences that can be packed together
    texts = [
        "Hello!",
        "How are you?",
        "I'm fine, thanks.",
        "What's the weather?",
        "It's sunny today.",
        "Great to hear!",
        "See you later.",
        "Goodbye!",
        "Machine learning is interesting.",
        "I agree completely.",
    ]

    # Tokenize first
    token_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    # Configure packing
    config = PackingConfig(
        max_seq_length=128,
        pad_token_id=tokenizer.pad_token_id or 0,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("\nPacking configuration:")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Padding token ID:    {config.pad_token_id}")
    print(f"  EOS token ID:        {config.eos_token_id}")
    print(f"  Add EOS between:     {config.add_eos_between}")

    # Pack sequences
    packed: list[PackedSequence] = pack_sequences(token_sequences, config)

    print("\nPacking results:")
    print(f"  Input sequences:  {len(texts)}")
    print(f"  Packed sequences: {len(packed)}")
    print(f"  Packing ratio:    {len(texts) / len(packed):.2f}x")

    # Show details of packed sequences
    print("\nPacked sequence details:")
    for i, p in enumerate(packed[:3]):
        print(f"\n  Sequence {i + 1}:")
        print(f"    Token count:    {len(p.token_ids)}")
        print(f"    Real tokens:    {p.num_real_tokens}")
        print(f"    Padding:        {p.num_padding_tokens}")
        print(f"    Sources:        {p.source_indices}")


def demo_packing_efficiency():
    """Demonstrate packing efficiency calculation."""
    print("\n" + "=" * 60)
    print("Packing Efficiency Analysis")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Mix of short and long sequences
    texts = [
        "Hi!",
        "Hello there!",
        "How are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "A",
        "This is a longer sentence that contains more tokens.",
        "Short.",
    ]

    # Tokenize
    token_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    config = PackingConfig(
        max_seq_length=64,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    # Pack sequences
    packed = pack_sequences(token_sequences, config)

    # Calculate efficiency manually
    total_tokens = sum(len(p.token_ids) for p in packed)
    real_tokens = sum(p.num_real_tokens for p in packed)
    padding_tokens = sum(p.num_padding_tokens for p in packed)
    efficiency = real_tokens / total_tokens if total_tokens > 0 else 0

    print("\nEfficiency statistics:")
    print(f"  Total tokens:        {total_tokens}")
    print(f"  Real tokens:         {real_tokens}")
    print(f"  Padding tokens:      {padding_tokens}")
    print(f"  Efficiency:          {efficiency:.2%}")
    print(f"  Avg seqs per pack:   {len(texts) / len(packed):.2f}")


def demo_throughput_profiling():
    """Demonstrate throughput profiling."""
    print("\n" + "=" * 60)
    print("Throughput Profiling")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Generate test corpus
    texts = [
        f"Sample text number {i} for throughput testing with varying length." for i in range(100)
    ]

    # Profile tokenization using profiler class
    profiler = ThroughputProfiler(tokenizer)

    # Profile in batches
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        profiler.profile_batch(batch)

    # Get accumulated metrics
    metrics: ThroughputMetrics = profiler.get_metrics()

    print("\nThroughput metrics:")
    print(f"  Tokens/second:     {metrics.tokens_per_second:,.0f}")
    print(f"  Chars/second:      {metrics.chars_per_second:,.0f}")
    print(f"  Total tokens:      {metrics.total_tokens:,}")
    print(f"  Total texts:       {metrics.total_texts}")
    print(f"  Elapsed time:      {metrics.elapsed_seconds:.4f} s")
    print(f"  Avg tokens/text:   {metrics.avg_tokens_per_text:.1f}")
    print(f"  Avg chars/token:   {metrics.avg_chars_per_token:.2f}")


def demo_profile_tokenization():
    """Demonstrate convenience function for profiling."""
    print("\n" + "=" * 60)
    print("Quick Profile Function")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    texts = [f"Quick profile test {i}." for i in range(50)]

    # Use convenience function
    metrics = profile_tokenization(texts, tokenizer)

    print("\nQuick profile results:")
    print(f"  Throughput:      {metrics.tokens_per_second:,.0f} tokens/s")
    print(f"  Total texts:     {metrics.total_texts}")
    print(f"  Total tokens:    {metrics.total_tokens}")
    print(f"  Elapsed:         {metrics.elapsed_seconds:.4f} s")


def demo_estimate_training_tokens():
    """Demonstrate training token estimation."""
    print("\n" + "=" * 60)
    print("Training Token Estimation")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample from dataset
    sample_texts = [
        "This is a sample from the training dataset.",
        "Another example of training data.",
        "Machine learning requires lots of data.",
        "The model learns patterns from examples.",
        "Training takes time and compute.",
    ]

    # Estimate for larger dataset (sample is 10% of full dataset)
    estimate = estimate_training_tokens(sample_texts, tokenizer, epochs=1, sample_ratio=0.1)

    print("\nTraining token estimation:")
    print(f"  Sample size:          {estimate['sample_texts']}")
    print(f"  Sample tokens:        {estimate['sample_tokens']}")
    print(f"  Avg tokens/text:      {estimate['avg_tokens_per_text']:.1f}")
    print(f"  Sample ratio:         {estimate['sample_ratio']:.0%}")
    print(f"  Estimated dataset:    {estimate['estimated_dataset_tokens']:,} tokens")

    # Show for different epoch counts
    for epochs in [1, 3, 10]:
        est = estimate_training_tokens(sample_texts, tokenizer, epochs=epochs, sample_ratio=0.1)
        print(f"  {epochs} epoch(s):          {est['total_training_tokens']:,} tokens")


def main():
    """Run all training utilities demos."""
    print("Training Utilities Demo")
    print("=" * 60)

    demo_sequence_packing()
    demo_packing_efficiency()
    demo_throughput_profiling()
    demo_profile_tokenization()
    demo_estimate_training_tokens()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
