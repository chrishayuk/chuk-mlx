"""
Curriculum Learning Example

Demonstrates tokenizer-aware curriculum learning tools:
- Length-based buckets: Group samples by token length
- Curriculum schedules: Progressive training from easy to hard
- Reasoning density: Score texts by reasoning complexity

Uses Pydantic models for all data structures.
"""

from chuk_lazarus.data.tokenizers.curriculum import (
    CurriculumSchedule,
    LengthBucket,
    LengthBucketConfig,
    ReasoningConfig,
    ReasoningDensityScore,
    create_length_buckets,
    get_curriculum_schedule,
    get_difficulty_percentiles,
    score_reasoning_density,
    sort_by_length,
    sort_by_reasoning_density,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def demo_length_buckets():
    """Demonstrate length-based curriculum buckets."""
    print("=" * 60)
    print("Length-Based Curriculum Buckets")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Texts of varying lengths
    texts = [
        "Hi there!",  # Very short
        "How are you doing today?",  # Short
        "The quick brown fox jumps over the lazy dog.",  # Medium
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",  # Long
        "Natural language processing combines linguistics and computer science to enable machines to understand, interpret, and generate human language in meaningful ways.",  # Very long
        "Hello!",
        "What is the weather like?",
        "Please explain the concept of backpropagation in neural networks.",
        "Good morning.",
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
    ]

    # Create length buckets
    config = LengthBucketConfig(
        num_buckets=4,
        log_scale=False,
    )

    buckets: list[LengthBucket] = create_length_buckets(texts, tokenizer, config=config)

    print(f"\nCreated {len(buckets)} buckets from {len(texts)} texts:\n")
    for i, bucket in enumerate(buckets):
        print(f"Bucket {i + 1}:")
        print(f"  Range:    {bucket.min_tokens}-{bucket.max_tokens} tokens")
        print(f"  Samples:  {bucket.sample_count}")
        print(f"  Avg len:  {bucket.avg_length:.1f}")
        if bucket.sample_indices:
            sample_texts = [texts[idx][:30] + "..." for idx in bucket.sample_indices[:2]]
            print(f"  Examples: {sample_texts}")
        print()


def demo_curriculum_schedule():
    """Demonstrate curriculum schedule generation."""
    print("=" * 60)
    print("Curriculum Schedule")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Training corpus
    texts = [
        "Hello!",
        "Hi there!",
        "Good morning.",
        "How are you?",
        "What is your name?",
        "The weather is nice today.",
        "I enjoy learning about machine learning.",
        "Natural language processing is fascinating.",
        "The transformer architecture uses self-attention mechanisms to process sequences.",
        "Reinforcement learning from human feedback helps align language models with human preferences.",
    ]

    # Generate schedule
    config = LengthBucketConfig(num_buckets=3)
    schedule: CurriculumSchedule = get_curriculum_schedule(
        texts, tokenizer, config=config, warmup_ratio=0.1, ramp_ratio=0.3
    )

    print(f"\nCurriculum Schedule:")
    print(f"  Total buckets:  {len(schedule.buckets)}")
    print(f"  Total samples:  {schedule.total_samples}")
    print(f"  Warmup samples: {schedule.warmup_samples}")
    print(f"  Ramp samples:   {schedule.ramp_samples}")
    print(f"  Schedule order: {schedule.schedule_order} (bucket IDs, easy to hard)")

    print("\nBuckets (in schedule order):")
    for bucket_id in schedule.schedule_order:
        bucket = schedule.buckets[bucket_id]
        print(f"\n  Bucket {bucket_id}:")
        print(f"    Range: {bucket.min_tokens}-{bucket.max_tokens} tokens")
        print(f"    Sample count: {bucket.sample_count}")
        print(f"    Avg length: {bucket.avg_length:.1f}")


def demo_sort_by_length():
    """Demonstrate sorting by token length."""
    print("\n" + "=" * 60)
    print("Sort by Token Length")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    texts = [
        "Medium length sentence here.",
        "Short.",
        "This is a much longer sentence that contains many more tokens.",
        "Hello!",
        "A bit longer than the shortest one.",
    ]

    # Sort ascending (shortest first - typical for curriculum)
    sorted_asc = sort_by_length(texts, tokenizer, reverse=False)
    print("\nAscending order (curriculum learning):")
    for idx, text, length in sorted_asc[:5]:
        print(f"  {length:3} tokens: {text[:40]}...")

    # Sort descending (longest first)
    sorted_desc = sort_by_length(texts, tokenizer, reverse=True)
    print("\nDescending order:")
    for idx, text, length in sorted_desc[:5]:
        print(f"  {length:3} tokens: {text[:40]}...")


def demo_reasoning_density():
    """Demonstrate reasoning density scoring."""
    print("\n" + "=" * 60)
    print("Reasoning Density Scoring")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Texts with varying reasoning complexity
    texts = [
        "Hello, how are you?",  # Simple, no reasoning
        "The cat sat on the mat.",  # Narrative, no reasoning
        "If x = 5, then x + 3 = 8.",  # Math reasoning
        "Therefore, we can conclude that A implies B.",  # Logical reasoning
        "Let f(x) = x^2. Then f'(x) = 2x by the power rule.",  # Complex math
        "Step 1: Define the problem. Step 2: Analyze. Step 3: Solve.",  # Step-by-step
        "Given that p => q and p is true, we deduce q is true (modus ponens).",  # Formal logic
        "The weather is nice today.",  # Simple statement
    ]

    # Configure reasoning detection
    config = ReasoningConfig(
        math_symbol_weight=0.25,
        bracket_depth_weight=0.2,
        variable_weight=0.15,
        numeric_weight=0.15,
        operator_weight=0.15,
        length_weight=0.1,
    )

    print("\nScoring individual texts:\n")
    for i, text in enumerate(texts):
        score: ReasoningDensityScore = score_reasoning_density(
            text, i, tokenizer, config=config
        )
        print(f"  [{i}] Score: {score.overall_score:.4f}")
        print(f"      Text: {text[:50]}...")
        print()

    # Get sorted order for curriculum
    sorted_scores = sort_by_reasoning_density(texts, tokenizer, reverse=False)
    print("Sorted by reasoning density (easiest first):")
    for score in sorted_scores:
        print(f"  [{score.text_index}] {score.overall_score:.4f}: {texts[score.text_index][:40]}...")


def demo_difficulty_percentiles():
    """Demonstrate difficulty percentile calculation."""
    print("\n" + "=" * 60)
    print("Difficulty Percentiles")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Larger corpus for meaningful percentiles
    texts = [
        "Hello!",
        "Hi there.",
        "Good morning.",
        "The sky is blue.",
        "Water is wet.",
        "If x > 0, then x^2 > 0.",
        "Let n be a positive integer.",
        "Therefore, by induction, the claim holds.",
        "The derivative of x^n is nx^(n-1).",
        "Using the chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x).",
        "Proof: Assume the contrary. Then...",
        "Step 1: Initialize. Step 2: Iterate. Step 3: Terminate.",
    ]

    percentiles = get_difficulty_percentiles(texts, tokenizer)

    print(f"\nDifficulty Distribution:")
    print(f"  Mean:   {percentiles.mean_score:.4f}")
    print(f"  P25:    {percentiles.p25:.4f}")
    print(f"  P50:    {percentiles.p50:.4f} (median)")
    print(f"  P75:    {percentiles.p75:.4f}")
    print(f"  P90:    {percentiles.p90:.4f}")

    # Use percentiles for curriculum thresholds
    print("\nCurriculum thresholds:")
    print(f"  Easy (< P25):   score < {percentiles.p25:.4f}")
    print(f"  Medium (P25-P75): {percentiles.p25:.4f} <= score < {percentiles.p75:.4f}")
    print(f"  Hard (>= P75):  score >= {percentiles.p75:.4f}")


def main():
    """Run all curriculum learning demos."""
    print("Curriculum Learning Tools Demo")
    print("=" * 60)

    demo_length_buckets()
    demo_curriculum_schedule()
    demo_sort_by_length()
    demo_reasoning_density()
    demo_difficulty_percentiles()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
