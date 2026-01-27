#!/usr/bin/env python3
"""
Diversity analysis: Compare our training data with GSM-8K.

Analyzes:
- Question length distribution
- Value/number ranges
- Name diversity
- Domain/context variety
- Vocabulary richness
- Operation patterns
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_virtual_expert_arithmetic.generators import TraceGenerator
from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import load_gsm8k, SAMPLE_PROBLEMS


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    # Match integers, decimals, and dollar amounts
    pattern = r'\$?([\d,]+\.?\d*)'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return numbers


def extract_names(text: str) -> list[str]:
    """Extract capitalized words that look like names."""
    # Common non-name capitalized words
    non_names = {
        'The', 'A', 'An', 'If', 'How', 'What', 'When', 'Where', 'Why',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December', 'GB', 'Each', 'Every',
        'After', 'Before', 'During', 'Then', 'First', 'Second', 'Third',
        'Machine', 'Factory', 'Store', 'Shop', 'Day', 'Week', 'Month', 'Year',
        'In', 'On', 'At', 'For', 'To', 'From', 'With', 'By', 'Plus', 'Total',
    }
    words = re.findall(r'\b([A-Z][a-z]+)\b', text)
    return [w for w in words if w not in non_names]


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def analyze_dataset(questions: list[str], name: str) -> dict:
    """Analyze a dataset of questions."""
    print(f"\n{'='*60}")
    print(f"  {name} Analysis ({len(questions)} questions)")
    print(f"{'='*60}")

    # Word counts
    word_counts = [word_count(q) for q in questions]
    avg_words = sum(word_counts) / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)

    print(f"\n--- Question Length ---")
    print(f"  Average: {avg_words:.1f} words")
    print(f"  Range: {min_words} - {max_words} words")

    # Word count distribution
    short = sum(1 for w in word_counts if w < 20)
    medium = sum(1 for w in word_counts if 20 <= w < 40)
    long = sum(1 for w in word_counts if 40 <= w < 60)
    very_long = sum(1 for w in word_counts if w >= 60)

    print(f"  Distribution:")
    print(f"    <20 words:  {short:3} ({100*short/len(questions):5.1f}%)")
    print(f"    20-39:      {medium:3} ({100*medium/len(questions):5.1f}%)")
    print(f"    40-59:      {long:3} ({100*long/len(questions):5.1f}%)")
    print(f"    60+:        {very_long:3} ({100*very_long/len(questions):5.1f}%)")

    # Numbers
    all_numbers = []
    for q in questions:
        all_numbers.extend(extract_numbers(q))

    print(f"\n--- Number Values ---")
    print(f"  Total numbers: {len(all_numbers)}")
    print(f"  Avg per question: {len(all_numbers)/len(questions):.1f}")

    if all_numbers:
        print(f"  Range: {min(all_numbers):.1f} - {max(all_numbers):.1f}")

        # Value distribution
        small = sum(1 for n in all_numbers if n < 10)
        medium_n = sum(1 for n in all_numbers if 10 <= n < 100)
        large = sum(1 for n in all_numbers if 100 <= n < 1000)
        very_large = sum(1 for n in all_numbers if n >= 1000)
        decimals = sum(1 for n in all_numbers if n != int(n))

        print(f"  Distribution:")
        print(f"    <10:        {small:4} ({100*small/len(all_numbers):5.1f}%)")
        print(f"    10-99:      {medium_n:4} ({100*medium_n/len(all_numbers):5.1f}%)")
        print(f"    100-999:    {large:4} ({100*large/len(all_numbers):5.1f}%)")
        print(f"    1000+:      {very_large:4} ({100*very_large/len(all_numbers):5.1f}%)")
        print(f"    Decimals:   {decimals:4} ({100*decimals/len(all_numbers):5.1f}%)")

    # Names
    all_names = []
    for q in questions:
        all_names.extend(extract_names(q))

    name_counts = Counter(all_names)
    unique_names = len(name_counts)

    print(f"\n--- Name Diversity ---")
    print(f"  Total name mentions: {len(all_names)}")
    print(f"  Unique names: {unique_names}")
    print(f"  Top 10 names: {', '.join(n for n, _ in name_counts.most_common(10))}")

    # Questions with names vs without
    with_names = sum(1 for q in questions if extract_names(q))
    print(f"  Questions with names: {with_names} ({100*with_names/len(questions):.1f}%)")

    # Domain keywords
    domains = {
        'money': ['$', 'dollar', 'cost', 'price', 'pay', 'earn', 'profit', 'sell', 'buy'],
        'time': ['hour', 'minute', 'day', 'week', 'month', 'year', 'time'],
        'distance': ['mile', 'km', 'meter', 'feet', 'inch'],
        'food': ['egg', 'apple', 'cookie', 'bread', 'cake', 'meal', 'breakfast', 'lunch'],
        'animals': ['dog', 'cat', 'chicken', 'duck', 'sheep', 'horse', 'fish'],
        'work': ['job', 'task', 'work', 'employee', 'factory', 'machine', 'produce'],
        'school': ['student', 'class', 'grade', 'book', 'page', 'read'],
        'transport': ['car', 'bus', 'train', 'drive', 'trip', 'travel'],
    }

    print(f"\n--- Domain Coverage ---")
    domain_counts = {}
    for domain, keywords in domains.items():
        count = sum(1 for q in questions if any(kw in q.lower() for kw in keywords))
        domain_counts[domain] = count
        print(f"  {domain:12}: {count:3} ({100*count/len(questions):5.1f}%)")

    # Word numbers vs digits
    word_number_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b'
    with_word_numbers = sum(1 for q in questions if re.search(word_number_pattern, q.lower()))

    print(f"\n--- Number Format ---")
    print(f"  Word numbers (three vs 3): {with_word_numbers} ({100*with_word_numbers/len(questions):.1f}%)")

    # Vocabulary richness
    all_words = []
    for q in questions:
        all_words.extend(q.lower().split())

    vocab = set(all_words)
    print(f"\n--- Vocabulary ---")
    print(f"  Total words: {len(all_words)}")
    print(f"  Unique words: {len(vocab)}")
    print(f"  Type-token ratio: {len(vocab)/len(all_words):.3f}")

    return {
        'count': len(questions),
        'avg_words': avg_words,
        'unique_names': unique_names,
        'domains': domain_counts,
        'word_numbers_pct': 100*with_word_numbers/len(questions),
        'vocab_size': len(vocab),
        'ttr': len(vocab)/len(all_words),
        'decimal_pct': 100*decimals/len(all_numbers) if all_numbers else 0,
    }


def main():
    print("=" * 60)
    print("  DIVERSITY ANALYSIS: Our Training Data vs GSM-8K")
    print("=" * 60)

    # Load our training data
    print("\nGenerating training data...")
    gen = TraceGenerator(seed=42)
    train_data = gen.generate_balanced(n=300, include_composition=True)

    our_questions = []
    for ex in train_data:
        if isinstance(ex, dict):
            our_questions.append(ex.get('query', ''))
        else:
            our_questions.append(ex.query)

    our_questions = [q for q in our_questions if q]  # Filter empty

    # Load GSM-8K
    print("Loading GSM-8K test set...")
    try:
        gsm8k = load_gsm8k(n=300, split="test", shuffle=True, seed=42)
        gsm_questions = [p.question for p in gsm8k]
    except Exception as e:
        print(f"Could not load GSM-8K from HuggingFace: {e}")
        print("Using sample problems instead...")
        gsm_questions = [p.question for p in SAMPLE_PROBLEMS]

    # Analyze both
    our_stats = analyze_dataset(our_questions, "OUR TRAINING DATA")
    gsm_stats = analyze_dataset(gsm_questions, "GSM-8K TEST SET")

    # Comparison summary
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Ours':>12} {'GSM-8K':>12} {'Gap':>12}")
    print("-" * 60)

    metrics = [
        ('Avg question length', 'avg_words', 'words'),
        ('Unique names', 'unique_names', ''),
        ('Word numbers usage', 'word_numbers_pct', '%'),
        ('Decimal values', 'decimal_pct', '%'),
        ('Vocabulary size', 'vocab_size', ''),
        ('Type-token ratio', 'ttr', ''),
    ]

    for label, key, unit in metrics:
        ours = our_stats.get(key, 0)
        gsm = gsm_stats.get(key, 0)
        if isinstance(ours, float):
            gap = gsm - ours
            print(f"{label:<25} {ours:>10.1f}{unit:>2} {gsm:>10.1f}{unit:>2} {gap:>+10.1f}")
        else:
            gap = gsm - ours
            print(f"{label:<25} {ours:>12} {gsm:>12} {gap:>+12}")

    print("\n--- Domain Coverage Comparison ---")
    print(f"{'Domain':<15} {'Ours':>10} {'GSM-8K':>10} {'Gap':>10}")
    print("-" * 45)
    for domain in our_stats['domains']:
        ours = our_stats['domains'].get(domain, 0)
        gsm = gsm_stats['domains'].get(domain, 0)
        gap = gsm - ours
        print(f"{domain:<15} {ours:>9}% {gsm:>9}% {gap:>+9}%")

    print("\n--- Key Gaps to Address ---")

    gaps = []
    if our_stats['avg_words'] < gsm_stats['avg_words'] * 0.8:
        gaps.append(f"Question length: {our_stats['avg_words']:.0f} vs {gsm_stats['avg_words']:.0f} words")
    if our_stats['unique_names'] < gsm_stats['unique_names'] * 0.6:
        gaps.append(f"Name diversity: {our_stats['unique_names']} vs {gsm_stats['unique_names']} unique")
    if our_stats['word_numbers_pct'] < gsm_stats['word_numbers_pct'] * 0.5:
        gaps.append(f"Word numbers: {our_stats['word_numbers_pct']:.0f}% vs {gsm_stats['word_numbers_pct']:.0f}%")
    if our_stats['decimal_pct'] < gsm_stats['decimal_pct'] * 0.5:
        gaps.append(f"Decimals: {our_stats['decimal_pct']:.0f}% vs {gsm_stats['decimal_pct']:.0f}%")

    for domain, our_pct in our_stats['domains'].items():
        gsm_pct = gsm_stats['domains'].get(domain, 0)
        if gsm_pct > 20 and our_pct < gsm_pct * 0.5:
            gaps.append(f"{domain} domain: {our_pct}% vs {gsm_pct}%")

    if gaps:
        for gap in gaps:
            print(f"  - {gap}")
    else:
        print("  No major gaps detected!")


if __name__ == "__main__":
    main()
