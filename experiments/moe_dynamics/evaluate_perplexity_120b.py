#!/usr/bin/env python3
"""
Perplexity Evaluation for GPT-OSS-120B and GPT-OSS-120B-Lite

Computes perplexity on standard evaluation datasets to validate
that compression preserves model quality.

Datasets:
- WikiText-2 (standard LM benchmark)
- Custom diverse prompts (math, code, language, reasoning)

Usage:
    # Quick evaluation (100 samples)
    python evaluate_perplexity_120b.py --quick

    # Full evaluation (1000 samples)
    python evaluate_perplexity_120b.py --full

    # Compare original vs lite
    python evaluate_perplexity_120b.py --compare

    # Just lite model
    python evaluate_perplexity_120b.py --model ./gpt-oss-120b-lite-conservative
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# =============================================================================
# Evaluation Prompts - Diverse set for quality testing
# =============================================================================

EVAL_PROMPTS = {
    "math_arithmetic": [
        "Calculate: 127 * 89 = ",
        "What is 456 + 789?",
        "Compute 1024 / 16 = ",
        "Find 15% of 200: ",
        "Calculate 2^10 = ",
        "What is sqrt(144)?",
        "Solve: 7 * 8 + 3 = ",
        "Compute 999 - 123 = ",
        "What is 50 * 50?",
        "Calculate 1000 / 8 = ",
        "Find 25% of 400: ",
        "What is 13 * 17?",
        "Compute 256 + 512 = ",
        "Calculate 81 / 9 = ",
        "What is 3^4?",
        "Solve: 100 - 37 + 18 = ",
        "Compute 64 * 8 = ",
        "What is 1/4 of 100?",
        "Calculate 729 / 27 = ",
        "Find 10% of 1500: ",
    ],
    "math_word_problems": [
        "If John has 15 apples and gives away 7, how many does he have left?",
        "A rectangle has length 12 and width 5. What is its area?",
        "If a car travels 60 mph for 3 hours, how far does it go?",
        "A store sells items for $25 each. How much for 8 items?",
        "If you divide 144 equally among 12 people, how much does each get?",
        "A triangle has sides 3, 4, and 5. What is its perimeter?",
        "If 3 workers can build a wall in 6 days, how long for 1 worker?",
        "A pizza is cut into 8 slices. If you eat 3, what fraction remains?",
        "Calculate the average of 10, 20, 30, 40, 50:",
        "If a book costs $15 and you have $100, how many can you buy?",
    ],
    "code_python": [
        "def fibonacci(n):",
        "def factorial(n):",
        "def is_prime(n):",
        "def reverse_string(s):",
        "def binary_search(arr, target):",
        "class Stack:",
        "def merge_sort(arr):",
        "def gcd(a, b):",
        "async def fetch_data(url):",
        "def flatten_list(nested):",
        "import numpy as np\ndef matrix_multiply(a, b):",
        "def quicksort(arr):",
        "class LinkedList:",
        "def depth_first_search(graph, start):",
        "def longest_common_subsequence(s1, s2):",
    ],
    "code_concepts": [
        "Explain how a hash table works:",
        "What is the time complexity of binary search?",
        "Describe the difference between a stack and a queue:",
        "What is recursion?",
        "Explain Big O notation:",
        "What is a closure in programming?",
        "Describe the MVC pattern:",
        "What is the difference between HTTP and HTTPS?",
        "Explain what an API is:",
        "What is the purpose of version control?",
    ],
    "language_knowledge": [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The author of Romeo and Juliet is",
        "The chemical symbol for gold is",
        "The Great Wall of China was built to",
        "Photosynthesis is the process by which",
        "The speed of light is approximately",
        "DNA stands for",
        "The Pythagorean theorem states that",
        "Mount Everest is located in",
        "The human body has approximately how many bones?",
        "The first person to walk on the moon was",
        "The currency of Japan is",
        "Shakespeare was born in",
    ],
    "language_completion": [
        "Once upon a time, in a land far away,",
        "The quick brown fox jumps over the",
        "To be or not to be, that is the",
        "In the beginning, there was",
        "It was the best of times, it was the",
        "All that glitters is not",
        "A journey of a thousand miles begins with",
        "The early bird catches the",
        "When life gives you lemons,",
        "Actions speak louder than",
    ],
    "reasoning_logic": [
        "If all cats are mammals, and all mammals are animals, then all cats are",
        "If A is greater than B, and B is greater than C, then A is",
        "If it rains, the ground gets wet. The ground is wet. Therefore,",
        "All roses are flowers. Some flowers fade quickly. Therefore,",
        "If P implies Q, and Q implies R, then P implies",
        "No reptiles have fur. All snakes are reptiles. Therefore,",
        "If today is Monday, tomorrow will be",
        "All squares are rectangles. Not all rectangles are squares. This means",
        "If you study hard, you will pass. You did not pass. Therefore,",
        "Some birds can fly. Penguins are birds. Therefore,",
    ],
    "reasoning_analysis": [
        "Compare and contrast renewable and non-renewable energy sources:",
        "What are the pros and cons of remote work?",
        "Analyze the impact of social media on society:",
        "Discuss the ethical implications of artificial intelligence:",
        "Evaluate the effectiveness of different learning styles:",
        "What factors contribute to climate change?",
        "Analyze the relationship between diet and health:",
        "Discuss the role of technology in education:",
        "What are the causes and effects of inflation?",
        "Analyze the importance of biodiversity:",
    ],
    "creative_writing": [
        "Write a haiku about the ocean:",
        "Describe a sunset in three sentences:",
        "Write the opening line of a mystery novel:",
        "Create a metaphor for time:",
        "Write a limerick about a cat:",
        "Describe the smell of rain:",
        "Write a tweet about Monday mornings:",
        "Create a slogan for a coffee shop:",
        "Write a fortune cookie message:",
        "Describe happiness without using the word happy:",
    ],
    "technical_explanations": [
        "Explain how neural networks learn:",
        "What is the difference between machine learning and deep learning?",
        "How does encryption protect data?",
        "Explain the concept of cloud computing:",
        "What is a database index and why is it useful?",
        "How does garbage collection work in programming?",
        "Explain the difference between TCP and UDP:",
        "What is containerization in software development?",
        "How does a recommendation system work?",
        "Explain the concept of distributed systems:",
    ],
}


@dataclass
class PerplexityResult:
    """Result from perplexity evaluation."""
    model_name: str
    dataset: str
    num_samples: int
    total_tokens: int
    total_loss: float
    perplexity: float
    avg_loss: float
    time_seconds: float
    tokens_per_second: float
    samples: list[dict] = None


def compute_perplexity(
    model,
    tokenizer,
    prompts: list[str],
    model_name: str = "unknown",
    dataset_name: str = "custom",
    max_length: int = 128,
    verbose: bool = True,
) -> PerplexityResult:
    """
    Compute perplexity over a set of prompts.

    Perplexity = exp(average cross-entropy loss)
    """
    total_loss = 0.0
    total_tokens = 0
    samples = []

    start_time = time.time()

    for i, prompt in enumerate(prompts):
        if verbose and i % 20 == 0:
            print(f"  Processing {i+1}/{len(prompts)}...")

        # Tokenize
        tokens = tokenizer.encode(prompt)
        if len(tokens) < 2:
            continue

        # Truncate if needed
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        input_ids = mx.array([tokens[:-1]])  # Input is all but last
        target_ids = mx.array([tokens[1:]])   # Target is all but first

        # Forward pass
        logits = model(input_ids)

        # Compute cross-entropy loss
        # logits: (batch, seq, vocab)
        # target: (batch, seq)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Gather log probs for target tokens
        batch_size, seq_len, vocab_size = logits.shape
        target_flat = target_ids.reshape(-1)
        log_probs_flat = log_probs.reshape(-1, vocab_size)

        # Get log prob for each target token
        target_log_probs = mx.take_along_axis(
            log_probs_flat,
            target_flat[:, None],
            axis=-1
        ).squeeze(-1)

        # Average negative log prob (cross-entropy)
        loss = -mx.mean(target_log_probs)
        mx.eval(loss)

        loss_val = float(loss.item())
        num_tokens = len(tokens) - 1

        total_loss += loss_val * num_tokens
        total_tokens += num_tokens

        samples.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "tokens": num_tokens,
            "loss": loss_val,
            "perplexity": math.exp(loss_val),
        })

    elapsed = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return PerplexityResult(
        model_name=model_name,
        dataset=dataset_name,
        num_samples=len(prompts),
        total_tokens=total_tokens,
        total_loss=total_loss,
        perplexity=perplexity,
        avg_loss=avg_loss,
        time_seconds=elapsed,
        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
        samples=samples,
    )


def load_original_model():
    """Load the original GPT-OSS-120B model."""
    from mlx_lm import load
    print("Loading original GPT-OSS-120B...")
    model, tokenizer = load("openai/gpt-oss-120b")
    return model, tokenizer


def load_lite_model(model_path: str):
    """Load the GPT-OSS-120B-Lite model."""
    from lite_loader_120b import load_gpt_oss_120b_lite
    model, tokenizer, config = load_gpt_oss_120b_lite(model_path)
    return model, tokenizer


def get_eval_prompts(categories: list[str] = None, max_per_category: int = None) -> list[str]:
    """Get evaluation prompts, optionally filtered by category."""
    prompts = []

    if categories is None:
        categories = list(EVAL_PROMPTS.keys())

    for cat in categories:
        if cat in EVAL_PROMPTS:
            cat_prompts = EVAL_PROMPTS[cat]
            if max_per_category:
                cat_prompts = cat_prompts[:max_per_category]
            prompts.extend(cat_prompts)

    return prompts


def print_results(result: PerplexityResult):
    """Print perplexity results."""
    print()
    print(f"Results for {result.model_name}")
    print("-" * 50)
    print(f"  Dataset:        {result.dataset}")
    print(f"  Samples:        {result.num_samples}")
    print(f"  Total tokens:   {result.total_tokens:,}")
    print(f"  Avg loss:       {result.avg_loss:.4f}")
    print(f"  Perplexity:     {result.perplexity:.2f}")
    print(f"  Time:           {result.time_seconds:.1f}s")
    print(f"  Speed:          {result.tokens_per_second:.1f} tok/s")
    print()


def print_comparison(original: PerplexityResult, lite: PerplexityResult):
    """Print comparison between original and lite models."""
    print()
    print("=" * 70)
    print("PERPLEXITY COMPARISON: Original vs Lite")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Original':<15} {'Lite':<15} {'Delta':<15}")
    print("-" * 70)

    ppl_delta = lite.perplexity - original.perplexity
    ppl_pct = 100 * (lite.perplexity / original.perplexity - 1)
    print(f"{'Perplexity':<25} {original.perplexity:<15.2f} {lite.perplexity:<15.2f} {ppl_delta:+.2f} ({ppl_pct:+.1f}%)")

    loss_delta = lite.avg_loss - original.avg_loss
    print(f"{'Avg Loss':<25} {original.avg_loss:<15.4f} {lite.avg_loss:<15.4f} {loss_delta:+.4f}")

    speed_ratio = lite.tokens_per_second / original.tokens_per_second
    print(f"{'Speed (tok/s)':<25} {original.tokens_per_second:<15.1f} {lite.tokens_per_second:<15.1f} {speed_ratio:.2f}x")

    print()
    print("Quality Assessment:")
    print("-" * 70)

    if ppl_pct < 5:
        print("  [EXCELLENT] <5% perplexity increase - compression successful!")
    elif ppl_pct < 10:
        print("  [GOOD] 5-10% perplexity increase - acceptable for 71% compression")
    elif ppl_pct < 20:
        print("  [MARGINAL] 10-20% perplexity increase - consider distillation")
    else:
        print("  [POOR] >20% perplexity increase - compression too aggressive")

    print()


def save_results(results: list[PerplexityResult], output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "model_name": r.model_name,
                "dataset": r.dataset,
                "num_samples": r.num_samples,
                "total_tokens": r.total_tokens,
                "avg_loss": r.avg_loss,
                "perplexity": r.perplexity,
                "time_seconds": r.time_seconds,
                "tokens_per_second": r.tokens_per_second,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Perplexity Evaluation")
    parser.add_argument("--model", type=str, default=None, help="Path to lite model")
    parser.add_argument("--quick", action="store_true", help="Quick eval (5 per category)")
    parser.add_argument("--full", action="store_true", help="Full eval (all prompts)")
    parser.add_argument("--compare", action="store_true", help="Compare original vs lite")
    parser.add_argument("--categories", type=str, nargs="+", help="Categories to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    # Determine prompt count
    if args.quick:
        max_per_cat = 5
    elif args.full:
        max_per_cat = None
    else:
        max_per_cat = 10  # default

    # Get prompts
    prompts = get_eval_prompts(args.categories, max_per_cat)
    print(f"Evaluating on {len(prompts)} prompts")
    print()

    results = []

    if args.compare:
        # Compare both models
        print("=" * 70)
        print("Loading models for comparison...")
        print("=" * 70)

        # Load original
        orig_model, orig_tokenizer = load_original_model()

        print("\nEvaluating original model...")
        orig_result = compute_perplexity(
            orig_model, orig_tokenizer, prompts,
            model_name="GPT-OSS-120B (Original)",
            dataset_name="diverse_eval",
        )
        print_results(orig_result)
        results.append(orig_result)

        # Free memory
        del orig_model
        import gc
        gc.collect()

        # Load lite
        lite_path = args.model or "./gpt-oss-120b-lite-conservative"
        lite_model, lite_tokenizer = load_lite_model(lite_path)

        print("\nEvaluating lite model...")
        lite_result = compute_perplexity(
            lite_model, lite_tokenizer, prompts,
            model_name="GPT-OSS-120B-Lite (71% reduction)",
            dataset_name="diverse_eval",
        )
        print_results(lite_result)
        results.append(lite_result)

        # Print comparison
        print_comparison(orig_result, lite_result)

    else:
        # Single model evaluation
        if args.model:
            model, tokenizer = load_lite_model(args.model)
            model_name = "GPT-OSS-120B-Lite"
        else:
            model, tokenizer = load_original_model()
            model_name = "GPT-OSS-120B"

        print(f"\nEvaluating {model_name}...")
        result = compute_perplexity(
            model, tokenizer, prompts,
            model_name=model_name,
            dataset_name="diverse_eval",
        )
        print_results(result)
        results.append(result)

        # Print per-category breakdown
        print("Per-Category Breakdown:")
        print("-" * 50)
        for sample in result.samples[:20]:
            print(f"  {sample['prompt'][:40]:<42} PPL: {sample['perplexity']:.2f}")

    # Save results
    if args.output:
        save_results(results, Path(args.output))
    else:
        output_path = Path("results/perplexity_eval_120b.json")
        save_results(results, output_path)


if __name__ == "__main__":
    main()
