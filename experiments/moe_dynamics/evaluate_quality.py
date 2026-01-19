#!/usr/bin/env python3
"""
Evaluate Quality: Compare GPT-OSS and TieredLightweight models.

Metrics:
1. Perplexity on held-out text
2. Generation quality (coherence check)
3. Parameter efficiency (quality per param)

Usage:
    python experiments/moe_dynamics/evaluate_quality.py
    python experiments/moe_dynamics/evaluate_quality.py --prompts-file custom_prompts.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    model_name: str
    perplexity: float
    avg_loss: float
    num_tokens: int
    inference_time_ms: float
    params: int = 0
    generations: list[str] = field(default_factory=list)


# Test prompts for evaluation
EVAL_PROMPTS = [
    # Arithmetic
    "Calculate 127 * 89 = ",
    "What is the square root of 256? The answer is",
    # Code
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\n# Filter rows where age > 30\nfiltered =",
    # Language
    "The capital of Japan is",
    "Shakespeare's most famous tragedy is probably",
    "In the beginning, there was",
    # Reasoning
    "If all dogs are mammals, and all mammals are animals, then all dogs are",
    "The pattern 2, 4, 8, 16, 32 continues as",
    # General
    "The quick brown fox jumps over",
    "To make a good cup of coffee, you should",
]

# Longer texts for perplexity evaluation
PERPLEXITY_TEXTS = [
    """Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on developing
    computer programs that can access data and use it to learn for themselves.""",

    """The mitochondria is often referred to as the powerhouse of the cell because it generates
    most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of
    chemical energy. In addition to supplying cellular energy, mitochondria are involved in
    other tasks such as signaling, cellular differentiation, and cell death.""",

    """def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)

    # Calculate the first 10 Fibonacci numbers
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")""",

    """Climate change refers to long-term shifts in temperatures and weather patterns. These
    shifts may be natural, but since the 1800s, human activities have been the main driver
    of climate change, primarily due to burning fossil fuels like coal, oil and gas.""",

    """The French Revolution was a period of radical political and societal change in France
    that began with the Estates General of 1789 and ended with the formation of the French
    Consulate in November 1799. Many of its ideas are considered fundamental principles of
    liberal democracy.""",
]


def load_model(model_id: str):
    """Load a model for evaluation."""
    from mlx_lm import load

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)
    return model, tokenizer


def count_parameters(model) -> int:
    """Count total parameters in model."""
    def _count_recursive(params) -> int:
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count_recursive(v)
        elif isinstance(params, list):
            for v in params:
                total += _count_recursive(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    return _count_recursive(model.parameters())


def compute_perplexity(model, tokenizer, texts: list[str]) -> tuple[float, float, int]:
    """
    Compute perplexity on a list of texts.

    Returns:
        perplexity: exp(avg_loss)
        avg_loss: average cross-entropy loss
        num_tokens: total tokens evaluated
    """
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        # Tokenize
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        input_ids = mx.array([tokens[:-1]])
        target_ids = mx.array([tokens[1:]])

        # Forward pass
        logits = model(input_ids)

        # Compute cross-entropy loss
        # logits: (1, seq_len, vocab_size)
        # target: (1, seq_len)
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        # Log softmax
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

        # Gather log probs for target tokens
        batch_size = targets_flat.shape[0]
        indices = mx.arange(batch_size)
        target_log_probs = log_probs[indices, targets_flat]

        # Sum negative log likelihood
        loss = -mx.sum(target_log_probs)
        total_loss += float(loss)
        total_tokens += len(tokens) - 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return perplexity, avg_loss, total_tokens


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    """Generate text continuation from a prompt."""
    from mlx_lm import generate

    try:
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return output
    except Exception as e:
        logger.warning(f"Generation failed for prompt: {prompt[:50]}... Error: {e}")
        return f"[ERROR: {e}]"


def evaluate_model(
    model,
    tokenizer,
    model_name: str,
    eval_prompts: list[str],
    perplexity_texts: list[str],
) -> EvaluationResult:
    """
    Run full evaluation on a model.

    Returns EvaluationResult with all metrics.
    """
    logger.info(f"Evaluating: {model_name}")

    # Count parameters
    params = count_parameters(model)
    logger.info(f"  Parameters: {params:,}")

    # Compute perplexity
    logger.info("  Computing perplexity...")
    start_time = time.time()
    perplexity, avg_loss, num_tokens = compute_perplexity(model, tokenizer, perplexity_texts)
    ppl_time = (time.time() - start_time) * 1000
    logger.info(f"  Perplexity: {perplexity:.2f} (loss: {avg_loss:.4f}, tokens: {num_tokens})")

    # Generate from prompts
    logger.info("  Generating from prompts...")
    generations = []
    start_time = time.time()
    for prompt in eval_prompts[:5]:  # Only first 5 for speed
        output = generate_text(model, tokenizer, prompt, max_tokens=30)
        generations.append(output)
    gen_time = (time.time() - start_time) * 1000 / len(generations[:5])

    return EvaluationResult(
        model_name=model_name,
        perplexity=perplexity,
        avg_loss=avg_loss,
        num_tokens=num_tokens,
        inference_time_ms=gen_time,
        params=params,
        generations=generations,
    )


def print_comparison(results: list[EvaluationResult]) -> None:
    """Print comparison table of results."""
    print()
    print("=" * 80)
    print("QUALITY EVALUATION RESULTS")
    print("=" * 80)
    print()

    # Summary table
    print(f"{'Model':<30} {'Params':>12} {'Perplexity':>12} {'Loss':>10} {'Time(ms)':>10}")
    print("-" * 80)

    baseline = results[0] if results else None

    for result in results:
        param_str = f"{result.params / 1e9:.2f}B"
        ppl_str = f"{result.perplexity:.2f}"
        loss_str = f"{result.avg_loss:.4f}"
        time_str = f"{result.inference_time_ms:.1f}"

        # Compare to baseline
        if baseline and result != baseline:
            ppl_ratio = result.perplexity / baseline.perplexity
            param_ratio = result.params / baseline.params
            ppl_str += f" ({ppl_ratio:.2f}x)"
            param_str += f" ({param_ratio:.2f}x)"

        print(f"{result.model_name:<30} {param_str:>12} {ppl_str:>12} {loss_str:>10} {time_str:>10}")

    print()

    # Quality verdict
    if len(results) >= 2:
        baseline = results[0]
        student = results[1]

        ppl_ratio = student.perplexity / baseline.perplexity
        param_ratio = student.params / baseline.params

        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print()
        print(f"Parameter reduction: {(1 - param_ratio) * 100:.1f}%")
        print(f"Perplexity increase: {(ppl_ratio - 1) * 100:.1f}%")
        print()

        if ppl_ratio <= 1.05:
            print("✅ EXCELLENT: <5% perplexity increase - ready to ship!")
        elif ppl_ratio <= 1.10:
            print("✅ GOOD: 5-10% perplexity increase - publishable with caveats")
        elif ppl_ratio <= 1.20:
            print("⚠️ MODERATE: 10-20% increase - needs more distillation")
        elif ppl_ratio <= 1.50:
            print("⚠️ CONCERNING: 20-50% increase - architecture revision needed")
        else:
            print("❌ FAILURE: >50% increase - fundamental issue")

        print()

        # Efficiency score
        if ppl_ratio > 0:
            efficiency = (1 - param_ratio) / (ppl_ratio - 1 + 0.01)  # Param reduction per quality loss
            print(f"Efficiency score: {efficiency:.2f} (higher is better)")
            print("  (Parameter reduction / Perplexity increase)")

    print()

    # Show sample generations
    print("=" * 80)
    print("SAMPLE GENERATIONS")
    print("=" * 80)
    print()

    for result in results:
        print(f"--- {result.model_name} ---")
        for i, gen in enumerate(result.generations[:3]):
            print(f"  [{i+1}] {gen[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model quality")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["openai/gpt-oss-20b"],
        help="Model IDs to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    results = []

    for model_id in args.models:
        try:
            model, tokenizer = load_model(model_id)
            result = evaluate_model(
                model,
                tokenizer,
                model_name=model_id,
                eval_prompts=EVAL_PROMPTS,
                perplexity_texts=PERPLEXITY_TEXTS,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_id}: {e}")

    # Print comparison
    print_comparison(results)

    # Save results
    output_path = Path(args.output)
    results_dict = [
        {
            "model_name": r.model_name,
            "perplexity": r.perplexity,
            "avg_loss": r.avg_loss,
            "num_tokens": r.num_tokens,
            "inference_time_ms": r.inference_time_ms,
            "params": r.params,
            "generations": r.generations,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
