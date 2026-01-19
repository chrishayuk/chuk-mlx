#!/usr/bin/env python3
"""
Times Table Memory Structure Extraction

Systematically extract how the multiplication table is organized in
GPT-OSS-20B's internal representations by analyzing which "wrong" answers
activate alongside correct ones at Layer 20.

Research questions:
1. Is the neighborhood by row (same first operand)?
2. By column (same second operand)?
3. By product proximity (36, 42, 48...)?
4. By shared factors?
5. Do squares cluster together?
"""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def run_analysis(a: int, b: int, layer: int = 20, top_k: int = 30) -> dict:
    """Run lazarus introspect analyze for a*b= and capture top predictions."""
    prompt = f"{a}*{b}="
    correct = a * b

    cmd = [
        "uv",
        "run",
        "lazarus",
        "introspect",
        "analyze",
        "-m",
        "openai/gpt-oss-20b",
        "-p",
        prompt,
        "--layers",
        str(layer),
        "--top-k",
        str(top_k),
        "-n",
        "1",  # Minimal generation, we just want layer analysis
        "--raw",  # Skip chat template for cleaner analysis
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout

    # Parse the logit lens output for this layer
    # Looking for lines like: '56' (0.087)
    predictions = []

    # Find the layer section
    layer_pattern = rf"Layer {layer}:.*?(?=Layer \d+:|$)"
    layer_match = re.search(layer_pattern, output, re.DOTALL)

    if layer_match:
        layer_text = layer_match.group(0)
        # Match prediction lines: '56' (0.087) or similar
        pred_pattern = r"'(\d+)'\s*\(([0-9.]+)\)"
        for match in re.finditer(pred_pattern, layer_text):
            token = match.group(1)
            prob = float(match.group(2))
            try:
                value = int(token)
                predictions.append({"value": value, "prob": prob, "token": token})
            except ValueError:
                pass

    return {
        "prompt": prompt,
        "a": a,
        "b": b,
        "correct": correct,
        "layer": layer,
        "predictions": predictions,
    }


def analyze_neighborhood(data: dict) -> dict:
    """Analyze what's in the neighborhood of each multiplication."""
    a, b = data["a"], data["b"]
    correct = data["correct"]
    preds = data["predictions"]

    neighborhood = {
        "correct_rank": None,
        "correct_prob": None,
        "same_row": [],  # Same first operand (a*x for various x)
        "same_col": [],  # Same second operand (x*b for various x)
        "adjacent_products": [],  # Products close to correct answer
        "shared_factors": [],  # Products sharing a factor with correct
        "squares": [],  # Perfect squares
        "other": [],
    }

    # Build lookup of what products come from what operands
    product_sources = defaultdict(list)
    for i in range(2, 10):
        for j in range(2, 10):
            product_sources[i * j].append((i, j))

    for i, pred in enumerate(preds):
        val = pred["value"]
        prob = pred["prob"]

        if val == correct:
            neighborhood["correct_rank"] = i + 1
            neighborhood["correct_prob"] = prob
            continue

        # Check if this is a valid times table entry
        sources = product_sources.get(val, [])

        categorized = False

        # Check same row (a*x)
        for src_a, src_b in sources:
            if src_a == a or src_b == a:
                neighborhood["same_row"].append(
                    {"value": val, "prob": prob, "from": f"{src_a}*{src_b}"}
                )
                categorized = True
                break

        if not categorized:
            # Check same column (x*b)
            for src_a, src_b in sources:
                if src_a == b or src_b == b:
                    neighborhood["same_col"].append(
                        {"value": val, "prob": prob, "from": f"{src_a}*{src_b}"}
                    )
                    categorized = True
                    break

        if not categorized:
            # Check if it's a square
            sqrt = int(val**0.5)
            if sqrt * sqrt == val and 2 <= sqrt <= 9:
                neighborhood["squares"].append(
                    {"value": val, "prob": prob, "from": f"{sqrt}*{sqrt}"}
                )
                categorized = True

        if not categorized:
            # Check adjacent products (within ±10 of correct)
            if abs(val - correct) <= 10 and sources:
                neighborhood["adjacent_products"].append(
                    {"value": val, "prob": prob, "from": sources[0] if sources else "?"}
                )
                categorized = True

        if not categorized and sources:
            # Check shared factors
            correct_factors = set()
            if correct % 2 == 0:
                correct_factors.add(2)
            if correct % 3 == 0:
                correct_factors.add(3)
            if correct % 5 == 0:
                correct_factors.add(5)
            if correct % 7 == 0:
                correct_factors.add(7)

            val_factors = set()
            if val % 2 == 0:
                val_factors.add(2)
            if val % 3 == 0:
                val_factors.add(3)
            if val % 5 == 0:
                val_factors.add(5)
            if val % 7 == 0:
                val_factors.add(7)

            if correct_factors & val_factors:
                neighborhood["shared_factors"].append(
                    {
                        "value": val,
                        "prob": prob,
                        "from": f"{sources[0][0]}*{sources[0][1]}",
                        "shared": list(correct_factors & val_factors),
                    }
                )
                categorized = True

        if not categorized and sources:
            neighborhood["other"].append(
                {
                    "value": val,
                    "prob": prob,
                    "from": f"{sources[0][0]}*{sources[0][1]}" if sources else "?",
                }
            )

    return neighborhood


def main():
    results = []

    print("Extracting times table neighborhood structure from GPT-OSS-20B")
    print("=" * 70)

    # Run for all single-digit multiplications
    for a in range(2, 10):
        for b in range(2, 10):
            print(f"\nAnalyzing {a}*{b}={a * b}...")

            try:
                data = run_analysis(a, b, layer=20, top_k=30)
                neighborhood = analyze_neighborhood(data)

                result = {
                    **data,
                    "neighborhood": neighborhood,
                }
                results.append(result)

                # Print summary
                n = neighborhood
                print(
                    f"  Correct rank: {n['correct_rank']}, prob: {n['correct_prob']:.3f}"
                    if n["correct_prob"]
                    else "  Correct not in top-k!"
                )
                print(f"  Same row: {len(n['same_row'])}, Same col: {len(n['same_col'])}")
                print(f"  Adjacent: {len(n['adjacent_products'])}, Squares: {len(n['squares'])}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # Save raw results
    output_path = Path("times_table_structure.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nRaw results saved to: {output_path}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    # Count how often each category appears in top-k
    category_counts = defaultdict(int)
    category_probs = defaultdict(list)

    for r in results:
        n = r["neighborhood"]
        for cat in [
            "same_row",
            "same_col",
            "adjacent_products",
            "squares",
            "shared_factors",
            "other",
        ]:
            category_counts[cat] += len(n[cat])
            for item in n[cat]:
                category_probs[cat].append(item["prob"])

    print("\nNeighborhood composition (across all 64 multiplications):")
    for cat in ["same_row", "same_col", "adjacent_products", "squares", "shared_factors", "other"]:
        count = category_counts[cat]
        avg_prob = sum(category_probs[cat]) / len(category_probs[cat]) if category_probs[cat] else 0
        print(f"  {cat:20}: {count:4} occurrences, avg prob: {avg_prob:.4f}")

    # Find which multiplications have the most "confused" neighbors
    print("\nMost 'confused' multiplications (most high-prob wrong answers):")
    confusion_scores = []
    for r in results:
        n = r["neighborhood"]
        wrong_prob = sum(
            item["prob"]
            for cat in ["same_row", "same_col", "adjacent_products", "squares", "shared_factors"]
            for item in n[cat]
        )
        confusion_scores.append((r["prompt"], r["correct"], wrong_prob, n["correct_prob"] or 0))

    confusion_scores.sort(key=lambda x: -x[2])
    for prompt, correct, wrong_prob, correct_prob in confusion_scores[:10]:
        print(
            f"  {prompt:8} = {correct:3}  wrong_prob={wrong_prob:.3f}  correct_prob={correct_prob:.3f}"
        )


if __name__ == "__main__":
    main()
