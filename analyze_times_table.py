#!/usr/bin/env python3
"""
Deep analysis of extracted times table structure.
"""

import json
from collections import defaultdict

# Load the data
with open("times_table_structure.json") as f:
    data = json.load(f)

print("=" * 80)
print("TIMES TABLE MEMORY STRUCTURE ANALYSIS")
print("=" * 80)

# 1. Overall accuracy by position
print("\n1. ACCURACY BY OPERAND SIZE")
print("-" * 40)

accuracy_by_a = defaultdict(list)
accuracy_by_b = defaultdict(list)
accuracy_by_product = defaultdict(list)

for r in data:
    rank = r["neighborhood"]["correct_rank"]
    prob = r["neighborhood"]["correct_prob"] or 0
    is_top1 = rank == 1 if rank else False

    accuracy_by_a[r["a"]].append((is_top1, prob))
    accuracy_by_b[r["b"]].append((is_top1, prob))
    accuracy_by_product[r["correct"]].append((is_top1, prob))

print("\nBy first operand (a*_):")
for a in range(2, 10):
    items = accuracy_by_a[a]
    top1_rate = sum(1 for t, _ in items if t) / len(items)
    avg_prob = sum(p for _, p in items) / len(items)
    print(f"  {a}*_: top-1 rate={top1_rate:.1%}, avg_prob={avg_prob:.3f}")

print("\nBy second operand (_*b):")
for b in range(2, 10):
    items = accuracy_by_b[b]
    top1_rate = sum(1 for t, _ in items if t) / len(items)
    avg_prob = sum(p for _, p in items) / len(items)
    print(f"  _*{b}: top-1 rate={top1_rate:.1%}, avg_prob={avg_prob:.3f}")

# 2. Neighborhood structure
print("\n\n2. NEIGHBORHOOD STRUCTURE")
print("-" * 40)

print("\nFor each multiplication, what activates alongside it?")
print("(Row = same first operand, Col = same second operand)")

# Create a table showing for each problem: row/col bias
print(
    f"\n{'Problem':<10} {'Correct':<8} {'Rank':<6} {'Same Row':<10} {'Same Col':<10} {'Bias':<12}"
)
print("-" * 60)

row_bias_count = 0
col_bias_count = 0
neutral_count = 0

for r in data:
    a, b = r["a"], r["b"]
    correct = r["correct"]
    rank = r["neighborhood"]["correct_rank"] or "N/A"
    n_row = len(r["neighborhood"]["same_row"])
    n_col = len(r["neighborhood"]["same_col"])

    if n_row > n_col:
        bias = "ROW"
        row_bias_count += 1
    elif n_col > n_row:
        bias = "COL"
        col_bias_count += 1
    else:
        bias = "neutral"
        neutral_count += 1

    print(f"{a}*{b}=      {correct:<8} {str(rank):<6} {n_row:<10} {n_col:<10} {bias:<12}")

print(f"\nRow-biased: {row_bias_count}, Col-biased: {col_bias_count}, Neutral: {neutral_count}")

# 3. Squares analysis
print("\n\n3. SQUARES CLUSTER ANALYSIS")
print("-" * 40)

squares = [r for r in data if r["a"] == r["b"]]
print("\nDo squares (n*n) have special structure?")

for r in squares:
    n = r["a"]
    correct = r["correct"]
    rank = r["neighborhood"]["correct_rank"] or "N/A"
    prob = r["neighborhood"]["correct_prob"] or 0
    n_squares = len(r["neighborhood"]["squares"])

    # Get other squares that appear
    other_squares = [s["value"] for s in r["neighborhood"]["squares"]]

    print(f"  {n}*{n}={correct:>3}: rank={rank}, prob={prob:.3f}, other squares: {other_squares}")

# 4. Confusion matrix-style analysis
print("\n\n4. WHICH WRONG ANSWERS ARE MOST COMMON?")
print("-" * 40)

wrong_answer_counts = defaultdict(int)
wrong_answer_probs = defaultdict(list)

for r in data:
    correct = r["correct"]
    for cat in ["same_row", "same_col", "adjacent_products", "squares", "shared_factors"]:
        for item in r["neighborhood"][cat]:
            wrong_answer_counts[item["value"]] += 1
            wrong_answer_probs[item["value"]].append(item["prob"])

print("\nMost frequently activated wrong answers:")
sorted_wrongs = sorted(wrong_answer_counts.items(), key=lambda x: -x[1])[:20]
for val, count in sorted_wrongs:
    avg_prob = sum(wrong_answer_probs[val]) / len(wrong_answer_probs[val])
    print(f"  {val:>3}: appears {count} times, avg_prob={avg_prob:.3f}")

# 5. "Hardest" multiplications
print("\n\n5. HARDEST MULTIPLICATIONS (correct not in top-3)")
print("-" * 40)

hard = [
    r
    for r in data
    if r["neighborhood"]["correct_rank"] is None or r["neighborhood"]["correct_rank"] > 3
]
hard.sort(key=lambda x: x["neighborhood"]["correct_rank"] or 999)

for r in hard:
    a, b = r["a"], r["b"]
    correct = r["correct"]
    rank = r["neighborhood"]["correct_rank"] or "NOT IN TOP-30"
    prob = r["neighborhood"]["correct_prob"] or 0

    # What DID activate?
    top_wrong = []
    for cat in ["same_row", "same_col", "squares"]:
        for item in r["neighborhood"][cat][:2]:
            top_wrong.append(f"{item['value']}({item['prob']:.2f})")

    print(f"  {a}*{b}={correct}: rank={rank}, prob={prob:.3f}")
    print(f"    Top wrong: {', '.join(top_wrong[:5])}")

# 6. Product proximity analysis
print("\n\n6. PRODUCT PROXIMITY STRUCTURE")
print("-" * 40)

print("\nWhen the model is wrong, how close is the wrong answer to the correct one?")

diffs = []
for r in data:
    correct = r["correct"]
    for cat in ["same_row", "same_col"]:
        for item in r["neighborhood"][cat]:
            diff = abs(item["value"] - correct)
            diffs.append(diff)

# Histogram
diff_counts = defaultdict(int)
for d in diffs:
    bucket = (d // 5) * 5  # 5-unit buckets
    diff_counts[bucket] += 1

print("\nDistance from correct answer (buckets of 5):")
for bucket in sorted(diff_counts.keys()):
    count = diff_counts[bucket]
    bar = "#" * (count // 3)
    print(f"  {bucket:>3}-{bucket + 4:<3}: {count:>3} {bar}")

# 7. Asymmetry analysis
print("\n\n7. ASYMMETRY ANALYSIS (a*b vs b*a)")
print("-" * 40)

print("\nDoes the model treat a*b differently from b*a?")

for a in range(2, 9):
    for b in range(a + 1, 10):
        # Find both
        ab = next((r for r in data if r["a"] == a and r["b"] == b), None)
        ba = next((r for r in data if r["a"] == b and r["b"] == a), None)

        if ab and ba:
            rank_ab = ab["neighborhood"]["correct_rank"] or 99
            rank_ba = ba["neighborhood"]["correct_rank"] or 99
            prob_ab = ab["neighborhood"]["correct_prob"] or 0
            prob_ba = ba["neighborhood"]["correct_prob"] or 0

            if abs(rank_ab - rank_ba) > 2 or abs(prob_ab - prob_ba) > 0.1:
                print(f"  {a}*{b}: rank={rank_ab}, prob={prob_ab:.3f}")
                print(f"  {b}*{a}: rank={rank_ba}, prob={prob_ba:.3f}")
                print(f"    Δrank={rank_ab - rank_ba:+d}, Δprob={prob_ab - prob_ba:+.3f}")
                print()
