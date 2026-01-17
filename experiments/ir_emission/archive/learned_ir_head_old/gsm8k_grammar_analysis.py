"""
GSM8K Grammar Analysis

Goal: How many semantic patterns cover GSM8K?

Outcome → Implication:
- <50 patterns cover 80%  → Enumerate and win
- 50-200 patterns → Harder but doable
- Long tail, no structure → Need frontier-scale pretraining
"""

import re
import json
from collections import defaultdict, Counter
from pathlib import Path

import functools
print = functools.partial(print, flush=True)


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_computations(answer: str) -> list[dict]:
    """Extract <<expr=result>> from GSM8K answer."""
    pattern = r'<<([^>]+)>>'
    matches = re.findall(pattern, answer)

    computations = []
    for match in matches:
        if '=' in match:
            expr, result = match.rsplit('=', 1)
            # Classify operation
            expr_clean = expr.replace(" ", "")
            if '+' in expr_clean and '-' not in expr_clean:
                op = 'ADD'
            elif '-' in expr_clean and '+' not in expr_clean:
                op = 'SUB'
            elif '*' in expr_clean and '/' not in expr_clean:
                op = 'MUL'
            elif '/' in expr_clean and '*' not in expr_clean:
                op = 'DIV'
            else:
                op = 'MIXED'

            computations.append({
                "expr": expr.strip(),
                "result": result.strip(),
                "op": op
            })

    return computations


def extract_verb_context(text: str, window: int = 8) -> list[tuple]:
    """Extract verbs and their surrounding context."""
    # Common action verbs in math problems
    verb_patterns = [
        r'\b(eats?|ate)\b',
        r'\b(uses?|used)\b',
        r'\b(spends?|spent)\b',
        r'\b(gives?|gave)\b',
        r'\b(sells?|sold)\b',
        r'\b(buys?|bought)\b',
        r'\b(makes?|made)\b',
        r'\b(earns?|earned)\b',
        r'\b(gets?|got)\b',
        r'\b(finds?|found)\b',
        r'\b(loses?|lost)\b',
        r'\b(takes?|took)\b',
        r'\b(has|have|had)\b',
        r'\b(needs?|needed)\b',
        r'\b(wants?|wanted)\b',
        r'\b(collects?|collected)\b',
        r'\b(picks?|picked)\b',
        r'\b(reads?|read)\b',
        r'\b(works?|worked)\b',
        r'\b(walks?|walked)\b',
        r'\b(runs?|ran)\b',
        r'\b(drives?|drove)\b',
        r'\b(shares?|shared)\b',
        r'\b(divides?|divided)\b',
        r'\b(splits?|split)\b',
        r'\b(distributes?|distributed)\b',
        r'\b(saves?|saved)\b',
        r'\b(pays?|paid)\b',
        r'\b(receives?|received)\b',
        r'\b(produces?|produced)\b',
        r'\b(bakes?|baked)\b',
        r'\b(cooks?|cooked)\b',
        r'\b(plants?|planted)\b',
        r'\b(grows?|grew)\b',
        r'\b(drinks?|drank)\b',
        r'\b(writes?|wrote)\b',
    ]

    contexts = []
    text_lower = text.lower()
    words = text_lower.split()

    for pattern in verb_patterns:
        for match in re.finditer(pattern, text_lower):
            verb = match.group(1)
            start_pos = match.start()

            # Find word index
            word_idx = len(text_lower[:start_pos].split()) - 1

            # Get surrounding words
            start = max(0, word_idx - window)
            end = min(len(words), word_idx + window + 1)
            context = ' '.join(words[start:end])

            contexts.append((verb, context))

    return contexts


def extract_quantity_patterns(text: str) -> list[str]:
    """Extract quantity patterns like '$5 each', '3 times', 'half of'."""
    patterns = []

    # Money patterns
    for m in re.finditer(r'\$\d+(?:\.\d+)?\s*(?:each|per|every|a)', text, re.I):
        patterns.append(f"MONEY_RATE: {m.group()}")

    # "X times" patterns
    for m in re.finditer(r'\d+\s*times', text, re.I):
        patterns.append(f"TIMES: {m.group()}")

    # "twice/double/triple"
    for m in re.finditer(r'\b(twice|double|triple|half|third|quarter)\b', text, re.I):
        patterns.append(f"MULTIPLIER: {m.group()}")

    # "X per Y"
    for m in re.finditer(r'\d+\s*per\s+\w+', text, re.I):
        patterns.append(f"RATE: {m.group()}")

    # "each X"
    for m in re.finditer(r'each\s+\w+', text, re.I):
        patterns.append(f"EACH: {m.group()}")

    # "X of them"
    for m in re.finditer(r'\d+\s+of\s+them', text, re.I):
        patterns.append(f"PARTIAL: {m.group()}")

    return patterns


def normalize_verb(verb: str) -> str:
    """Normalize verb to base form."""
    mappings = {
        'eats': 'eat', 'ate': 'eat',
        'uses': 'use', 'used': 'use',
        'spends': 'spend', 'spent': 'spend',
        'gives': 'give', 'gave': 'give',
        'sells': 'sell', 'sold': 'sell',
        'buys': 'buy', 'bought': 'buy',
        'makes': 'make', 'made': 'make',
        'earns': 'earn', 'earned': 'earn',
        'gets': 'get', 'got': 'get',
        'finds': 'find', 'found': 'find',
        'loses': 'lose', 'lost': 'lose',
        'takes': 'take', 'took': 'take',
        'has': 'have', 'had': 'have',
        'needs': 'need', 'needed': 'need',
        'wants': 'want', 'wanted': 'want',
        'collects': 'collect', 'collected': 'collect',
        'picks': 'pick', 'picked': 'pick',
        'reads': 'read',
        'works': 'work', 'worked': 'work',
        'walks': 'walk', 'walked': 'walk',
        'runs': 'run', 'ran': 'run',
        'drives': 'drive', 'drove': 'drive',
        'shares': 'share', 'shared': 'share',
        'divides': 'divide', 'divided': 'divide',
        'splits': 'split',
        'distributes': 'distribute', 'distributed': 'distribute',
        'saves': 'save', 'saved': 'save',
        'pays': 'pay', 'paid': 'pay',
        'receives': 'receive', 'received': 'receive',
        'produces': 'produce', 'produced': 'produce',
        'bakes': 'bake', 'baked': 'bake',
        'cooks': 'cook', 'cooked': 'cook',
        'plants': 'plant', 'planted': 'plant',
        'grows': 'grow', 'grew': 'grow',
        'drinks': 'drink', 'drank': 'drink',
        'writes': 'write', 'wrote': 'write',
    }
    return mappings.get(verb, verb)


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_gsm8k():
    """Analyze GSM8K for semantic patterns."""
    from datasets import load_dataset

    print("Loading GSM8K train set...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"Total: {len(ds)} examples")

    # Collect data
    verb_to_op = defaultdict(Counter)  # verb -> {op: count}
    verb_context_to_op = defaultdict(Counter)  # (verb, context_pattern) -> {op: count}
    quantity_to_op = defaultdict(Counter)  # quantity_pattern -> {op: count}

    problem_signatures = Counter()  # (verb_tuple, op_tuple) -> count

    all_verbs = Counter()
    all_ops = Counter()

    for item in ds:
        question = item["question"]
        answer = item["answer"]

        # Extract computations
        comps = extract_computations(answer)
        if not comps:
            continue

        ops = tuple(c["op"] for c in comps)
        for c in comps:
            all_ops[c["op"]] += 1

        # Extract verbs
        verb_contexts = extract_verb_context(question)
        verbs_in_problem = []

        for verb, context in verb_contexts:
            norm_verb = normalize_verb(verb)
            all_verbs[norm_verb] += 1
            verbs_in_problem.append(norm_verb)

            # Map verb to operations in this problem
            for c in comps:
                verb_to_op[norm_verb][c["op"]] += 1

            # Check for specific context patterns
            if 'for breakfast' in context or 'for lunch' in context or 'for dinner' in context:
                verb_context_to_op[(norm_verb, 'for_meal')][comps[0]["op"]] += 1
            if 'each' in context or 'per' in context:
                verb_context_to_op[(norm_verb, 'rate')][comps[0]["op"]] += 1
            if 'half' in context or 'third' in context:
                verb_context_to_op[(norm_verb, 'fraction')][comps[0]["op"]] += 1

        # Extract quantity patterns
        qty_patterns = extract_quantity_patterns(question)
        for qp in qty_patterns:
            for c in comps:
                quantity_to_op[qp.split(':')[0]][c["op"]] += 1

        # Problem signature
        sig = (tuple(sorted(set(verbs_in_problem))), ops)
        problem_signatures[sig] += 1

    # ==========================================================================
    # REPORT
    # ==========================================================================

    print("\n" + "=" * 70)
    print("OPERATION DISTRIBUTION")
    print("=" * 70)
    total_ops = sum(all_ops.values())
    for op, count in all_ops.most_common():
        print(f"  {op}: {count} ({count/total_ops:.1%})")

    print("\n" + "=" * 70)
    print("TOP VERBS")
    print("=" * 70)
    for verb, count in all_verbs.most_common(30):
        ops_for_verb = verb_to_op[verb]
        dominant_op = ops_for_verb.most_common(1)[0] if ops_for_verb else ('?', 0)
        dominant_pct = dominant_op[1] / sum(ops_for_verb.values()) if ops_for_verb else 0
        ambiguity = "✓" if dominant_pct > 0.7 else "△" if dominant_pct > 0.5 else "✗"
        print(f"  {ambiguity} {verb:15} {count:5}  → {dominant_op[0]} ({dominant_pct:.0%})")

    print("\n" + "=" * 70)
    print("VERB → OPERATION MAPPING (high signal verbs)")
    print("=" * 70)

    # Find verbs with strong operation signals
    strong_signals = []
    ambiguous = []

    for verb, ops in verb_to_op.items():
        total = sum(ops.values())
        if total < 20:
            continue

        dominant = ops.most_common(1)[0]
        pct = dominant[1] / total

        if pct > 0.6:
            strong_signals.append((verb, dominant[0], pct, total))
        else:
            ambiguous.append((verb, dict(ops.most_common(3)), total))

    print("\nStrong signals (>60% dominant op):")
    for verb, op, pct, total in sorted(strong_signals, key=lambda x: -x[2]):
        print(f"  {verb:15} → {op:5} ({pct:.0%}, n={total})")

    print("\nAmbiguous verbs (no dominant op):")
    for verb, ops, total in sorted(ambiguous, key=lambda x: -x[2])[:15]:
        ops_str = ", ".join(f"{k}:{v}" for k, v in ops.items())
        print(f"  {verb:15} → {ops_str} (n={total})")

    print("\n" + "=" * 70)
    print("QUANTITY PATTERNS → OPERATION")
    print("=" * 70)
    for pattern, ops in sorted(quantity_to_op.items(), key=lambda x: -sum(x[1].values())):
        total = sum(ops.values())
        dominant = ops.most_common(1)[0]
        pct = dominant[1] / total
        print(f"  {pattern:20} → {dominant[0]:5} ({pct:.0%}, n={total})")

    print("\n" + "=" * 70)
    print("PROBLEM SIGNATURE ANALYSIS")
    print("=" * 70)

    # How many unique signatures?
    total_problems = sum(problem_signatures.values())
    unique_sigs = len(problem_signatures)

    print(f"\nUnique problem signatures: {unique_sigs}")
    print(f"Total problems: {total_problems}")

    # Coverage analysis
    sorted_sigs = problem_signatures.most_common()

    cumulative = 0
    coverage_milestones = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    milestone_idx = 0

    print("\nCoverage analysis:")
    for i, (sig, count) in enumerate(sorted_sigs):
        cumulative += count
        coverage = cumulative / total_problems

        while milestone_idx < len(coverage_milestones) and coverage >= coverage_milestones[milestone_idx]:
            print(f"  {coverage_milestones[milestone_idx]:.0%} coverage: {i+1} patterns")
            milestone_idx += 1

    print(f"  100% coverage: {unique_sigs} patterns")

    # Top signatures
    print("\nTop 20 problem signatures:")
    for sig, count in sorted_sigs[:20]:
        verbs, ops = sig
        pct = count / total_problems
        verbs_str = ','.join(verbs[:5]) if verbs else '(none)'
        ops_str = '→'.join(ops)
        print(f"  {count:4} ({pct:4.1%}): [{verbs_str}] → [{ops_str}]")

    # Save detailed analysis
    output = {
        "strong_verb_signals": [(v, o, p, t) for v, o, p, t in strong_signals],
        "ambiguous_verbs": [(v, o, t) for v, o, t in ambiguous],
        "quantity_patterns": {k: dict(v) for k, v in quantity_to_op.items()},
        "top_signatures": [(list(s[0]), list(s[1]), c) for s, c in sorted_sigs[:100]],
    }

    output_path = Path(__file__).parent / "gsm8k_grammar_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed analysis saved to {output_path}")

    return verb_to_op, problem_signatures


if __name__ == "__main__":
    analyze_gsm8k()
