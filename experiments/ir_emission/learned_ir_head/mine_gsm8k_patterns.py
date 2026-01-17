"""
Mine GSM8K for verb→operation mappings.

Goal: Find what phrases in GSM8K map to what operations.
- "eats X for breakfast" → SUBTRACT
- "bakes with X" → SUBTRACT
- "sells at $X per" → MULTIPLY
"""

import sys
from pathlib import Path
import re
import json
from collections import defaultdict, Counter

import functools
print = functools.partial(print, flush=True)


def extract_computations(answer_text: str) -> list[dict]:
    """Extract <<expr=result>> computations from GSM8K answer."""
    pattern = r'<<([^>]+)>>'
    matches = re.findall(pattern, answer_text)

    computations = []
    for match in matches:
        if '=' in match:
            expr, result = match.rsplit('=', 1)
            computations.append({
                "expr": expr.strip(),
                "result": result.strip()
            })

    return computations


def classify_operation(expr: str) -> str:
    """Classify an expression by its primary operation."""
    # Remove spaces
    expr = expr.replace(" ", "")

    # Count operations
    ops = {'+': expr.count('+'), '-': expr.count('-'),
           '*': expr.count('*'), '/': expr.count('/')}

    # Find dominant operation
    if ops['+'] > 0 and ops['-'] == 0:
        return 'ADD'
    elif ops['-'] > 0 and ops['+'] == 0:
        return 'SUBTRACT'
    elif ops['*'] > 0 and ops['/'] == 0:
        return 'MULTIPLY'
    elif ops['/'] > 0 and ops['*'] == 0:
        return 'DIVIDE'
    elif ops['+'] > 0 or ops['-'] > 0:
        return 'ADD_SUB'  # mixed add/subtract
    elif ops['*'] > 0 or ops['/'] > 0:
        return 'MUL_DIV'  # mixed multiply/divide
    else:
        return 'UNKNOWN'


def find_context_for_number(question: str, number: str, window: int = 50) -> list[str]:
    """Find text context around a number in the question."""
    contexts = []

    # Handle different number formats
    patterns = [
        rf'\b{re.escape(number)}\b',  # exact match
        rf'\${re.escape(number)}',     # $X format
        rf'{re.escape(number)}%',      # X% format
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(question), match.end() + window)
            context = question[start:end]
            contexts.append(context)

    return contexts


def extract_verb_patterns(question: str, answer: str) -> list[dict]:
    """Extract verb→operation patterns from a single example."""
    computations = extract_computations(answer)
    patterns = []

    for comp in computations:
        expr = comp["expr"]
        op = classify_operation(expr)

        # Extract operands from expression
        operands = re.findall(r'\d+\.?\d*', expr)

        # Find context for each operand in the question
        for operand in operands:
            contexts = find_context_for_number(question, operand)
            for ctx in contexts:
                patterns.append({
                    "operation": op,
                    "operand": operand,
                    "context": ctx.lower(),
                    "expr": expr
                })

    return patterns


def extract_verbs_near_number(context: str) -> list[str]:
    """Extract verbs that appear near a number."""
    # Common verb patterns that indicate operations
    verb_patterns = [
        r'(\w+s)\s+\d',           # verb ending in 's' before number
        r'(\w+ed)\s+\d',          # past tense before number
        r'(\w+ing)\s+\d',         # gerund before number
        r'\d+\s+(\w+ed)',         # past tense after number
        r'(\w+)\s+\$?\d',         # any word before number
        r'for\s+(\w+)',           # "for X"
        r'to\s+(\w+)',            # "to X"
    ]

    verbs = []
    for pattern in verb_patterns:
        matches = re.findall(pattern, context.lower())
        verbs.extend(matches)

    return verbs


def mine_gsm8k():
    """Mine GSM8K for semantic patterns."""
    from datasets import load_dataset

    print("Loading GSM8K train set...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"Total examples: {len(ds)}")

    # Collect patterns by operation
    patterns_by_op = defaultdict(list)
    verb_by_op = defaultdict(Counter)
    phrase_by_op = defaultdict(Counter)

    # Common semantic patterns to look for
    semantic_markers = {
        'ADD': ['gets', 'receives', 'finds', 'earns', 'gains', 'adds', 'more', 'additional', 'plus', 'increases', 'collects', 'gathers'],
        'SUBTRACT': ['eats', 'uses', 'spends', 'loses', 'gives', 'removes', 'takes', 'sells', 'drops', 'breaks', 'wastes', 'consumes', 'decreases', 'leaves', 'remaining'],
        'MULTIPLY': ['each', 'per', 'every', 'times', 'twice', 'double', 'triple', 'groups', 'rows', 'sets', 'packs', 'boxes', 'bags', 'daily', 'weekly', 'hourly'],
        'DIVIDE': ['split', 'divide', 'share', 'distribute', 'among', 'between', 'half', 'third', 'quarter', 'equally', 'average'],
    }

    for item in ds:
        question = item["question"].lower()
        answer = item["answer"]

        patterns = extract_verb_patterns(item["question"], answer)

        for p in patterns:
            op = p["operation"]
            ctx = p["context"]

            patterns_by_op[op].append(p)

            # Count semantic markers
            for marker in semantic_markers.get(op, []):
                if marker in ctx:
                    phrase_by_op[op][marker] += 1

            # Also check cross-contamination (markers in wrong op)
            for other_op, markers in semantic_markers.items():
                if other_op != op:
                    for marker in markers:
                        if marker in ctx:
                            phrase_by_op[f"{op}_has_{other_op}_marker"][marker] += 1

    # Report findings
    print("\n" + "=" * 70)
    print("OPERATION DISTRIBUTION")
    print("=" * 70)

    for op in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'ADD_SUB', 'MUL_DIV']:
        count = len(patterns_by_op[op])
        pct = count / sum(len(v) for v in patterns_by_op.values()) * 100
        print(f"  {op}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("SEMANTIC MARKERS BY OPERATION")
    print("=" * 70)

    for op in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']:
        print(f"\n{op}:")
        top_markers = phrase_by_op[op].most_common(15)
        for marker, count in top_markers:
            print(f"  {marker}: {count}")

    print("\n" + "=" * 70)
    print("CROSS-CONTAMINATION (markers in unexpected operations)")
    print("=" * 70)

    for key in sorted(phrase_by_op.keys()):
        if '_has_' in key:
            top = phrase_by_op[key].most_common(5)
            if top:
                print(f"\n{key}:")
                for marker, count in top:
                    print(f"  {marker}: {count}")

    # Sample contexts
    print("\n" + "=" * 70)
    print("SAMPLE CONTEXTS BY OPERATION")
    print("=" * 70)

    for op in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']:
        print(f"\n{op} examples:")
        samples = patterns_by_op[op][:10]
        for s in samples:
            ctx_short = s['context'][:80].replace('\n', ' ')
            print(f"  {s['expr']:20} | ...{ctx_short}...")

    # Save detailed patterns
    output = {
        "semantic_markers": {op: dict(phrase_by_op[op].most_common(30)) for op in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']},
        "cross_contamination": {k: dict(v.most_common(10)) for k, v in phrase_by_op.items() if '_has_' in k},
        "sample_patterns": {op: patterns_by_op[op][:50] for op in ['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE']}
    }

    output_path = Path(__file__).parent / "gsm8k_semantic_patterns.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed patterns saved to {output_path}")

    return patterns_by_op, phrase_by_op


if __name__ == "__main__":
    patterns_by_op, phrase_by_op = mine_gsm8k()
