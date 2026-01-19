"""
Number → Role Extraction from GSM8K

Hypothesis: Numbers have ROLES determined by local context.
- "16 eggs per day" → INITIAL or RATE
- "eats 3" → SUBTRACT_OPERAND
- "$2 each" → MULTIPLY_RATE

The model learns number→role, then composition is deterministic.
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path

import functools
print = functools.partial(print, flush=True)


def extract_numbers_with_context(text: str, window: int = 6) -> list[dict]:
    """Extract numbers and their surrounding context."""
    numbers = []
    words = text.split()

    # Find all numbers (including $X and X%)
    for i, word in enumerate(words):
        # Clean the word
        clean = word.strip('.,!?:;()[]')

        # Check if it's a number
        num_match = re.match(r'^\$?(\d+(?:\.\d+)?)\%?$', clean)
        if num_match:
            num_val = num_match.group(1)

            # Get context window
            start = max(0, i - window)
            end = min(len(words), i + window + 1)

            left_context = ' '.join(words[start:i])
            right_context = ' '.join(words[i+1:end])

            numbers.append({
                'value': num_val,
                'position': i,
                'word': word,
                'left_context': left_context.lower(),
                'right_context': right_context.lower(),
                'full_context': f"{left_context} [{word}] {right_context}".lower()
            })

    return numbers


def extract_computation_numbers(answer: str) -> list[dict]:
    """Extract numbers and their operations from <<expr=result>> annotations."""
    pattern = r'<<([^>]+)>>'
    matches = re.findall(pattern, answer)

    computation_numbers = []

    for match in matches:
        if '=' not in match:
            continue

        expr, result = match.rsplit('=', 1)
        expr = expr.strip()

        # Parse expression to find numbers and their roles
        # Simple patterns: a+b, a-b, a*b, a/b

        # Find operation
        if '+' in expr and '-' not in expr:
            op = 'ADD'
            parts = expr.split('+')
        elif '-' in expr and '+' not in expr:
            op = 'SUB'
            parts = expr.split('-')
        elif '*' in expr and '/' not in expr:
            op = 'MUL'
            parts = expr.split('*')
        elif '/' in expr and '*' not in expr:
            op = 'DIV'
            parts = expr.split('/')
        else:
            continue  # Skip complex expressions for now

        # Extract numbers from parts
        for i, part in enumerate(parts):
            part = part.strip()
            if re.match(r'^\d+(?:\.\d+)?$', part):
                role = f"{op}_LEFT" if i == 0 else f"{op}_RIGHT"
                computation_numbers.append({
                    'value': part,
                    'operation': op,
                    'role': role,
                    'expression': expr
                })

    return computation_numbers


def map_numbers_to_roles(question: str, answer: str) -> list[dict]:
    """Map numbers in question to their computational roles."""
    q_numbers = extract_numbers_with_context(question)
    comp_numbers = extract_computation_numbers(answer)

    # Try to match question numbers to computation numbers
    matched = []

    for q_num in q_numbers:
        q_val = q_num['value']

        # Find matching computation
        for c_num in comp_numbers:
            if c_num['value'] == q_val:
                matched.append({
                    'value': q_val,
                    'context': q_num['full_context'],
                    'left_context': q_num['left_context'],
                    'right_context': q_num['right_context'],
                    'role': c_num['role'],
                    'operation': c_num['operation'],
                    'expression': c_num['expression']
                })
                break  # Use first match

    return matched


def analyze_gsm8k_number_roles():
    """Analyze number→role patterns in GSM8K."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")
    print(f"Total: {len(ds)} examples")

    # Collect all number-role mappings
    all_mappings = []
    role_contexts = defaultdict(list)  # role → list of contexts

    context_features_by_role = defaultdict(Counter)  # role → feature counts

    for item in ds:
        mappings = map_numbers_to_roles(item["question"], item["answer"])
        all_mappings.extend(mappings)

        for m in mappings:
            role = m['role']
            ctx = m['context']
            role_contexts[role].append(ctx)

            # Extract features from context
            left = m['left_context']
            right = m['right_context']

            # Feature: words near the number
            for word in (left + ' ' + right).split():
                if len(word) > 2:  # Skip short words
                    context_features_by_role[role][word] += 1

    print(f"\nExtracted {len(all_mappings)} number→role mappings")

    # Role distribution
    print("\n" + "=" * 70)
    print("ROLE DISTRIBUTION")
    print("=" * 70)

    role_counts = Counter(m['role'] for m in all_mappings)
    for role, count in role_counts.most_common():
        print(f"  {role}: {count}")

    # Distinctive features per role
    print("\n" + "=" * 70)
    print("DISTINCTIVE CONTEXT FEATURES BY ROLE")
    print("=" * 70)

    # For each role, find words that appear more often than baseline
    total_by_word = Counter()
    for role, features in context_features_by_role.items():
        for word, count in features.items():
            total_by_word[word] += count

    for role in ['MUL_LEFT', 'MUL_RIGHT', 'SUB_LEFT', 'SUB_RIGHT', 'ADD_LEFT', 'ADD_RIGHT', 'DIV_LEFT', 'DIV_RIGHT']:
        if role not in context_features_by_role:
            continue

        print(f"\n{role}:")
        role_total = sum(context_features_by_role[role].values())

        # Find distinctive words (high lift)
        distinctive = []
        for word, count in context_features_by_role[role].items():
            if count < 10:
                continue
            expected = total_by_word[word] * role_total / len(all_mappings)
            if expected > 0:
                lift = count / expected
                if lift > 1.5:  # 50% more than expected
                    distinctive.append((word, count, lift))

        distinctive.sort(key=lambda x: -x[2])
        for word, count, lift in distinctive[:10]:
            print(f"  {word:15} count={count:4}  lift={lift:.1f}x")

    # Sample contexts by role
    print("\n" + "=" * 70)
    print("SAMPLE CONTEXTS BY ROLE")
    print("=" * 70)

    for role in ['MUL_RIGHT', 'SUB_RIGHT', 'ADD_RIGHT', 'DIV_RIGHT']:
        if role not in role_contexts:
            continue
        print(f"\n{role}:")
        for ctx in role_contexts[role][:5]:
            print(f"  {ctx[:80]}")

    # Key patterns
    print("\n" + "=" * 70)
    print("KEY PATTERN ANALYSIS")
    print("=" * 70)

    # Check: "each"/"per" context
    each_per_roles = Counter()
    for m in all_mappings:
        if 'each' in m['context'] or 'per' in m['context']:
            each_per_roles[m['role']] += 1

    print("\nNumbers near 'each'/'per':")
    for role, count in each_per_roles.most_common():
        print(f"  {role}: {count}")

    # Check: consumption verbs
    consume_roles = Counter()
    consume_words = ['eats', 'uses', 'spends', 'gives', 'loses', 'takes']
    for m in all_mappings:
        if any(w in m['context'] for w in consume_words):
            consume_roles[m['role']] += 1

    print("\nNumbers near consumption verbs (eats/uses/spends/gives/loses/takes):")
    for role, count in consume_roles.most_common():
        print(f"  {role}: {count}")

    # Check: position (first number in problem)
    first_number_roles = Counter()
    for item in ds:
        mappings = map_numbers_to_roles(item["question"], item["answer"])
        if mappings:
            first_number_roles[mappings[0]['role']] += 1

    print("\nFirst number in problem:")
    for role, count in first_number_roles.most_common():
        print(f"  {role}: {count}")

    # Save data for training
    output_path = Path(__file__).parent / "number_role_data.json"
    with open(output_path, 'w') as f:
        json.dump(all_mappings, f, indent=2)

    print(f"\nSaved {len(all_mappings)} mappings to {output_path}")

    return all_mappings


if __name__ == "__main__":
    analyze_gsm8k_number_roles()
