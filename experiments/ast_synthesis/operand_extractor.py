"""
Context-Aware Operand Extractor

The naive regex approach extracts ALL numbers from text, including numbers
that are part of phrases like "reach 1 from" or "base 10".

This module implements context-aware extraction that:
1. Identifies phrase patterns with embedded numbers
2. Filters out non-operand numbers
3. Uses positional heuristics for ambiguous cases
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# PHRASE PATTERNS TO FILTER
# =============================================================================

# These patterns contain numbers that are NOT operands
# Format: (pattern_regex, description)
NON_OPERAND_PATTERNS = [
    # Collatz-specific: "reach 1" means reach the value 1
    (r'\breach\s+1\s+from\b', "reach 1 from"),
    (r'\buntil\s+1\b', "until 1"),
    (r'\bto\s+1\s+from\b', "to 1 from"),
    (r'\bconverge\s+to\s+1\b', "converge to 1"),

    # Range patterns where lower bound is implicit start (1)
    # "from 1 to N" - the 1 is the start, N is the operand
    (r'\bfrom\s+1\s+to\b', "from 1 to"),
    (r'\b1\s+to\s+\d', "1 to N"),  # "1 to 100"
    (r'\bnumbers\s+1\s+to\b', "numbers 1 to"),

    # Mathematical constants
    (r'\bbase\s+10\b', "base 10"),
    (r'\bbase\s+2\b', "base 2"),
    (r'\bmod\s+2\b', "mod 2"),
    (r'\bmodulo\s+2\b', "modulo 2"),

    # Ordinal patterns (1st, 2nd, etc. are usually not operands)
    (r'\b1st\b', "1st"),
    (r'\b2nd\b', "2nd"),
    (r'\b3rd\b', "3rd"),
]


# =============================================================================
# OPERAND EXTRACTION STRATEGIES
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of operand extraction."""
    operands: List[int]
    method: str  # Which method was used
    confidence: float  # 0-1 confidence score


def extract_numbers_with_positions(text: str) -> List[Tuple[int, int, int]]:
    """
    Extract all numbers with their positions.

    Returns: List of (value, start_pos, end_pos)
    """
    results = []
    for match in re.finditer(r'\b(\d+)\b', text):
        value = int(match.group(1))
        start = match.start()
        end = match.end()
        results.append((value, start, end))
    return results


def filter_non_operand_numbers(text: str, numbers: List[Tuple[int, int, int]]) -> List[int]:
    """
    Filter out numbers that are part of known non-operand phrases.
    """
    text_lower = text.lower()

    # Find positions of non-operand patterns
    non_operand_ranges = []
    for pattern, desc in NON_OPERAND_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            non_operand_ranges.append((match.start(), match.end()))

    # Filter numbers that fall within non-operand ranges
    operands = []
    for value, start, end in numbers:
        is_non_operand = False
        for range_start, range_end in non_operand_ranges:
            if range_start <= start and end <= range_end:
                is_non_operand = True
                break

        if not is_non_operand:
            operands.append(value)

    return operands


def extract_by_position(text: str) -> List[int]:
    """
    Extract operands using positional heuristics.

    Heuristics:
    - Operands usually appear after keywords like "of", "for", "from", "to"
    - Operands are often at the end of the phrase
    - Multiple operands are usually separated by "and", ",", "to"
    """
    # Keywords that typically precede operands
    operand_keywords = [
        r'\bof\s+(\d+)\b',
        r'\bfor\s+(\d+)\b',
        r'\bfrom\s+(\d+)\b',
        r'\bstarting\s+at\s+(\d+)\b',
        r'\b(\d+)\s+and\s+(\d+)\b',
        r'\b(\d+)\s+to\s+(\d+)\b',  # But filter "1 to N" patterns
        r'\((\d+)\)',  # Numbers in parentheses
        r'\b(\d+)[!^]',  # Factorial or power notation
    ]

    operands = []
    text_lower = text.lower()

    for pattern in operand_keywords:
        for match in re.finditer(pattern, text_lower):
            for group in match.groups():
                if group and group.isdigit():
                    val = int(group)
                    if val not in operands:
                        operands.append(val)

    return operands


def extract_last_numbers(text: str, n: int = 2) -> List[int]:
    """
    Extract the last N numbers from text.

    Often operands appear at the end: "Collatz length of 27"
    """
    numbers = extract_numbers_with_positions(text)
    if not numbers:
        return []

    # Sort by position (should already be sorted, but ensure)
    numbers.sort(key=lambda x: x[1])

    # Take last N
    return [val for val, _, _ in numbers[-n:]]


# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

def extract_operands(text: str, expected_count: Optional[int] = None) -> List[int]:
    """
    Extract operands from natural language text using context-aware parsing.

    Args:
        text: Natural language input
        expected_count: Optional hint for how many operands to expect

    Returns:
        List of extracted operand values
    """
    # Step 1: Extract all numbers with positions
    all_numbers = extract_numbers_with_positions(text)

    if not all_numbers:
        return []

    # Step 2: Filter non-operand numbers
    filtered = filter_non_operand_numbers(text, all_numbers)

    # Step 3: If filtering removed all numbers, fall back to positional
    if not filtered:
        # Use last number as fallback
        filtered = extract_last_numbers(text, n=expected_count or 1)

    # Step 4: If we have expected count, trim to that
    if expected_count and len(filtered) > expected_count:
        # Take the last N (operands usually at end)
        filtered = filtered[-expected_count:]

    return filtered


def extract_operands_for_template(text: str, template_name: str) -> List[int]:
    """
    Extract operands with template-specific knowledge.

    Different templates expect different numbers of operands:
    - LOOP_CONDITIONAL_ACCUMULATE (collatz, sum_even): 1 operand
    - LOOP_ACCUMULATE (sum, factorial, power): 1-2 operands
    - IF_BRANCH (max, abs_diff): 2 operands
    """
    expected_counts = {
        "collatz_length": 1,
        "sum_even": 1,
        "sum_1_to_n": 1,
        "sum_a_to_b": 2,
        "factorial": 1,
        "power": 2,
        "max_of_two": 2,
        "abs_diff": 2,
    }

    expected = expected_counts.get(template_name)
    return extract_operands(text, expected_count=expected)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def extract_operands_naive(text: str) -> List[int]:
    """Original naive extraction for comparison."""
    numbers = re.findall(r'\b(\d+)\b', text)
    return [int(n) for n in numbers]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Context-Aware Operand Extractor Test")
    print("=" * 70)

    test_cases = [
        # (text, expected_operands)
        ("Collatz length of 27", [27]),
        ("Steps to reach 1 from 355 via Collatz", [355]),
        ("Steps to reach 1 from 861 via Collatz", [861]),
        ("Sum of even numbers from 1 to 100", [100]),
        ("Sum 5 to 100", [5, 100]),
        ("5 factorial", [5]),
        ("2 to the power of 10", [2, 10]),
        ("Max of 5 and 3", [5, 3]),
        ("How many Collatz steps for 877?", [877]),
        ("Count Collatz iterations for 683", [683]),
        ("Collatz(308) length", [308]),
    ]

    print("\nComparison: Naive vs Context-Aware")
    print("-" * 70)

    naive_correct = 0
    context_correct = 0

    for text, expected in test_cases:
        naive = extract_operands_naive(text)
        context = extract_operands(text)

        naive_ok = naive == expected
        context_ok = context == expected

        if naive_ok:
            naive_correct += 1
        if context_ok:
            context_correct += 1

        status = "OK" if context_ok else "FIXED" if not naive_ok and context_ok else "WRONG"

        print(f"\n[{status}] \"{text}\"")
        print(f"  Expected: {expected}")
        print(f"  Naive:    {naive} {'✓' if naive_ok else '✗'}")
        print(f"  Context:  {context} {'✓' if context_ok else '✗'}")

    print("\n" + "=" * 70)
    print(f"Results: Naive {naive_correct}/{len(test_cases)}, Context {context_correct}/{len(test_cases)}")
    print("=" * 70)
