"""Fact dataset generators and loaders.

This module provides structured fact datasets for memory analysis,
training data generation, and evaluation. Facts are returned as
lists of dictionaries with consistent schema.

Fact Schema:
    {
        "query": str,      # The prompt/question
        "answer": str,     # The expected answer
        "category": str,   # Primary category for grouping
        ...                # Additional fields vary by fact type
    }
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FactType(str, Enum):
    """Types of facts available."""

    MULTIPLICATION = "multiplication"
    ADDITION = "addition"
    CAPITALS = "capitals"
    ELEMENTS = "elements"
    CUSTOM = "custom"


def load_facts(fact_type: FactType | str) -> list[dict[str, Any]]:
    """Load facts by type.

    Args:
        fact_type: The type of facts to load.

    Returns:
        List of fact dictionaries.

    Raises:
        ValueError: If fact_type is unknown or CUSTOM (use file loading).
    """
    if isinstance(fact_type, str):
        fact_type = FactType(fact_type)

    if fact_type == FactType.MULTIPLICATION:
        return load_multiplication_facts()
    elif fact_type == FactType.ADDITION:
        return load_addition_facts()
    elif fact_type == FactType.CAPITALS:
        return load_capital_facts()
    elif fact_type == FactType.ELEMENTS:
        return load_element_facts()
    elif fact_type == FactType.CUSTOM:
        raise ValueError("CUSTOM facts must be loaded from file")
    else:
        raise ValueError(f"Unknown fact type: {fact_type}")


def load_multiplication_facts(
    min_operand: int = 2,
    max_operand: int = 9,
) -> list[dict[str, Any]]:
    """Generate multiplication facts.

    Args:
        min_operand: Minimum operand value (inclusive).
        max_operand: Maximum operand value (inclusive).

    Returns:
        List of multiplication facts with schema:
        {
            "query": "A*B=",
            "answer": "product",
            "operand_a": A,
            "operand_b": B,
            "category": "Ax",      # Row category
            "category_alt": "xB",  # Column category
        }
    """
    facts = []
    for a in range(min_operand, max_operand + 1):
        for b in range(min_operand, max_operand + 1):
            facts.append(
                {
                    "query": f"{a}*{b}=",
                    "answer": str(a * b),
                    "operand_a": a,
                    "operand_b": b,
                    "category": f"{a}x",
                    "category_alt": f"x{b}",
                }
            )
    return facts


def load_addition_facts(
    min_operand: int = 1,
    max_operand: int = 9,
) -> list[dict[str, Any]]:
    """Generate addition facts.

    Args:
        min_operand: Minimum operand value (inclusive).
        max_operand: Maximum operand value (inclusive).

    Returns:
        List of addition facts with schema:
        {
            "query": "A+B=",
            "answer": "sum",
            "operand_a": A,
            "operand_b": B,
            "category": "A+",
            "category_alt": "+B",
        }
    """
    facts = []
    for a in range(min_operand, max_operand + 1):
        for b in range(min_operand, max_operand + 1):
            facts.append(
                {
                    "query": f"{a}+{b}=",
                    "answer": str(a + b),
                    "operand_a": a,
                    "operand_b": b,
                    "category": f"{a}+",
                    "category_alt": f"+{b}",
                }
            )
    return facts


# Capital facts data - stored as module constant
_CAPITALS_DATA: list[tuple[str, str, str]] = [
    # (country, capital, region)
    ("France", "Paris", "Europe"),
    ("Germany", "Berlin", "Europe"),
    ("Italy", "Rome", "Europe"),
    ("Spain", "Madrid", "Europe"),
    ("United Kingdom", "London", "Europe"),
    ("Poland", "Warsaw", "Europe"),
    ("Netherlands", "Amsterdam", "Europe"),
    ("Belgium", "Brussels", "Europe"),
    ("Sweden", "Stockholm", "Europe"),
    ("Norway", "Oslo", "Europe"),
    ("Denmark", "Copenhagen", "Europe"),
    ("Finland", "Helsinki", "Europe"),
    ("Greece", "Athens", "Europe"),
    ("Portugal", "Lisbon", "Europe"),
    ("Austria", "Vienna", "Europe"),
    ("Switzerland", "Bern", "Europe"),
    ("Japan", "Tokyo", "Asia"),
    ("China", "Beijing", "Asia"),
    ("India", "New Delhi", "Asia"),
    ("South Korea", "Seoul", "Asia"),
    ("Thailand", "Bangkok", "Asia"),
    ("Vietnam", "Hanoi", "Asia"),
    ("Indonesia", "Jakarta", "Asia"),
    ("Malaysia", "Kuala Lumpur", "Asia"),
    ("Turkey", "Ankara", "Asia"),
    ("Iran", "Tehran", "Asia"),
    ("Iraq", "Baghdad", "Asia"),
    ("Saudi Arabia", "Riyadh", "Asia"),
    ("Israel", "Jerusalem", "Asia"),
    ("United States", "Washington D.C.", "Americas"),
    ("Canada", "Ottawa", "Americas"),
    ("Mexico", "Mexico City", "Americas"),
    ("Brazil", "Brasilia", "Americas"),
    ("Argentina", "Buenos Aires", "Americas"),
    ("Chile", "Santiago", "Americas"),
    ("Colombia", "Bogota", "Americas"),
    ("Peru", "Lima", "Americas"),
    ("Egypt", "Cairo", "Africa"),
    ("South Africa", "Pretoria", "Africa"),
    ("Nigeria", "Abuja", "Africa"),
    ("Kenya", "Nairobi", "Africa"),
    ("Morocco", "Rabat", "Africa"),
    ("Australia", "Canberra", "Oceania"),
    ("New Zealand", "Wellington", "Oceania"),
    ("Russia", "Moscow", "Europe"),
]


def load_capital_facts() -> list[dict[str, Any]]:
    """Load country capital facts.

    Returns:
        List of capital facts with schema:
        {
            "query": "The capital of {country} is",
            "answer": "capital",
            "country": "country name",
            "category": "region",
        }
    """
    facts = []
    for country, capital, region in _CAPITALS_DATA:
        facts.append(
            {
                "query": f"The capital of {country} is",
                "answer": capital,
                "country": country,
                "category": region,
            }
        )
    return facts


# Element facts data - stored as module constant
_ELEMENTS_DATA: list[tuple[int, str, str]] = [
    # (atomic_number, symbol, name)
    (1, "H", "Hydrogen"),
    (2, "He", "Helium"),
    (3, "Li", "Lithium"),
    (4, "Be", "Beryllium"),
    (5, "B", "Boron"),
    (6, "C", "Carbon"),
    (7, "N", "Nitrogen"),
    (8, "O", "Oxygen"),
    (9, "F", "Fluorine"),
    (10, "Ne", "Neon"),
    (11, "Na", "Sodium"),
    (12, "Mg", "Magnesium"),
    (13, "Al", "Aluminum"),
    (14, "Si", "Silicon"),
    (15, "P", "Phosphorus"),
    (16, "S", "Sulfur"),
    (17, "Cl", "Chlorine"),
    (18, "Ar", "Argon"),
    (19, "K", "Potassium"),
    (20, "Ca", "Calcium"),
    (21, "Sc", "Scandium"),
    (22, "Ti", "Titanium"),
    (23, "V", "Vanadium"),
    (24, "Cr", "Chromium"),
    (25, "Mn", "Manganese"),
    (26, "Fe", "Iron"),
    (27, "Co", "Cobalt"),
    (28, "Ni", "Nickel"),
    (29, "Cu", "Copper"),
    (30, "Zn", "Zinc"),
]


def load_element_facts() -> list[dict[str, Any]]:
    """Load periodic table element facts.

    Returns:
        List of element facts with schema:
        {
            "query": "Element {number} is",
            "answer": "element name",
            "number": atomic_number,
            "symbol": "symbol",
            "category": "Period N",
        }
    """
    facts = []
    for num, symbol, name in _ELEMENTS_DATA:
        # Determine period
        if num <= 2:
            period = 1
        elif num <= 10:
            period = 2
        elif num <= 18:
            period = 3
        else:
            period = 4

        facts.append(
            {
                "query": f"Element {num} is",
                "answer": name,
                "number": num,
                "symbol": symbol,
                "category": f"Period {period}",
            }
        )
    return facts


__all__ = [
    "FactType",
    "load_facts",
    "load_multiplication_facts",
    "load_addition_facts",
    "load_capital_facts",
    "load_element_facts",
]
