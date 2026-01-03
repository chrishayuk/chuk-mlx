"""Tests for facts Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.enums import FactType, Region
from chuk_lazarus.introspection.models.facts import (
    CapitalFact,
    ElementFact,
    Fact,
    FactNeighborhood,
    FactSet,
    MathFact,
)


class TestFact:
    """Tests for Fact base model."""

    def test_instantiation_minimal(self):
        """Test creating fact with minimal required fields."""
        fact = Fact(query="What is 2+2?", answer="4")
        assert fact.query == "What is 2+2?"
        assert fact.answer == "4"
        assert fact.category == ""
        assert fact.category_alt is None

    def test_instantiation_with_all_fields(self):
        """Test creating fact with all fields."""
        fact = Fact(
            query="What is 2+2?",
            answer="4",
            category="arithmetic",
            category_alt="addition",
        )
        assert fact.category == "arithmetic"
        assert fact.category_alt == "addition"

    def test_fact_type_property(self):
        """Test fact_type property returns CUSTOM for base Fact."""
        fact = Fact(query="test", answer="test")
        assert fact.fact_type == FactType.CUSTOM

    def test_default_values(self):
        """Test default values for optional fields."""
        fact = Fact(query="q", answer="a")
        assert fact.category == ""
        assert fact.category_alt is None


class TestMathFact:
    """Tests for MathFact model."""

    def test_instantiation_minimal(self):
        """Test creating math fact with required fields."""
        fact = MathFact(
            query="2 * 3 = ",
            answer="6",
            operand_a=2,
            operand_b=3,
            operator="*",
        )
        assert fact.query == "2 * 3 = "
        assert fact.answer == "6"
        assert fact.operand_a == 2
        assert fact.operand_b == 3
        assert fact.operator == "*"

    def test_instantiation_with_categories(self):
        """Test creating math fact with categories."""
        fact = MathFact(
            query="2 * 3 = ",
            answer="6",
            operand_a=2,
            operand_b=3,
            operator="*",
            category="2x",
            category_alt="x3",
        )
        assert fact.category == "2x"
        assert fact.category_alt == "x3"

    def test_fact_type_multiplication(self):
        """Test fact_type returns MULTIPLICATION for * operator."""
        fact = MathFact(
            query="2 * 3 = ",
            answer="6",
            operand_a=2,
            operand_b=3,
            operator="*",
        )
        assert fact.fact_type == FactType.MULTIPLICATION

    def test_fact_type_multiplication_with_x(self):
        """Test fact_type returns MULTIPLICATION for x operator."""
        fact = MathFact(
            query="2 x 3 = ",
            answer="6",
            operand_a=2,
            operand_b=3,
            operator="x",
        )
        assert fact.fact_type == FactType.MULTIPLICATION

    def test_fact_type_multiplication_with_unicode(self):
        """Test fact_type returns MULTIPLICATION for × operator."""
        fact = MathFact(
            query="2 × 3 = ",
            answer="6",
            operand_a=2,
            operand_b=3,
            operator="×",
        )
        assert fact.fact_type == FactType.MULTIPLICATION

    def test_fact_type_addition(self):
        """Test fact_type returns ADDITION for + operator."""
        fact = MathFact(
            query="2 + 3 = ",
            answer="5",
            operand_a=2,
            operand_b=3,
            operator="+",
        )
        assert fact.fact_type == FactType.ADDITION

    def test_fact_type_custom_for_other_operators(self):
        """Test fact_type returns CUSTOM for unsupported operators."""
        fact = MathFact(
            query="10 - 3 = ",
            answer="7",
            operand_a=10,
            operand_b=3,
            operator="-",
        )
        assert fact.fact_type == FactType.CUSTOM


class TestCapitalFact:
    """Tests for CapitalFact model."""

    def test_instantiation_minimal(self):
        """Test creating capital fact with required fields."""
        fact = CapitalFact(
            query="The capital of France is",
            answer="Paris",
            country="France",
        )
        assert fact.query == "The capital of France is"
        assert fact.answer == "Paris"
        assert fact.country == "France"
        assert fact.region == Region.OTHER

    def test_instantiation_with_region(self):
        """Test creating capital fact with region."""
        fact = CapitalFact(
            query="The capital of France is",
            answer="Paris",
            country="France",
            region=Region.EUROPE,
        )
        assert fact.region == Region.EUROPE

    def test_fact_type_property(self):
        """Test fact_type property returns CAPITALS."""
        fact = CapitalFact(
            query="The capital of Japan is",
            answer="Tokyo",
            country="Japan",
        )
        assert fact.fact_type == FactType.CAPITALS

    def test_default_region(self):
        """Test default region is OTHER."""
        fact = CapitalFact(
            query="test",
            answer="test",
            country="test",
        )
        assert fact.region == Region.OTHER


class TestElementFact:
    """Tests for ElementFact model."""

    def test_instantiation_minimal(self):
        """Test creating element fact with required fields."""
        fact = ElementFact(
            query="Element 1 is",
            answer="Hydrogen",
            atomic_number=1,
            symbol="H",
            period=1,
        )
        assert fact.query == "Element 1 is"
        assert fact.answer == "Hydrogen"
        assert fact.atomic_number == 1
        assert fact.symbol == "H"
        assert fact.period == 1

    def test_fact_type_property(self):
        """Test fact_type property returns ELEMENTS."""
        fact = ElementFact(
            query="Element 6 is",
            answer="Carbon",
            atomic_number=6,
            symbol="C",
            period=2,
        )
        assert fact.fact_type == FactType.ELEMENTS

    def test_with_category(self):
        """Test creating element fact with category."""
        fact = ElementFact(
            query="Element 1 is",
            answer="Hydrogen",
            atomic_number=1,
            symbol="H",
            period=1,
            category="Period 1",
        )
        assert fact.category == "Period 1"


class TestFactNeighborhood:
    """Tests for FactNeighborhood model."""

    def test_instantiation_with_defaults(self):
        """Test creating neighborhood with default values."""
        neighborhood = FactNeighborhood()
        assert neighborhood.correct_rank is None
        assert neighborhood.correct_prob is None
        assert neighborhood.same_category == []
        assert neighborhood.same_category_alt == []
        assert neighborhood.other_answers == []
        assert neighborhood.non_answers == []

    def test_instantiation_with_values(self):
        """Test creating neighborhood with specific values."""
        neighborhood = FactNeighborhood(
            correct_rank=1,
            correct_prob=0.95,
            same_category=[{"token": "6", "prob": 0.8}],
            same_category_alt=[{"token": "8", "prob": 0.6}],
            other_answers=[{"token": "9", "prob": 0.4}],
            non_answers=[{"token": "hello", "prob": 0.1}],
        )
        assert neighborhood.correct_rank == 1
        assert neighborhood.correct_prob == 0.95
        assert len(neighborhood.same_category) == 1
        assert len(neighborhood.same_category_alt) == 1
        assert len(neighborhood.other_answers) == 1
        assert len(neighborhood.non_answers) == 1


class TestFactSet:
    """Tests for FactSet model."""

    def test_instantiation_minimal(self):
        """Test creating fact set with minimal fields."""
        fact_set = FactSet(fact_type=FactType.MULTIPLICATION)
        assert fact_set.fact_type == FactType.MULTIPLICATION
        assert fact_set.facts == []

    def test_instantiation_with_facts(self):
        """Test creating fact set with facts."""
        facts = [
            MathFact(query="2*3=", answer="6", operand_a=2, operand_b=3, operator="*"),
            MathFact(query="4*5=", answer="20", operand_a=4, operand_b=5, operator="*"),
        ]
        fact_set = FactSet(fact_type=FactType.MULTIPLICATION, facts=facts)
        assert len(fact_set.facts) == 2

    def test_multiplication_table_default(self):
        """Test generating default multiplication table."""
        fact_set = FactSet.multiplication_table()
        assert fact_set.fact_type == FactType.MULTIPLICATION
        # Default range is 2-9, so 8*8 = 64 facts
        assert len(fact_set.facts) == 64

    def test_multiplication_table_custom_range(self):
        """Test generating multiplication table with custom range."""
        fact_set = FactSet.multiplication_table(start=2, end=4)
        assert len(fact_set.facts) == 9  # 3*3
        # Check first and last
        assert fact_set.facts[0].operand_a == 2
        assert fact_set.facts[0].operand_b == 2
        assert fact_set.facts[-1].operand_a == 4
        assert fact_set.facts[-1].operand_b == 4

    def test_multiplication_table_facts_correct(self):
        """Test multiplication table generates correct facts."""
        fact_set = FactSet.multiplication_table(start=2, end=3)
        # Should have 2*2, 2*3, 3*2, 3*3
        fact_2_3 = next(f for f in fact_set.facts if f.operand_a == 2 and f.operand_b == 3)
        assert fact_2_3.query == "2*3="
        assert fact_2_3.answer == "6"
        assert fact_2_3.operator == "*"

    def test_multiplication_table_categories(self):
        """Test multiplication table sets categories correctly."""
        fact_set = FactSet.multiplication_table(start=2, end=3)
        fact_2_3 = next(f for f in fact_set.facts if f.operand_a == 2 and f.operand_b == 3)
        assert fact_2_3.category == "2x"
        assert fact_2_3.category_alt == "x3"

    def test_addition_table_default(self):
        """Test generating default addition table."""
        fact_set = FactSet.addition_table()
        assert fact_set.fact_type == FactType.ADDITION
        # Default range is 1-9, so 9*9 = 81 facts
        assert len(fact_set.facts) == 81

    def test_addition_table_custom_range(self):
        """Test generating addition table with custom range."""
        fact_set = FactSet.addition_table(start=1, end=3)
        assert len(fact_set.facts) == 9  # 3*3

    def test_addition_table_facts_correct(self):
        """Test addition table generates correct facts."""
        fact_set = FactSet.addition_table(start=2, end=3)
        fact_2_3 = next(f for f in fact_set.facts if f.operand_a == 2 and f.operand_b == 3)
        assert fact_2_3.query == "2+3="
        assert fact_2_3.answer == "5"
        assert fact_2_3.operator == "+"

    def test_addition_table_categories(self):
        """Test addition table sets categories correctly."""
        fact_set = FactSet.addition_table(start=2, end=3)
        fact_2_3 = next(f for f in fact_set.facts if f.operand_a == 2 and f.operand_b == 3)
        assert fact_2_3.category == "2+"
        assert fact_2_3.category_alt == "+3"

    def test_world_capitals(self):
        """Test generating world capitals fact set."""
        fact_set = FactSet.world_capitals()
        assert fact_set.fact_type == FactType.CAPITALS
        assert len(fact_set.facts) > 0
        # Check that all facts are CapitalFact instances
        assert all(isinstance(f, CapitalFact) for f in fact_set.facts)

    def test_world_capitals_contains_expected(self):
        """Test world capitals contains expected entries."""
        fact_set = FactSet.world_capitals()
        # Check for France
        france = next((f for f in fact_set.facts if f.country == "France"), None)
        assert france is not None
        assert france.answer == "Paris"
        assert france.region == Region.EUROPE

    def test_world_capitals_regions(self):
        """Test world capitals have proper regions."""
        fact_set = FactSet.world_capitals()
        # Check various regions are present
        regions = {f.region for f in fact_set.facts}
        assert Region.EUROPE in regions
        assert Region.ASIA in regions
        assert Region.AMERICAS in regions

    def test_periodic_elements_default(self):
        """Test generating periodic elements with default max."""
        fact_set = FactSet.periodic_elements()
        assert fact_set.fact_type == FactType.ELEMENTS
        assert len(fact_set.facts) == 20  # Default max is 20
        assert all(isinstance(f, ElementFact) for f in fact_set.facts)

    def test_periodic_elements_custom_max(self):
        """Test generating periodic elements with custom max."""
        fact_set = FactSet.periodic_elements(max_number=10)
        assert len(fact_set.facts) == 10

    def test_periodic_elements_facts_correct(self):
        """Test periodic elements generates correct facts."""
        fact_set = FactSet.periodic_elements(max_number=5)
        # Check hydrogen
        hydrogen = fact_set.facts[0]
        assert hydrogen.atomic_number == 1
        assert hydrogen.symbol == "H"
        assert hydrogen.answer == "Hydrogen"
        assert hydrogen.period == 1

    def test_periodic_elements_periods(self):
        """Test periodic elements assigns correct periods."""
        fact_set = FactSet.periodic_elements(max_number=20)
        # Element 1-2 should be period 1
        assert fact_set.facts[0].period == 1  # H
        assert fact_set.facts[1].period == 1  # He
        # Element 3-10 should be period 2
        assert fact_set.facts[2].period == 2  # Li
        # Element 11-20 should be period 3
        assert fact_set.facts[10].period == 3  # Na

    def test_from_type_multiplication(self):
        """Test from_type creates multiplication table."""
        fact_set = FactSet.from_type(FactType.MULTIPLICATION)
        assert fact_set.fact_type == FactType.MULTIPLICATION
        assert len(fact_set.facts) > 0

    def test_from_type_addition(self):
        """Test from_type creates addition table."""
        fact_set = FactSet.from_type(FactType.ADDITION)
        assert fact_set.fact_type == FactType.ADDITION
        assert len(fact_set.facts) > 0

    def test_from_type_capitals(self):
        """Test from_type creates capitals fact set."""
        fact_set = FactSet.from_type(FactType.CAPITALS)
        assert fact_set.fact_type == FactType.CAPITALS
        assert len(fact_set.facts) > 0

    def test_from_type_elements(self):
        """Test from_type creates elements fact set."""
        fact_set = FactSet.from_type(FactType.ELEMENTS)
        assert fact_set.fact_type == FactType.ELEMENTS
        assert len(fact_set.facts) > 0

    def test_from_type_string(self):
        """Test from_type accepts string representation."""
        fact_set = FactSet.from_type("multiplication")
        assert fact_set.fact_type == FactType.MULTIPLICATION

    def test_from_type_custom_raises_error(self):
        """Test from_type raises error for CUSTOM type."""
        with pytest.raises(ValueError, match="Cannot auto-generate"):
            FactSet.from_type(FactType.CUSTOM)

    def test_from_type_invalid_raises_error(self):
        """Test from_type raises error for invalid type string."""
        with pytest.raises(ValueError):
            FactSet.from_type("invalid_type")
