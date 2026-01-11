"""Pydantic models for fact datasets."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..enums import FactType, Region


class Fact(BaseModel):
    """Base class for a fact that can be queried."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The query prompt")
    answer: str = Field(description="The expected answer")
    category: str = Field(default="", description="Primary category for grouping")
    category_alt: str | None = Field(default=None, description="Alternative category")

    @property
    def fact_type(self) -> FactType:
        """Return the fact type."""
        return FactType.CUSTOM


class MathFact(Fact):
    """A mathematical fact (arithmetic operation)."""

    operand_a: int = Field(description="First operand")
    operand_b: int = Field(description="Second operand")
    operator: str = Field(description="Operator symbol")

    @property
    def fact_type(self) -> FactType:
        """Return the fact type based on operator."""
        if self.operator in ["*", "x", "×"]:
            return FactType.MULTIPLICATION
        elif self.operator == "+":
            return FactType.ADDITION
        return FactType.CUSTOM


class CapitalFact(Fact):
    """A country capital fact."""

    country: str = Field(description="Country name")
    region: Region = Field(default=Region.OTHER, description="Geographic region")

    @property
    def fact_type(self) -> FactType:
        return FactType.CAPITALS


class ElementFact(Fact):
    """A periodic table element fact."""

    atomic_number: int = Field(description="Atomic number")
    symbol: str = Field(description="Element symbol")
    period: int = Field(description="Periodic table period")

    @property
    def fact_type(self) -> FactType:
        return FactType.ELEMENTS


class FactNeighborhood(BaseModel):
    """Analysis of what facts appear near a queried fact."""

    model_config = ConfigDict(frozen=True)

    correct_rank: int | None = Field(
        default=None, description="Rank of correct answer in predictions"
    )
    correct_prob: float | None = Field(default=None, description="Probability of correct answer")
    same_category: list[dict] = Field(
        default_factory=list, description="Other facts from same category"
    )
    same_category_alt: list[dict] = Field(
        default_factory=list, description="Other facts from alt category"
    )
    other_answers: list[dict] = Field(default_factory=list, description="Other known answers")
    non_answers: list[dict] = Field(
        default_factory=list, description="Tokens that aren't known answers"
    )


class FactSet(BaseModel):
    """A collection of facts for analysis."""

    fact_type: FactType = Field(description="Type of facts in this set")
    facts: list[Fact] = Field(default_factory=list, description="The facts")

    @classmethod
    def multiplication_table(cls, start: int = 2, end: int = 9) -> FactSet:
        """Generate single-digit multiplication facts."""
        facts = []
        for a in range(start, end + 1):
            for b in range(start, end + 1):
                facts.append(
                    MathFact(
                        query=f"{a}*{b}=",
                        answer=str(a * b),
                        operand_a=a,
                        operand_b=b,
                        operator="*",
                        category=f"{a}x",
                        category_alt=f"x{b}",
                    )
                )
        return cls(fact_type=FactType.MULTIPLICATION, facts=facts)

    @classmethod
    def addition_table(cls, start: int = 1, end: int = 9) -> FactSet:
        """Generate single-digit addition facts."""
        facts = []
        for a in range(start, end + 1):
            for b in range(start, end + 1):
                facts.append(
                    MathFact(
                        query=f"{a}+{b}=",
                        answer=str(a + b),
                        operand_a=a,
                        operand_b=b,
                        operator="+",
                        category=f"{a}+",
                        category_alt=f"+{b}",
                    )
                )
        return cls(fact_type=FactType.ADDITION, facts=facts)

    @classmethod
    def world_capitals(cls) -> FactSet:
        """Generate country capital facts."""
        capitals_data = [
            ("France", "Paris", Region.EUROPE),
            ("Germany", "Berlin", Region.EUROPE),
            ("Italy", "Rome", Region.EUROPE),
            ("Spain", "Madrid", Region.EUROPE),
            ("UK", "London", Region.EUROPE),
            ("Poland", "Warsaw", Region.EUROPE),
            ("Netherlands", "Amsterdam", Region.EUROPE),
            ("Belgium", "Brussels", Region.EUROPE),
            ("Sweden", "Stockholm", Region.EUROPE),
            ("Norway", "Oslo", Region.EUROPE),
            ("Denmark", "Copenhagen", Region.EUROPE),
            ("Finland", "Helsinki", Region.EUROPE),
            ("Greece", "Athens", Region.EUROPE),
            ("Japan", "Tokyo", Region.ASIA),
            ("China", "Beijing", Region.ASIA),
            ("India", "Delhi", Region.ASIA),
            ("Turkey", "Ankara", Region.ASIA),
            ("Iran", "Tehran", Region.ASIA),
            ("Iraq", "Baghdad", Region.ASIA),
            ("Saudi Arabia", "Riyadh", Region.ASIA),
            ("Israel", "Jerusalem", Region.ASIA),
            ("Thailand", "Bangkok", Region.ASIA),
            ("Brazil", "Brasilia", Region.AMERICAS),
            ("Canada", "Ottawa", Region.AMERICAS),
            ("Mexico", "Mexico City", Region.AMERICAS),
            ("Argentina", "Buenos Aires", Region.AMERICAS),
            ("Russia", "Moscow", Region.EUROPE),
            ("Australia", "Canberra", Region.OCEANIA),
            ("Egypt", "Cairo", Region.AFRICA),
            ("South Africa", "Pretoria", Region.AFRICA),
        ]

        facts = []
        for country, capital, region in capitals_data:
            facts.append(
                CapitalFact(
                    query=f"The capital of {country} is",
                    answer=capital,
                    country=country,
                    region=region,
                    category=region.value,
                )
            )
        return cls(fact_type=FactType.CAPITALS, facts=facts)

    @classmethod
    def periodic_elements(cls, max_number: int = 20) -> FactSet:
        """Generate periodic table element facts."""
        elements_data = [
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
        ]

        facts = []
        for num, symbol, name in elements_data:
            if num > max_number:
                break
            period = 1 if num <= 2 else 2 if num <= 10 else 3
            facts.append(
                ElementFact(
                    query=f"Element {num} is",
                    answer=name,
                    atomic_number=num,
                    symbol=symbol,
                    period=period,
                    category=f"Period {period}",
                )
            )
        return cls(fact_type=FactType.ELEMENTS, facts=facts)

    @classmethod
    def from_type(cls, fact_type: FactType | str) -> FactSet:
        """Create a fact set from a type identifier."""
        if isinstance(fact_type, str):
            fact_type = FactType(fact_type)

        if fact_type == FactType.MULTIPLICATION:
            return cls.multiplication_table()
        elif fact_type == FactType.ADDITION:
            return cls.addition_table()
        elif fact_type == FactType.CAPITALS:
            return cls.world_capitals()
        elif fact_type == FactType.ELEMENTS:
            return cls.periodic_elements()
        else:
            raise ValueError(f"Cannot auto-generate facts for type: {fact_type}")
