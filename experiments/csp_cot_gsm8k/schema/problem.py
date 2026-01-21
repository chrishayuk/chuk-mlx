"""
Problem Specification Schema.

This is what the LLM extracts from natural language.
No regex - the LLM does semantic parsing into this structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal


class ProblemType(Enum):
    """Types of GSM-8K problems."""

    ENTITY_TRACKING = "entity_tracking"    # Alice has X, gives Y to Bob
    ARITHMETIC_CHAIN = "arithmetic_chain"  # Sequential operations on one value
    RATE_EQUATION = "rate_equation"        # Workers/time, speed/distance
    ALLOCATION = "allocation"              # Split X where A = 2*B
    COMPARISON = "comparison"              # How many more does X have than Y
    PERCENTAGE = "percentage"              # X% of Y, discount, tax
    TIME_CALC = "time_calc"                # Start at X, add Y hours
    GEOMETRY = "geometry"                  # Area, perimeter, volume
    UNKNOWN = "unknown"


class OperationType(Enum):
    """Types of operations in a problem."""

    INIT = "init"           # Entity starts with value
    ADD = "add"             # Add to entity
    SUBTRACT = "subtract"   # Subtract from entity
    MULTIPLY = "multiply"   # Multiply entity value
    DIVIDE = "divide"       # Divide entity value
    TRANSFER = "transfer"   # Move from one entity to another
    SET = "set"             # Set entity to specific value


@dataclass
class Entity:
    """
    An entity in the problem (person, object, variable).

    Examples:
    - Entity(name="jenny", attribute="apples", initial_value=5)
    - Entity(name="cookies", attribute="count", initial_value=12)
    - Entity(name="price", attribute="dollars", initial_value=40)
    """

    name: str
    attribute: str | None = None  # "apples", "dollars", "marbles"
    initial_value: Decimal | None = None
    unit: str | None = None  # "apples", "$", "hours"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "attribute": self.attribute,
            "initial_value": float(self.initial_value) if self.initial_value else None,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            name=d["name"],
            attribute=d.get("attribute"),
            initial_value=Decimal(str(d["initial_value"])) if d.get("initial_value") is not None else None,
            unit=d.get("unit"),
        )


@dataclass
class Operation:
    """
    An operation that transforms state.

    Examples:
    - Operation(type=ADD, target="jenny", amount=3)  # Jenny gets 3 more
    - Operation(type=TRANSFER, source="jenny", target="bob", amount=2)  # Jenny gives 2 to Bob
    - Operation(type=MULTIPLY, target="price", factor=0.75)  # 25% discount
    """

    type: OperationType
    target: str  # Entity being modified
    source: str | None = None  # For transfers
    amount: Decimal | None = None
    factor: Decimal | None = None  # For multiply/divide
    condition: str | None = None  # "each", "per day", etc.

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "target": self.target,
            "source": self.source,
            "amount": float(self.amount) if self.amount else None,
            "factor": float(self.factor) if self.factor else None,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Operation":
        return cls(
            type=OperationType(d["type"]),
            target=d["target"],
            source=d.get("source"),
            amount=Decimal(str(d["amount"])) if d.get("amount") is not None else None,
            factor=Decimal(str(d["factor"])) if d.get("factor") is not None else None,
            condition=d.get("condition"),
        )


@dataclass
class Query:
    """
    What the problem is asking for.

    Examples:
    - Query(target="jenny", question="how_many")  # How many does Jenny have?
    - Query(target="difference", question="compare", compare_a="tom", compare_b="jane")
    - Query(target="total", question="sum")
    """

    target: str  # Entity to query
    question: str = "value"  # "value", "how_many", "compare", "total"
    compare_a: str | None = None  # For comparison queries
    compare_b: str | None = None

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "question": self.question,
            "compare_a": self.compare_a,
            "compare_b": self.compare_b,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        return cls(
            target=d["target"],
            question=d.get("question", "value"),
            compare_a=d.get("compare_a"),
            compare_b=d.get("compare_b"),
        )


@dataclass
class Constraint:
    """
    A constraint or relationship in the problem.

    Used for allocation/equation problems.

    Examples:
    - Constraint(type="ratio", entities=["alice", "bob"], factor=2)  # Alice = 2 * Bob
    - Constraint(type="sum", entities=["alice", "bob", "carol"], value=100)  # Sum = 100
    - Constraint(type="difference", entities=["alice", "bob"], value=10)  # Alice - Bob = 10
    """

    type: str  # "ratio", "sum", "difference", "equals"
    entities: list[str] = field(default_factory=list)
    factor: Decimal | None = None
    value: Decimal | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "entities": self.entities,
            "factor": float(self.factor) if self.factor else None,
            "value": float(self.value) if self.value else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        return cls(
            type=d["type"],
            entities=d.get("entities", []),
            factor=Decimal(str(d["factor"])) if d.get("factor") is not None else None,
            value=Decimal(str(d["value"])) if d.get("value") is not None else None,
        )


@dataclass
class ProblemSpec:
    """
    Complete structured representation of a GSM-8K problem.

    This is what the LLM extracts from natural language.
    The trace generator uses this to produce a verifiable trace.
    """

    problem_type: ProblemType = ProblemType.UNKNOWN
    entities: list[Entity] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    query: Query | None = None
    raw_text: str = ""

    def is_valid(self) -> bool:
        """Check if spec has minimum required information."""
        has_entities = len(self.entities) > 0
        has_query = self.query is not None
        return has_entities and has_query

    def to_dict(self) -> dict:
        return {
            "problem_type": self.problem_type.value,
            "entities": [e.to_dict() for e in self.entities],
            "operations": [o.to_dict() for o in self.operations],
            "constraints": [c.to_dict() for c in self.constraints],
            "query": self.query.to_dict() if self.query else None,
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec":
        return cls(
            problem_type=ProblemType(d.get("problem_type", "unknown")),
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            operations=[Operation.from_dict(o) for o in d.get("operations", [])],
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])],
            query=Query.from_dict(d["query"]) if d.get("query") else None,
            raw_text=d.get("raw_text", ""),
        )

    def to_json_str(self) -> str:
        """Format as JSON string for LLM prompting."""
        import json
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# EXAMPLE PROBLEM SPECS
# =============================================================================

EXAMPLE_SPECS = {
    "entity_tracking": ProblemSpec(
        problem_type=ProblemType.ENTITY_TRACKING,
        entities=[
            Entity(name="jenny", attribute="apples", initial_value=Decimal(5)),
            Entity(name="bob", attribute="apples", initial_value=Decimal(0)),
        ],
        operations=[
            Operation(type=OperationType.TRANSFER, source="jenny", target="bob", amount=Decimal(2)),
        ],
        query=Query(target="jenny", question="how_many"),
        raw_text="Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?",
    ),

    "arithmetic_chain": ProblemSpec(
        problem_type=ProblemType.ARITHMETIC_CHAIN,
        entities=[
            Entity(name="sam", attribute="marbles", initial_value=Decimal(10)),
        ],
        operations=[
            Operation(type=OperationType.SUBTRACT, target="sam", amount=Decimal(3)),
            Operation(type=OperationType.ADD, target="sam", amount=Decimal(5)),
        ],
        query=Query(target="sam", question="how_many"),
        raw_text="Sam had 10 marbles. He lost 3, then found 5. How many does he have now?",
    ),

    "comparison": ProblemSpec(
        problem_type=ProblemType.COMPARISON,
        entities=[
            Entity(name="tom", attribute="marbles", initial_value=Decimal(15)),
            Entity(name="jane", attribute="marbles", initial_value=Decimal(5)),
        ],
        operations=[],
        query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
        raw_text="Tom has 15 marbles. Jane has 5 marbles. How many more does Tom have than Jane?",
    ),

    "allocation": ProblemSpec(
        problem_type=ProblemType.ALLOCATION,
        entities=[
            Entity(name="alice", attribute="dollars"),
            Entity(name="bob", attribute="dollars"),
        ],
        constraints=[
            Constraint(type="sum", entities=["alice", "bob"], value=Decimal(100)),
            Constraint(type="ratio", entities=["alice", "bob"], factor=Decimal(2)),  # Alice = 2 * Bob
        ],
        query=Query(target="alice", question="value"),
        raw_text="Split $100 between Alice and Bob. Alice gets twice what Bob gets. How much does Alice get?",
    ),

    "percentage": ProblemSpec(
        problem_type=ProblemType.PERCENTAGE,
        entities=[
            Entity(name="price", attribute="dollars", initial_value=Decimal(40)),
        ],
        operations=[
            Operation(type=OperationType.MULTIPLY, target="price", factor=Decimal("0.75")),  # 25% off
        ],
        query=Query(target="price", question="value"),
        raw_text="A shirt costs $40. It's 25% off. What's the final price?",
    ),
}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Problem Schema Tests")
    print("=" * 60)

    for name, spec in EXAMPLE_SPECS.items():
        print(f"\n{name}:")
        print(f"  Raw: {spec.raw_text[:50]}...")
        print(f"  Entities: {[e.name for e in spec.entities]}")
        print(f"  Operations: {len(spec.operations)}")
        print(f"  Query: {spec.query.target if spec.query else None}")
        print(f"  Valid: {spec.is_valid()}")

    # Test serialization
    print("\n" + "=" * 60)
    print("Serialization Test")
    spec = EXAMPLE_SPECS["entity_tracking"]
    d = spec.to_dict()
    spec2 = ProblemSpec.from_dict(d)
    print(f"Round-trip OK: {spec.to_dict() == spec2.to_dict()}")
