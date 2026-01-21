"""
Standalone Math Word Problem Expert for training.

No external dependencies on chuk_virtual_expert - uses trace generators directly.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .schema.problem import (
    ProblemSpec,
    ProblemType,
    Entity,
    Operation,
    OperationType,
    Query,
    Constraint,
)
from .schema.verifier import TraceVerifier, VerificationStatus
from .generators import generate_trace


class MathSolver:
    """
    Standalone math problem solver using verifiable traces.

    No VirtualExpert base class needed - just the core solve functionality.
    """

    def solve(
        self,
        problem_type: str,
        entities: list[dict],
        operations: list[dict] | None = None,
        constraints: list[dict] | None = None,
        query: dict | None = None,
        raw_text: str = "",
    ) -> dict[str, Any]:
        """
        Solve a math word problem with a verifiable trace.
        """
        try:
            spec = self._build_spec(
                problem_type=problem_type,
                entities=entities,
                operations=operations or [],
                constraints=constraints or [],
                query=query,
                raw_text=raw_text,
            )

            if not spec.is_valid():
                return {
                    "success": False,
                    "error": "Invalid problem specification",
                    "problem_type": problem_type,
                }

            trace = generate_trace(spec)
            if trace is None:
                return {
                    "success": False,
                    "error": f"No generator for problem type: {problem_type}",
                    "problem_type": problem_type,
                }

            verifier = TraceVerifier()
            verification = verifier.verify(trace)

            return {
                "success": True,
                "answer": float(trace.answer) if trace.answer is not None else None,
                "verified": verification.status == VerificationStatus.VALID,
                "verification_status": verification.status.value,
                "problem_type": problem_type,
                "trace_steps": len(trace.steps),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": problem_type,
            }

    def _build_spec(
        self,
        problem_type: str,
        entities: list[dict],
        operations: list[dict],
        constraints: list[dict],
        query: dict | None,
        raw_text: str,
    ) -> ProblemSpec:
        """Build a ProblemSpec from dict parameters with flexible parsing."""
        try:
            pt = ProblemType(problem_type)
        except ValueError:
            pt = ProblemType.UNKNOWN

        entity_values = {}
        parsed_entities = []
        for e in entities:
            name = e.get("name", "unknown")
            initial = self._parse_numeric(e.get("initial_value"))
            entity_values[name] = initial
            parsed_entities.append(
                Entity(
                    name=name,
                    attribute=e.get("attribute"),
                    initial_value=initial,
                    unit=e.get("unit"),
                )
            )

        parsed_operations = []
        for o in operations:
            op_type_str = o.get("type", "add")
            try:
                op_type = OperationType(op_type_str)
            except ValueError:
                op_type = OperationType.ADD

            amount = self._resolve_value(o, "amount", entity_values)
            if amount is None:
                amount = self._resolve_value(o, "value", entity_values)

            factor = self._resolve_value(o, "factor", entity_values)
            if factor is None:
                factor = self._resolve_value(o, "multiplier", entity_values)

            parsed_operations.append(
                Operation(
                    type=op_type,
                    target=o.get("target", ""),
                    source=o.get("source") or o.get("from"),
                    amount=amount,
                    factor=factor,
                )
            )

        parsed_constraints = []
        for c in constraints:
            parsed_constraints.append(
                Constraint(
                    type=c.get("type", ""),
                    entities=c.get("entities", []),
                    factor=self._parse_numeric(c.get("factor")),
                    value=self._parse_numeric(c.get("value")),
                )
            )

        parsed_query = None
        if query:
            target = query.get("target", "")
            if target in ("result", "answer", "final") and parsed_entities:
                target = parsed_entities[0].name
            parsed_query = Query(
                target=target,
                question=query.get("question", "value"),
                compare_a=query.get("compare_a"),
                compare_b=query.get("compare_b"),
            )

        return ProblemSpec(
            problem_type=pt,
            entities=parsed_entities,
            operations=parsed_operations,
            constraints=parsed_constraints,
            query=parsed_query,
            raw_text=raw_text,
        )

    def _parse_numeric(self, value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                cleaned = value.replace("$", "").replace(",", "").strip()
                return Decimal(cleaned)
            except:
                return None
        return None

    def _resolve_value(
        self,
        obj: dict,
        key: str,
        entity_values: dict[str, Decimal | None],
    ) -> Decimal | None:
        value = obj.get(key)
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            try:
                cleaned = value.replace("$", "").replace(",", "").strip()
                return Decimal(cleaned)
            except:
                pass

            if value in entity_values and entity_values[value] is not None:
                return entity_values[value]

        return None
