"""
Calculator Chain Expert.

Handles sequential arithmetic: entity tracking + operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec, Entity, Operation


@dataclass
class ChainResult:
    """Result from calculator chain."""
    final_states: dict[str, float]
    steps: list[str]
    answer: float | None


class CalculatorChainExpert:
    """
    Expert for sequential arithmetic problems.

    Tracks entity states through operations and computes final values.

    Example:
        "Jenny has 5 apples. She buys 3 more. Then gives 2 away."
        → Jenny: 5 + 3 - 2 = 6
    """

    def solve(self, spec: ProblemSpec) -> ChainResult:
        """
        Execute operation chain and track entity states.
        """
        # Initialize entity states
        state = {}
        entity_names = []
        for entity in spec.entities:
            if entity.value is not None:
                state[entity.name] = entity.value
                entity_names.append(entity.name)

        steps = []

        # Resolve pronouns to last mentioned entity
        def resolve_entity(name: str) -> str | None:
            if name is None:
                return entity_names[0] if entity_names else None
            if name in state:
                return name
            # Pronoun and adverb resolution
            if name.lower() in ("she", "he", "they", "it", "then", "next", "later", "afterwards"):
                return entity_names[0] if entity_names else None
            return name

        # Apply operations in sequence
        for op in spec.operations:
            target = resolve_entity(op.source or op.target)

            if op.type == "add":
                if target and target in state and op.amount:
                    old_val = state[target]
                    state[target] += op.amount
                    steps.append(f"{target}: {old_val} + {op.amount} = {state[target]}")
                elif target and op.amount:
                    state[target] = op.amount
                    steps.append(f"{target}: starts with {op.amount}")

            elif op.type == "subtract":
                if target and target in state and op.amount:
                    old_val = state[target]
                    state[target] -= op.amount
                    steps.append(f"{target}: {old_val} - {op.amount} = {state[target]}")

                    # Transfer to target if specified
                    if op.target and op.target != target:
                        if op.target not in state:
                            state[op.target] = 0
                        state[op.target] += op.amount
                        steps.append(f"{op.target}: receives {op.amount}")

        # Determine answer
        answer = None
        if spec.target and spec.target in state:
            answer = state[spec.target]
        elif len(state) == 1:
            answer = list(state.values())[0]
        elif state:
            # Default to first entity
            first_entity = spec.entities[0].name if spec.entities else list(state.keys())[0]
            answer = state.get(first_entity)

        return ChainResult(
            final_states=state,
            steps=steps,
            answer=answer,
        )

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.answer


if __name__ == "__main__":
    from ..extraction.gsm8k_extractor import extract_problem

    expert = CalculatorChainExpert()

    test = "Jenny has 5 apples. She buys 3 more. Then gives 2 away. How many does Jenny have?"
    spec = extract_problem(test)
    result = expert.solve(spec)

    print(f"Problem: {test}")
    print(f"Steps: {result.steps}")
    print(f"Answer: {result.answer}")
