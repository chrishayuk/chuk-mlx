"""
Trace Schema for CSP-CoT.

A trace is a sequence of steps that transform state.
Each step is verifiable by replaying the action on the prior state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal, ROUND_HALF_UP


class Action(Enum):
    """Atomic actions that transform state."""

    # Initialization
    INIT = "init"           # Initialize entity with value

    # Arithmetic
    ADD = "add"             # Add to entity value
    SUBTRACT = "subtract"   # Subtract from entity value
    MULTIPLY = "multiply"   # Multiply entity value
    DIVIDE = "divide"       # Divide entity value

    # Transfer
    TRANSFER = "transfer"   # Move amount from one entity to another

    # Comparison
    COMPARE = "compare"     # Compute difference between entities

    # Query
    QUERY = "query"         # Read final value (produces answer)


@dataclass
class State:
    """
    Immutable snapshot of all entity values.

    Uses Decimal for exact arithmetic (no floating point errors).
    """

    values: dict[str, Decimal] = field(default_factory=dict)

    def get(self, entity: str) -> Decimal:
        """Get entity value, default 0."""
        return self.values.get(entity, Decimal(0))

    def set(self, entity: str, value: Decimal | int | float) -> "State":
        """Return new state with updated value."""
        new_values = self.values.copy()
        new_values[entity] = Decimal(str(value))
        return State(values=new_values)

    def copy(self) -> "State":
        """Return a copy of this state."""
        return State(values=self.values.copy())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return self.values == other.values

    def to_dict(self) -> dict[str, float]:
        """Convert to JSON-serializable dict."""
        return {k: float(v) for k, v in self.values.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "State":
        """Create from dict."""
        return cls(values={k: Decimal(str(v)) for k, v in d.items()})


@dataclass
class Step:
    """
    A single step in a trace.

    Each step records:
    - The action taken
    - The parameters of the action
    - The state before the action
    - The state after the action

    Verification: apply(action, params, state_before) == state_after
    """

    action: Action
    params: dict[str, Any]
    state_before: State
    state_after: State

    def verify(self) -> bool:
        """
        Verify this step by replaying the action.

        Returns True if applying the action to state_before produces state_after.
        """
        computed = apply_action(self.action, self.params, self.state_before)
        return computed == self.state_after

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "action": self.action.value,
            "params": self.params,
            "state_before": self.state_before.to_dict(),
            "state_after": self.state_after.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        """Create from dict."""
        return cls(
            action=Action(d["action"]),
            params=d["params"],
            state_before=State.from_dict(d["state_before"]),
            state_after=State.from_dict(d["state_after"]),
        )


@dataclass
class Trace:
    """
    A complete trace of reasoning.

    A valid trace:
    1. Has sequential steps where step[i].state_after == step[i+1].state_before
    2. Each step verifies independently
    3. Final step is a QUERY that produces the answer
    """

    steps: list[Step] = field(default_factory=list)
    answer: Decimal | None = None
    problem_type: str = "unknown"

    def is_valid(self) -> bool:
        """
        Verify the entire trace.

        Checks:
        1. Each step verifies independently
        2. States chain correctly (step[i].after == step[i+1].before)
        3. Answer matches final query result
        """
        if not self.steps:
            return False

        # Check each step verifies
        for step in self.steps:
            if not step.verify():
                return False

        # Check state continuity
        for i in range(len(self.steps) - 1):
            if self.steps[i].state_after != self.steps[i + 1].state_before:
                return False

        # Check answer matches final query
        final_step = self.steps[-1]
        if final_step.action == Action.QUERY:
            expected_answer = final_step.params.get("result")
            if expected_answer is not None:
                if self.answer != Decimal(str(expected_answer)):
                    return False

        return True

    def replay(self) -> tuple[bool, State | None, int | None]:
        """
        Replay the trace from scratch.

        Returns:
            (success, final_state, failed_step_index)
            - success: True if replay succeeded
            - final_state: The state after replay (or None if failed)
            - failed_step_index: Index of first failing step (or None if success)
        """
        state = State()

        for i, step in enumerate(self.steps):
            # Verify state matches expected
            if state != step.state_before:
                return False, state, i

            # Apply action
            new_state = apply_action(step.action, step.params, state)

            # Verify result matches expected
            if new_state != step.state_after:
                return False, new_state, i

            state = new_state

        return True, state, None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "answer": float(self.answer) if self.answer is not None else None,
            "problem_type": self.problem_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        """Create from dict."""
        return cls(
            steps=[Step.from_dict(s) for s in d["steps"]],
            answer=Decimal(str(d["answer"])) if d.get("answer") is not None else None,
            problem_type=d.get("problem_type", "unknown"),
        )

    def to_yaml_str(self) -> str:
        """Format trace as readable YAML-like string."""
        lines = []
        for i, step in enumerate(self.steps):
            params_str = ", ".join(f"{k}: {v}" for k, v in step.params.items())
            lines.append(f"step_{i}: {{action: {step.action.value}, {params_str}}}")
        lines.append(f"answer: {self.answer}")
        return "\n".join(lines)


def apply_action(action: Action, params: dict[str, Any], state: State) -> State:
    """
    Apply an action to a state, returning the new state.

    This is the core execution engine - deterministic and exact.
    """
    if action == Action.INIT:
        entity = params["entity"]
        value = Decimal(str(params["value"]))
        return state.set(entity, value)

    elif action == Action.ADD:
        entity = params["entity"]
        amount = Decimal(str(params["amount"]))
        current = state.get(entity)
        return state.set(entity, current + amount)

    elif action == Action.SUBTRACT:
        entity = params["entity"]
        amount = Decimal(str(params["amount"]))
        current = state.get(entity)
        return state.set(entity, current - amount)

    elif action == Action.MULTIPLY:
        entity = params["entity"]
        factor = Decimal(str(params["factor"]))
        current = state.get(entity)
        return state.set(entity, current * factor)

    elif action == Action.DIVIDE:
        entity = params["entity"]
        divisor = Decimal(str(params["divisor"]))
        current = state.get(entity)
        # Use quantize for clean division results
        result = (current / divisor).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return state.set(entity, result)

    elif action == Action.TRANSFER:
        from_entity = params["from"]
        to_entity = params["to"]
        amount = Decimal(str(params["amount"]))

        from_val = state.get(from_entity)
        to_val = state.get(to_entity)

        new_state = state.set(from_entity, from_val - amount)
        new_state = new_state.set(to_entity, to_val + amount)
        return new_state

    elif action == Action.COMPARE:
        entity_a = params["entity_a"]
        entity_b = params["entity_b"]
        result_entity = params.get("result", "_comparison")

        val_a = state.get(entity_a)
        val_b = state.get(entity_b)
        diff = val_a - val_b

        return state.set(result_entity, diff)

    elif action == Action.QUERY:
        # Query doesn't change state, just reads a value
        # The result is stored in params for verification
        return state.copy()

    else:
        raise ValueError(f"Unknown action: {action}")


# =============================================================================
# TRACE BUILDER
# =============================================================================


class TraceBuilder:
    """
    Helper for building traces step by step.

    Automatically tracks state and creates verified steps.
    """

    def __init__(self, problem_type: str = "unknown"):
        self.state = State()
        self.steps: list[Step] = []
        self.problem_type = problem_type

    def _add_step(self, action: Action, params: dict[str, Any]) -> "TraceBuilder":
        """Add a step, computing the new state."""
        state_before = self.state.copy()
        state_after = apply_action(action, params, state_before)

        step = Step(
            action=action,
            params=params,
            state_before=state_before,
            state_after=state_after,
        )

        self.steps.append(step)
        self.state = state_after
        return self

    def init(self, entity: str, value: int | float | Decimal) -> "TraceBuilder":
        """Initialize an entity with a value."""
        return self._add_step(Action.INIT, {"entity": entity, "value": value})

    def add(self, entity: str, amount: int | float | Decimal) -> "TraceBuilder":
        """Add to an entity's value."""
        return self._add_step(Action.ADD, {"entity": entity, "amount": amount})

    def subtract(self, entity: str, amount: int | float | Decimal) -> "TraceBuilder":
        """Subtract from an entity's value."""
        return self._add_step(Action.SUBTRACT, {"entity": entity, "amount": amount})

    def multiply(self, entity: str, factor: int | float | Decimal) -> "TraceBuilder":
        """Multiply an entity's value."""
        return self._add_step(Action.MULTIPLY, {"entity": entity, "factor": factor})

    def divide(self, entity: str, divisor: int | float | Decimal) -> "TraceBuilder":
        """Divide an entity's value."""
        return self._add_step(Action.DIVIDE, {"entity": entity, "divisor": divisor})

    def transfer(
        self, from_entity: str, to_entity: str, amount: int | float | Decimal
    ) -> "TraceBuilder":
        """Transfer amount from one entity to another."""
        return self._add_step(
            Action.TRANSFER,
            {"from": from_entity, "to": to_entity, "amount": amount},
        )

    def compare(
        self, entity_a: str, entity_b: str, result: str = "_comparison"
    ) -> "TraceBuilder":
        """Compare two entities (a - b)."""
        return self._add_step(
            Action.COMPARE,
            {"entity_a": entity_a, "entity_b": entity_b, "result": result},
        )

    def query(self, entity: str) -> "TraceBuilder":
        """Query an entity's final value."""
        result = self.state.get(entity)
        return self._add_step(Action.QUERY, {"entity": entity, "result": float(result)})

    def build(self) -> Trace:
        """Build the final trace."""
        # Get answer from final query
        answer = None
        if self.steps and self.steps[-1].action == Action.QUERY:
            answer = Decimal(str(self.steps[-1].params["result"]))

        return Trace(
            steps=self.steps,
            answer=answer,
            problem_type=self.problem_type,
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Trace Schema Tests")
    print("=" * 60)

    # Test 1: Simple entity tracking
    print("\nTest 1: Jenny has 5 apples, gives 2 to Bob")
    trace = (
        TraceBuilder(problem_type="entity_tracking")
        .init("jenny", 5)
        .init("bob", 0)
        .transfer("jenny", "bob", 2)
        .query("jenny")
        .build()
    )

    print(f"Trace:\n{trace.to_yaml_str()}")
    print(f"Valid: {trace.is_valid()}")
    print(f"Answer: {trace.answer}")

    success, final_state, failed = trace.replay()
    print(f"Replay: success={success}, final_state={final_state.to_dict()}")

    # Test 2: Arithmetic chain
    print("\nTest 2: Start with 10, add 5, multiply by 2")
    trace2 = (
        TraceBuilder(problem_type="arithmetic_chain")
        .init("x", 10)
        .add("x", 5)
        .multiply("x", 2)
        .query("x")
        .build()
    )

    print(f"Trace:\n{trace2.to_yaml_str()}")
    print(f"Valid: {trace2.is_valid()}")
    print(f"Answer: {trace2.answer}")

    # Test 3: Comparison
    print("\nTest 3: Tom has 15, Jane has 5, how many more does Tom have?")
    trace3 = (
        TraceBuilder(problem_type="comparison")
        .init("tom", 15)
        .init("jane", 5)
        .compare("tom", "jane", "difference")
        .query("difference")
        .build()
    )

    print(f"Trace:\n{trace3.to_yaml_str()}")
    print(f"Valid: {trace3.is_valid()}")
    print(f"Answer: {trace3.answer}")

    # Test 4: Invalid trace (tampered)
    print("\nTest 4: Tampered trace (should fail verification)")
    bad_trace = Trace(
        steps=[
            Step(
                action=Action.INIT,
                params={"entity": "x", "value": 5},
                state_before=State(),
                state_after=State(values={"x": Decimal(10)}),  # Wrong! Should be 5
            )
        ],
        answer=Decimal(10),
        problem_type="test",
    )

    print(f"Valid: {bad_trace.is_valid()}")  # Should be False

    # Test 5: Division
    print("\nTest 5: 12 cookies split between 4 kids")
    trace5 = (
        TraceBuilder(problem_type="division")
        .init("cookies", 12)
        .divide("cookies", 4)
        .query("cookies")
        .build()
    )

    print(f"Trace:\n{trace5.to_yaml_str()}")
    print(f"Valid: {trace5.is_valid()}")
    print(f"Answer: {trace5.answer}")
