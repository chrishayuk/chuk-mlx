"""Trace verifier - replays traces to verify correctness."""

from __future__ import annotations
from typing import Any
import yaml


def compute_op(op: str, args: list[float]) -> float:
    """Execute arithmetic operation."""
    if op == "add":
        return sum(args)
    elif op == "sub":
        return args[0] - sum(args[1:])
    elif op == "mul":
        result = 1.0
        for a in args:
            result *= a
        return result
    elif op == "div":
        return args[0] / args[1] if args[1] != 0 else float('inf')
    else:
        raise ValueError(f"Unknown op: {op}")


def resolve_arg(arg: Any, state: dict) -> float:
    """Resolve argument - either a value or state lookup."""
    if isinstance(arg, (int, float)):
        return float(arg)
    elif isinstance(arg, str):
        return state.get(arg, 0.0)
    else:
        return float(arg)


def verify_trace(trace: list[dict], tolerance: float = 0.01) -> tuple[bool, str, dict]:
    """
    Replay trace and verify each step.

    Returns:
        (valid, error_message, final_state)
    """
    state = {}

    for i, step in enumerate(trace):
        try:
            # Init step: {init: var, value: n}
            if "init" in step:
                var = step["init"]
                value = step["value"]
                state[var] = float(value)

            # Transfer step: {transfer: {from: e1, to: e2, amount: n}}
            elif "transfer" in step:
                t = step["transfer"]
                from_e = t["from"]
                to_e = t["to"]
                amount = float(t["amount"])

                if state.get(from_e, 0) < amount - tolerance:
                    return False, f"Step {i}: insufficient {from_e} for transfer", state

                state[from_e] = state.get(from_e, 0) - amount
                state[to_e] = state.get(to_e, 0) + amount

            # Consume step: {consume: {entity: e, amount: n}}
            elif "consume" in step:
                c = step["consume"]
                entity = c["entity"]
                amount = float(c["amount"])

                if state.get(entity, 0) < amount - tolerance:
                    return False, f"Step {i}: insufficient {entity} for consume", state

                state[entity] = state.get(entity, 0) - amount

            # Compute step: {compute: {op: op, args: [...], var: name, result: n}}
            elif "compute" in step:
                c = step["compute"]
                op = c["op"]
                args = [resolve_arg(a, state) for a in c["args"]]
                expected_result = float(c["result"])

                computed = compute_op(op, args)

                if abs(computed - expected_result) > tolerance:
                    return False, f"Step {i}: compute {op}({args})={computed}, expected {expected_result}", state

                if "var" in c:
                    state[c["var"]] = expected_result

            # State assertion: {state: {var: value, ...}}
            elif "state" in step:
                for var, expected in step["state"].items():
                    actual = state.get(var, 0)
                    if abs(actual - float(expected)) > tolerance:
                        return False, f"Step {i}: state {var}={actual}, expected {expected}", state

            # Percent off: {percent_off: {base: var, rate: n, result: n}}
            elif "percent_off" in step:
                p = step["percent_off"]
                base_val = resolve_arg(p["base"], state)
                rate = float(p["rate"]) / 100.0
                expected = float(p["result"])
                computed = base_val * (1 - rate)

                if abs(computed - expected) > tolerance:
                    return False, f"Step {i}: percent_off {base_val}*{1-rate}={computed}, expected {expected}", state

                if "var" in p:
                    state[p["var"]] = expected

            # Compare step: {compare: {op: op, args: [...], var: name, result: n}}
            elif "compare" in step:
                c = step["compare"]
                op = c["op"]
                args = [resolve_arg(a, state) for a in c["args"]]
                expected_result = float(c["result"])

                computed = compute_op(op, args)

                if abs(computed - expected_result) > tolerance:
                    return False, f"Step {i}: compare {op}({args})={computed}, expected {expected_result}", state

                if "var" in c:
                    state[c["var"]] = expected_result

            # Given step (rate problems): {given: {rate: n, time: n, ...}}
            elif "given" in step:
                g = step["given"]
                for k, v in g.items():
                    if k not in ["unit"]:  # Skip non-numeric
                        state[k] = float(v) if isinstance(v, (int, float)) else v

            # Formula step: {formula: "equation"} - informational only
            elif "formula" in step:
                pass  # Just documentation

            # Query step: {query: var} - informational only
            elif "query" in step:
                pass  # Just marks what we're querying

            else:
                # Unknown step type - skip but warn
                pass

        except Exception as e:
            return False, f"Step {i}: exception {e}", state

    return True, "valid", state


def verify_yaml_output(yaml_str: str, expected_answer: Any = None, tolerance: float = 0.01) -> dict:
    """
    Verify a complete YAML trace output.

    Returns dict with:
        - parsed: bool
        - expert: str or None
        - trace_valid: bool
        - trace_error: str or None
        - answer: Any
        - answer_correct: bool (if expected_answer provided)
    """
    result = {
        "parsed": False,
        "expert": None,
        "trace_valid": False,
        "trace_error": None,
        "answer": None,
        "answer_correct": None,
        "final_state": {},
    }

    # Parse YAML
    try:
        data = yaml.safe_load(yaml_str)
        result["parsed"] = True
    except Exception as e:
        result["trace_error"] = f"YAML parse error: {e}"
        return result

    if not isinstance(data, dict):
        result["trace_error"] = "Output is not a dict"
        return result

    result["expert"] = data.get("expert")
    result["answer"] = data.get("answer")

    # Verify trace
    trace = data.get("trace", [])
    if not isinstance(trace, list):
        result["trace_error"] = "Trace is not a list"
        return result

    valid, error, final_state = verify_trace(trace, tolerance)
    result["trace_valid"] = valid
    result["trace_error"] = error if not valid else None
    result["final_state"] = final_state

    # Check answer if expected provided
    if expected_answer is not None:
        try:
            if isinstance(expected_answer, (int, float)) and isinstance(result["answer"], (int, float)):
                result["answer_correct"] = abs(float(result["answer"]) - float(expected_answer)) < tolerance
            else:
                result["answer_correct"] = result["answer"] == expected_answer
        except:
            result["answer_correct"] = False

    return result


# Quick test
if __name__ == "__main__":
    # Test entity tracking trace
    test_trace = [
        {"init": "eggs", "value": 16},
        {"consume": {"entity": "eggs", "amount": 3}},
        {"consume": {"entity": "eggs", "amount": 4}},
        {"state": {"eggs": 9}},
        {"compute": {"op": "mul", "args": [9, 2], "var": "revenue", "result": 18}},
    ]

    valid, error, state = verify_trace(test_trace)
    print(f"Valid: {valid}")
    print(f"Error: {error}")
    print(f"State: {state}")

    # Test YAML output
    yaml_output = """
expert: entity_track
trace:
  - {init: eggs, value: 16}
  - {consume: {entity: eggs, amount: 3}}
  - {consume: {entity: eggs, amount: 4}}
  - {state: {eggs: 9}}
  - {compute: {op: mul, args: [9, 2], var: revenue, result: 18}}
answer: 18
"""

    result = verify_yaml_output(yaml_output, expected_answer=18)
    print(f"\nYAML verification: {result}")
