"""Trace verifier - executes symbolic traces to compute answers."""

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
        if arg in state:
            return state[arg]
        else:
            raise KeyError(f"Variable not found: {arg}")
    else:
        return float(arg)


def execute_trace(trace: list[dict], tolerance: float = 0.01) -> tuple[bool, str, dict, Any]:
    """
    Execute symbolic trace and compute all values.

    The trace contains structure but no results - we compute them.

    Returns:
        (valid, error_message, final_state, computed_answer)
    """
    state = {}
    query_var = None

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
                amount = resolve_arg(t["amount"], state)

                if from_e not in state:
                    state[from_e] = 0
                if to_e not in state:
                    state[to_e] = 0

                if state[from_e] < amount - tolerance:
                    return False, f"Step {i}: insufficient {from_e} for transfer", state, None

                state[from_e] -= amount
                state[to_e] += amount

            # Consume step: {consume: {entity: e, amount: n}}
            elif "consume" in step:
                c = step["consume"]
                entity = c["entity"]
                amount = resolve_arg(c["amount"], state)

                if entity not in state:
                    return False, f"Step {i}: entity {entity} not initialized", state, None

                if state[entity] < amount - tolerance:
                    return False, f"Step {i}: insufficient {entity} for consume", state, None

                state[entity] -= amount

            # Add step: {add: {entity: e, amount: n}}
            elif "add" in step:
                a = step["add"]
                entity = a["entity"]
                amount = resolve_arg(a["amount"], state)

                if entity not in state:
                    state[entity] = 0
                state[entity] += amount

            # Compute step: {compute: {op: op, args: [...], var: name}}
            # No result field - we compute it
            elif "compute" in step:
                c = step["compute"]
                op = c["op"]
                args = [resolve_arg(a, state) for a in c["args"]]
                result = compute_op(op, args)

                if "var" in c:
                    state[c["var"]] = result

            # Percent off: {percent_off: {base: var, rate: n, var: result_var}}
            elif "percent_off" in step:
                p = step["percent_off"]
                base_val = resolve_arg(p["base"], state)
                rate = resolve_arg(p["rate"], state)
                result = base_val * (1 - rate / 100)

                if "var" in p:
                    state[p["var"]] = result

            # Percent increase: {percent_increase: {base: var, rate: n, var: result_var}}
            elif "percent_increase" in step:
                p = step["percent_increase"]
                base_val = resolve_arg(p["base"], state)
                rate = resolve_arg(p["rate"], state)
                result = base_val * (1 + rate / 100)

                if "var" in p:
                    state[p["var"]] = result

            # Compare step: {compare: {op: op, args: [...], var: name}}
            elif "compare" in step:
                c = step["compare"]
                op = c["op"]
                args = [resolve_arg(a, state) for a in c["args"]]
                result = compute_op(op, args)

                if "var" in c:
                    state[c["var"]] = result

            # State assertion: {state: {var: value, ...}}
            elif "state" in step:
                for var, expected in step["state"].items():
                    actual = state.get(var, 0)
                    if abs(actual - float(expected)) > tolerance:
                        return False, f"Step {i}: state {var}={actual}, expected {expected}", state, None

            # Given step (rate problems): {given: {var: value, ...}}
            elif "given" in step:
                g = step["given"]
                for k, v in g.items():
                    if isinstance(v, (int, float)):
                        state[k] = float(v)

            # Formula step: {formula: "equation"} - informational only
            elif "formula" in step:
                pass

            # Query step: {query: var}
            elif "query" in step:
                query_var = step["query"]

            else:
                # Unknown step type - skip
                pass

        except KeyError as e:
            return False, f"Step {i}: {e}", state, None
        except Exception as e:
            return False, f"Step {i}: exception {e}", state, None

    # Get answer from query var
    answer = None
    if query_var and query_var in state:
        answer = state[query_var]

    return True, "valid", state, answer


def verify_yaml_output(yaml_str: str, expected_answer: Any = None, tolerance: float = 0.01) -> dict:
    """
    Verify a complete YAML trace output by executing it.

    Returns dict with:
        - parsed: bool
        - expert: str or None
        - trace_valid: bool
        - trace_error: str or None
        - computed_answer: the answer computed by executing trace
        - answer_correct: bool (if expected_answer provided)
    """
    result = {
        "parsed": False,
        "expert": None,
        "trace_valid": False,
        "trace_error": None,
        "computed_answer": None,
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

    # Execute trace
    trace = data.get("trace", [])
    if not isinstance(trace, list):
        result["trace_error"] = "Trace is not a list"
        return result

    valid, error, final_state, computed_answer = execute_trace(trace, tolerance)
    result["trace_valid"] = valid
    result["trace_error"] = error if not valid else None
    result["final_state"] = final_state
    result["computed_answer"] = computed_answer

    # Check answer if expected provided
    if expected_answer is not None and computed_answer is not None:
        try:
            if isinstance(expected_answer, (int, float)) and isinstance(computed_answer, (int, float)):
                result["answer_correct"] = abs(float(computed_answer) - float(expected_answer)) < tolerance
            else:
                result["answer_correct"] = computed_answer == expected_answer
        except:
            result["answer_correct"] = False

    return result


# Quick test
if __name__ == "__main__":
    # Test symbolic entity tracking trace (no results)
    print("Test 1: Entity tracking (symbolic)")
    test_trace = [
        {"init": "eggs", "value": 16},
        {"consume": {"entity": "eggs", "amount": 3}},
        {"consume": {"entity": "eggs", "amount": 4}},
        {"compute": {"op": "mul", "args": ["eggs", 2], "var": "revenue"}},
        {"query": "revenue"},
    ]

    valid, error, state, answer = execute_trace(test_trace)
    print(f"Valid: {valid}")
    print(f"State: {state}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 18, Got: {answer}, Correct: {abs(answer - 18) < 0.01}")

    # Test symbolic arithmetic
    print("\nTest 2: Arithmetic (symbolic)")
    test_trace2 = [
        {"init": "price", "value": 100},
        {"init": "tax", "value": 8.5},
        {"init": "shipping", "value": 5},
        {"compute": {"op": "add", "args": ["price", "tax"], "var": "with_tax"}},
        {"compute": {"op": "add", "args": ["with_tax", "shipping"], "var": "total"}},
        {"query": "total"},
    ]

    valid, error, state, answer = execute_trace(test_trace2)
    print(f"Valid: {valid}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 113.5, Got: {answer}")

    # Test percent off
    print("\nTest 3: Percent off (symbolic)")
    test_trace3 = [
        {"init": "price", "value": 100},
        {"init": "discount_rate", "value": 20},
        {"percent_off": {"base": "price", "rate": "discount_rate", "var": "sale_price"}},
        {"query": "sale_price"},
    ]

    valid, error, state, answer = execute_trace(test_trace3)
    print(f"Valid: {valid}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 80, Got: {answer}")
