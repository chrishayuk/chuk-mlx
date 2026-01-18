"""
Slot Filler for AST Templates

Given a template ID and program hint, fills in the holes with concrete values.

For this experiment we use rule-based filling keyed on (template, program_hint).
This validates the core hypothesis: can template classification generalize
even when slot filling is deterministic?

The key insight: If we train on sum_even and get collatz to classify as
LOOP_CONDITIONAL_ACCUMULATE, we've achieved compositional generalization.
The slot filling for collatz can be rule-based since we know the program.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ast_nodes import (
    Program, Loop, If, SetLocal, Return, Block,
    BinOp, BinOpType, Const, Local, Slot, Expr,
    pretty_print,
)
from templates import TemplateID


# =============================================================================
# SLOT FILLING RULES
# =============================================================================

@dataclass
class SlotValues:
    """Values to fill template slots."""
    values: Dict[str, Any]

    def get(self, name: str, default=None):
        return self.values.get(name, default)


# Define slot values for each (template, program) pair
# The key is (template_id, program_name)

SLOT_RULES: Dict[tuple[TemplateID, str], SlotValues] = {}


# -----------------------------------------------------------------------------
# IF_BRANCH: max_of_two
# result = b; if a > b: result = a; return result
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.IF_BRANCH, "max_of_two")] = SlotValues({
    "default_value": Slot(1),         # result = b
    "cond_op": BinOpType.GT,          # if a > b
    "then_value": Slot(0),            # result = a
    "else_value": Local(0),           # keep result (nop)
})


# -----------------------------------------------------------------------------
# IF_BRANCH: abs_diff
# result = a - b; if a < b: result = b - a; return result
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.IF_BRANCH, "abs_diff")] = SlotValues({
    "default_value": BinOp(BinOpType.SUB, Slot(0), Slot(1)),  # a - b
    "cond_op": BinOpType.LT,          # if a < b
    "then_value": BinOp(BinOpType.SUB, Slot(1), Slot(0)),     # b - a
    "else_value": Local(0),           # keep result (nop)
})


# -----------------------------------------------------------------------------
# LOOP_ACCUMULATE: sum_1_to_n
# acc = 0; i = 1; while i <= n: acc += i; i++; return acc
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_ACCUMULATE, "sum_1_to_n")] = SlotValues({
    "init_acc": Const(0),
    "init_counter": Const(1),
    "acc_op": BinOpType.ADD,
    "acc_operand": Local(1),          # acc += i
    "counter_op": BinOpType.ADD,
    "counter_step": Const(1),
    "continue_op": BinOpType.LE,
    "continue_bound": Slot(0),        # while i <= n
})


# -----------------------------------------------------------------------------
# LOOP_ACCUMULATE: sum_a_to_b
# acc = 0; i = a; while i <= b: acc += i; i++; return acc
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_ACCUMULATE, "sum_a_to_b")] = SlotValues({
    "init_acc": Const(0),
    "init_counter": Slot(0),          # i = a
    "acc_op": BinOpType.ADD,
    "acc_operand": Local(1),          # acc += i
    "counter_op": BinOpType.ADD,
    "counter_step": Const(1),
    "continue_op": BinOpType.LE,
    "continue_bound": Slot(1),        # while i <= b
})


# -----------------------------------------------------------------------------
# LOOP_ACCUMULATE: factorial
# acc = 1; i = 2; while i <= n: acc *= i; i++; return acc
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_ACCUMULATE, "factorial")] = SlotValues({
    "init_acc": Const(1),
    "init_counter": Const(2),
    "acc_op": BinOpType.MUL,
    "acc_operand": Local(1),          # acc *= i
    "counter_op": BinOpType.ADD,
    "counter_step": Const(1),
    "continue_op": BinOpType.LE,
    "continue_bound": Slot(0),        # while i <= n
})


# -----------------------------------------------------------------------------
# LOOP_ACCUMULATE: power
# acc = 1; i = 1; while i <= exp: acc *= base; i++; return acc
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_ACCUMULATE, "power")] = SlotValues({
    "init_acc": Const(1),
    "init_counter": Const(1),
    "acc_op": BinOpType.MUL,
    "acc_operand": Slot(0),           # acc *= base
    "counter_op": BinOpType.ADD,
    "counter_step": Const(1),
    "continue_op": BinOpType.LE,
    "continue_bound": Slot(1),        # while i <= exp
})


# -----------------------------------------------------------------------------
# LOOP_CONDITIONAL_ACCUMULATE: sum_even
# count = 0; i = 1; while i <= n:
#   if i % 2 == 0: count += i else: nop
#   i++
# return count
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "sum_even")] = SlotValues({
    "init_acc": Const(0),
    "init_var": Const(1),             # i = 1
    "cond_op": BinOpType.EQ,          # == 0
    "cond_inner_op": BinOpType.REM,   # i % 2
    "cond_inner_operand": Const(2),
    "cond_compare_value": Const(0),
    "then_target": 0,                 # count
    "then_expr": BinOp(BinOpType.ADD, Local(0), Local(1)),  # count += i
    "else_target": 0,
    "else_expr": Local(0),            # nop: count = count
    "update_target": 1,               # i
    "update_op": BinOpType.ADD,
    "update_source": 1,
    "update_operand": Const(1),       # i++
    "continue_op": BinOpType.LE,
    "continue_bound": Slot(0),        # while i <= n
})


# -----------------------------------------------------------------------------
# LOOP_CONDITIONAL_ACCUMULATE: collatz_length
# count = 0; n = input; while n > 1:
#   if n % 2 != 0: n = 3n + 1 else: n = n / 2
#   count++
# return count
# -----------------------------------------------------------------------------
SLOT_RULES[(TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "collatz_length")] = SlotValues({
    "init_acc": Const(0),
    "init_var": Slot(0),              # n = input
    "cond_op": BinOpType.NE,          # != 0 (for n % 2 != 0, i.e., odd)
    "cond_inner_op": BinOpType.REM,   # n % 2
    "cond_inner_operand": Const(2),
    "cond_compare_value": Const(0),
    "then_target": 1,                 # n (working var)
    # 3n + 1 = (2 + 1) * n + 1 (since we don't have CONST_3)
    "then_expr": BinOp(
        BinOpType.ADD,
        BinOp(
            BinOpType.MUL,
            BinOp(BinOpType.ADD, Const(2), Const(1)),  # 3
            Local(1)                  # n
        ),
        Const(1)                      # + 1
    ),
    "else_target": 1,                 # n
    "else_expr": BinOp(BinOpType.DIV, Local(1), Const(2)),  # n / 2
    "update_target": 0,               # count
    "update_op": BinOpType.ADD,
    "update_source": 0,
    "update_operand": Const(1),       # count++
    "continue_op": BinOpType.GT,
    "continue_bound": Const(1),       # while n > 1
})


# =============================================================================
# SLOT FILLER
# =============================================================================

def fill_slots(
    template_id: TemplateID,
    program_name: str,
    operands: list[int],
) -> Optional[Program]:
    """
    Fill template slots to produce a concrete program AST.

    Args:
        template_id: Which template to use
        program_name: Which program's slot rules to use
        operands: Extracted operands from NL input (for Slot references)

    Returns:
        Filled Program AST, or None if no rules found
    """
    key = (template_id, program_name)
    if key not in SLOT_RULES:
        return None

    rules = SLOT_RULES[key]

    # Build the program based on template
    if template_id == TemplateID.IF_BRANCH:
        return _fill_if_branch(rules, operands)
    elif template_id == TemplateID.LOOP_ACCUMULATE:
        return _fill_loop_accumulate(rules, operands)
    elif template_id == TemplateID.LOOP_CONDITIONAL_ACCUMULATE:
        return _fill_loop_conditional_accumulate(rules, operands)
    else:
        return None


def _fill_if_branch(rules: SlotValues, operands: list[int]) -> Program:
    """Fill IF_BRANCH template."""
    return Program(
        init=[
            SetLocal(0, rules.get("default_value")),
        ],
        body=If(
            cond=BinOp(rules.get("cond_op"), Slot(0), Slot(1)),
            then_block=[
                SetLocal(0, rules.get("then_value")),
            ],
            else_block=[
                SetLocal(0, rules.get("else_value")),
            ]
        ),
        return_expr=Local(0),
    )


def _fill_loop_accumulate(rules: SlotValues, operands: list[int]) -> Program:
    """Fill LOOP_ACCUMULATE template."""
    return Program(
        init=[
            SetLocal(0, rules.get("init_acc")),
            SetLocal(1, rules.get("init_counter")),
        ],
        body=Loop(
            body=[
                SetLocal(0, BinOp(
                    rules.get("acc_op"),
                    Local(0),
                    rules.get("acc_operand"),
                )),
                SetLocal(1, BinOp(
                    rules.get("counter_op"),
                    Local(1),
                    rules.get("counter_step"),
                )),
            ],
            continue_cond=BinOp(
                rules.get("continue_op"),
                Local(1),
                rules.get("continue_bound"),
            ),
        ),
        return_expr=Local(0),
    )


def _fill_loop_conditional_accumulate(rules: SlotValues, operands: list[int]) -> Program:
    """Fill LOOP_CONDITIONAL_ACCUMULATE template."""
    return Program(
        init=[
            SetLocal(0, rules.get("init_acc")),
            SetLocal(1, rules.get("init_var")),
        ],
        body=Loop(
            body=[
                If(
                    cond=BinOp(
                        rules.get("cond_op"),
                        BinOp(
                            rules.get("cond_inner_op"),
                            Local(1),
                            rules.get("cond_inner_operand"),
                        ),
                        rules.get("cond_compare_value"),
                    ),
                    then_block=[
                        SetLocal(rules.get("then_target"), rules.get("then_expr")),
                    ],
                    else_block=[
                        SetLocal(rules.get("else_target"), rules.get("else_expr")),
                    ],
                ),
                SetLocal(
                    rules.get("update_target"),
                    BinOp(
                        rules.get("update_op"),
                        Local(rules.get("update_source")),
                        rules.get("update_operand"),
                    )
                ),
            ],
            continue_cond=BinOp(
                rules.get("continue_op"),
                Local(1),
                rules.get("continue_bound"),
            ),
        ),
        return_expr=Local(0),
    )


# =============================================================================
# PROGRAM HINT DETECTION
# =============================================================================

def detect_program_hint(nl_input: str, template_id: TemplateID) -> str:
    """
    Detect which program's slot rules to use based on NL and template.

    This is the "program hint" that tells us which slot filling to apply.
    For now, we use keyword detection. In a more advanced system,
    this could be a small classifier.
    """
    nl_lower = nl_input.lower()

    if template_id == TemplateID.IF_BRANCH:
        if "max" in nl_lower or "larger" in nl_lower or "bigger" in nl_lower:
            return "max_of_two"
        elif "diff" in nl_lower or "distance" in nl_lower or "abs" in nl_lower:
            return "abs_diff"

    elif template_id == TemplateID.LOOP_ACCUMULATE:
        if "factorial" in nl_lower or "!" in nl_lower:
            return "factorial"
        elif "power" in nl_lower or "^" in nl_lower or "raised" in nl_lower or "**" in nl_lower:
            return "power"
        elif "to" in nl_lower and "sum" in nl_lower:
            # Check if it's sum_a_to_b (two bounds) or sum_1_to_n (one bound)
            import re
            numbers = re.findall(r'\b\d+\b', nl_input)
            if len(numbers) >= 2 and int(numbers[0]) > 1:
                return "sum_a_to_b"
            return "sum_1_to_n"
        else:
            return "sum_1_to_n"

    elif template_id == TemplateID.LOOP_CONDITIONAL_ACCUMULATE:
        if "collatz" in nl_lower:
            return "collatz_length"
        elif "even" in nl_lower:
            return "sum_even"
        else:
            # Default - this is where generalization happens!
            # If the model classifies something new as this template,
            # we need a way to fill slots. For now, use sum_even as default.
            return "sum_even"

    return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Slot Filler Test")
    print("=" * 60)

    # Test each template/program combination
    test_cases = [
        (TemplateID.IF_BRANCH, "max_of_two", [5, 3]),
        (TemplateID.IF_BRANCH, "abs_diff", [10, 3]),
        (TemplateID.LOOP_ACCUMULATE, "sum_1_to_n", [10]),
        (TemplateID.LOOP_ACCUMULATE, "factorial", [5]),
        (TemplateID.LOOP_ACCUMULATE, "power", [2, 10]),
        (TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "sum_even", [10]),
        (TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "collatz_length", [27]),
    ]

    for template_id, program_name, operands in test_cases:
        print(f"\n{template_id.name} / {program_name}:")
        print(f"  Operands: {operands}")

        ast = fill_slots(template_id, program_name, operands)
        if ast:
            print(f"  AST generated successfully")
            print(pretty_print(ast))
        else:
            print(f"  No rules found!")

    print("\n" + "=" * 60)
    print("Program hint detection test:")
    print("=" * 60)

    nl_tests = [
        "Collatz length of 27",
        "Sum of even numbers from 1 to 100",
        "5 factorial",
        "2 to the power of 10",
        "Max of 5 and 3",
    ]

    for nl in nl_tests:
        # Try each template
        for tid in TemplateID:
            hint = detect_program_hint(nl, tid)
            if hint:
                print(f"  '{nl}' + {tid.name} → {hint}")
