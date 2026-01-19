"""
AST Templates for IR Synthesis

Templates are AST skeletons with holes. Each template represents
a structural equivalence class of programs.

Key insight: sum_even and collatz have the SAME template
(LOOP_CONDITIONAL_ACCUMULATE). If the classifier learns
sum_even maps to this template, it should generalize to collatz.

Template IDs:
- SIMPLE_BINOP: Single binary operation on inputs
- IF_BRANCH: Conditional with two branches
- LOOP_ACCUMULATE: Simple loop with accumulation
- LOOP_CONDITIONAL_ACCUMULATE: Loop with conditional inside

Each program maps to exactly one template.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List

from ast_nodes import (
    Program, Loop, If, SetLocal, Return, Block,
    BinOp, BinOpType, Const, Local, Slot, Hole, Expr,
    pretty_print,
)


# =============================================================================
# TEMPLATE ENUM
# =============================================================================

class TemplateID(IntEnum):
    """Template identifiers - the output space of our classifier."""
    SIMPLE_BINOP = 0        # Single operation: op(a, b)
    IF_BRANCH = 1           # if cond then A else B
    LOOP_ACCUMULATE = 2     # while cond: acc = op(acc, x)
    LOOP_CONDITIONAL_ACCUMULATE = 3  # while cond: if test: acc_update


NUM_TEMPLATES = len(TemplateID)


# =============================================================================
# TEMPLATE DEFINITIONS
# =============================================================================

@dataclass
class Template:
    """A template with holes to be filled."""
    id: TemplateID
    name: str
    description: str
    ast: Program
    holes: List[str]  # Names of holes in this template
    programs: List[str]  # Which programs map to this template


# Template: SIMPLE_BINOP
# Structure: return op(slot_0, slot_1)
SIMPLE_BINOP = Template(
    id=TemplateID.SIMPLE_BINOP,
    name="SIMPLE_BINOP",
    description="Single binary operation on two inputs",
    ast=Program(
        init=[],
        body=Return(
            value=BinOp(
                Hole("op", "op"),
                Slot(0),
                Slot(1)
            )
        ),
        return_expr=BinOp(
            Hole("op", "op"),
            Slot(0),
            Slot(1)
        )
    ),
    holes=["op"],
    programs=[],  # No programs use this simple form directly
)


# Template: IF_BRANCH
# Structure: if a > b then a else b (or similar)
IF_BRANCH = Template(
    id=TemplateID.IF_BRANCH,
    name="IF_BRANCH",
    description="Conditional branch selecting between two values",
    ast=Program(
        init=[
            SetLocal(0, Hole("default_value", "expr")),  # Default result
        ],
        body=If(
            cond=BinOp(Hole("cond_op", "op"), Slot(0), Slot(1)),
            then_block=[
                SetLocal(0, Hole("then_value", "expr")),
            ],
            else_block=[
                SetLocal(0, Hole("else_value", "expr")),
            ]
        ),
        return_expr=Local(0),
    ),
    holes=["default_value", "cond_op", "then_value", "else_value"],
    programs=["max_of_two", "abs_diff"],
)


# Template: LOOP_ACCUMULATE
# Structure: acc = init; while cond: acc = op(acc, expr); return acc
LOOP_ACCUMULATE = Template(
    id=TemplateID.LOOP_ACCUMULATE,
    name="LOOP_ACCUMULATE",
    description="Loop with simple accumulation",
    ast=Program(
        init=[
            SetLocal(0, Hole("init_acc", "const")),      # acc = 0 or 1
            SetLocal(1, Hole("init_counter", "expr")),   # i = 1 or slot
        ],
        body=Loop(
            body=[
                # acc = op(acc, expr)
                SetLocal(0, BinOp(
                    Hole("acc_op", "op"),
                    Local(0),
                    Hole("acc_operand", "expr"),
                )),
                # i = i + 1 (or i - 1)
                SetLocal(1, BinOp(
                    Hole("counter_op", "op"),
                    Local(1),
                    Hole("counter_step", "const"),
                )),
            ],
            continue_cond=BinOp(
                Hole("continue_op", "op"),
                Local(1),
                Hole("continue_bound", "expr"),
            ),
        ),
        return_expr=Local(0),
    ),
    holes=[
        "init_acc", "init_counter",
        "acc_op", "acc_operand",
        "counter_op", "counter_step",
        "continue_op", "continue_bound"
    ],
    programs=["sum_1_to_n", "sum_a_to_b", "factorial", "power"],
)


# Template: LOOP_CONDITIONAL_ACCUMULATE
# Structure: while cond: if test: branch_a else: branch_b; counter++
# This is the KEY template - both sum_even AND collatz use this!
LOOP_CONDITIONAL_ACCUMULATE = Template(
    id=TemplateID.LOOP_CONDITIONAL_ACCUMULATE,
    name="LOOP_CONDITIONAL_ACCUMULATE",
    description="Loop with conditional branching inside",
    ast=Program(
        init=[
            SetLocal(0, Hole("init_acc", "const")),      # counter/acc = 0
            SetLocal(1, Hole("init_var", "expr")),       # working var = slot or const
        ],
        body=Loop(
            body=[
                # if condition (e.g., n % 2 == 0)
                If(
                    cond=BinOp(
                        Hole("cond_op", "op"),
                        BinOp(
                            Hole("cond_inner_op", "op"),
                            Local(1),
                            Hole("cond_inner_operand", "const"),
                        ),
                        Hole("cond_compare_value", "const"),
                    ),
                    then_block=[
                        SetLocal(Hole("then_target", "local"), Hole("then_expr", "expr")),
                    ],
                    else_block=[
                        SetLocal(Hole("else_target", "local"), Hole("else_expr", "expr")),
                    ],
                ),
                # Update counter: acc = acc + 1 (for collatz counting)
                # Or this might be different for sum_even
                SetLocal(Hole("update_target", "local"), BinOp(
                    Hole("update_op", "op"),
                    Local(Hole("update_source", "local")),
                    Hole("update_operand", "expr"),
                )),
            ],
            continue_cond=BinOp(
                Hole("continue_op", "op"),
                Local(1),
                Hole("continue_bound", "expr"),
            ),
        ),
        return_expr=Local(0),
    ),
    holes=[
        "init_acc", "init_var",
        "cond_op", "cond_inner_op", "cond_inner_operand", "cond_compare_value",
        "then_target", "then_expr",
        "else_target", "else_expr",
        "update_target", "update_op", "update_source", "update_operand",
        "continue_op", "continue_bound"
    ],
    programs=["sum_even", "collatz_length"],  # KEY: Both share this template!
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

ALL_TEMPLATES = {
    TemplateID.SIMPLE_BINOP: SIMPLE_BINOP,
    TemplateID.IF_BRANCH: IF_BRANCH,
    TemplateID.LOOP_ACCUMULATE: LOOP_ACCUMULATE,
    TemplateID.LOOP_CONDITIONAL_ACCUMULATE: LOOP_CONDITIONAL_ACCUMULATE,
}

# Map program names to template IDs
PROGRAM_TO_TEMPLATE: Dict[str, TemplateID] = {
    # IF_BRANCH programs
    "max_of_two": TemplateID.IF_BRANCH,
    "abs_diff": TemplateID.IF_BRANCH,

    # LOOP_ACCUMULATE programs
    "sum_1_to_n": TemplateID.LOOP_ACCUMULATE,
    "sum_a_to_b": TemplateID.LOOP_ACCUMULATE,
    "factorial": TemplateID.LOOP_ACCUMULATE,
    "power": TemplateID.LOOP_ACCUMULATE,

    # LOOP_CONDITIONAL_ACCUMULATE programs
    # THIS IS THE KEY INSIGHT:
    # sum_even and collatz share the same structural template!
    "sum_even": TemplateID.LOOP_CONDITIONAL_ACCUMULATE,
    "collatz_length": TemplateID.LOOP_CONDITIONAL_ACCUMULATE,
}


def get_template(template_id: TemplateID) -> Template:
    """Get template by ID."""
    return ALL_TEMPLATES[template_id]


def get_program_template(program_name: str) -> TemplateID:
    """Get template ID for a program name."""
    return PROGRAM_TO_TEMPLATE[program_name]


def template_name(template_id: TemplateID) -> str:
    """Get human-readable template name."""
    return ALL_TEMPLATES[template_id].name


# =============================================================================
# DISPLAY
# =============================================================================

def display_template_mapping():
    """Display the program to template mapping."""
    print("=" * 60)
    print("PROGRAM → TEMPLATE MAPPING")
    print("=" * 60)

    for tid in TemplateID:
        template = ALL_TEMPLATES[tid]
        print(f"\n{template.name} ({tid.value}):")
        print(f"  Description: {template.description}")
        print(f"  Programs: {template.programs}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("  sum_even and collatz_length map to the SAME template!")
    print("  LOOP_CONDITIONAL_ACCUMULATE")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    display_template_mapping()

    # Print template ASTs
    print("\n" + "=" * 60)
    print("TEMPLATE AST STRUCTURES")
    print("=" * 60)

    for tid in TemplateID:
        template = ALL_TEMPLATES[tid]
        print(f"\n{template.name}:")
        print("-" * 40)
        print(pretty_print(template.ast))
        print(f"\nHoles: {template.holes}")
