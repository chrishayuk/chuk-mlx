"""
AST Node Definitions for IR Synthesis

This module defines the Abstract Syntax Tree (AST) nodes that represent
program structure. These nodes can be linearized to IR opcodes.

Key insight: We use AST templates with "holes" (slots) that get filled
to produce concrete programs. This is the structured intermediate
representation between template classification and IR emission.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, List


# =============================================================================
# BINARY OPERATIONS
# =============================================================================

class BinOpType(Enum):
    """Binary operation types."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    REM = auto()  # Remainder/modulo
    EQ = auto()   # Equal
    NE = auto()   # Not equal
    LT = auto()   # Less than
    GT = auto()   # Greater than
    LE = auto()   # Less than or equal
    GE = auto()   # Greater than or equal


# =============================================================================
# EXPRESSION NODES
# =============================================================================

@dataclass(frozen=True)
class Const:
    """Integer constant."""
    value: int


@dataclass(frozen=True)
class Local:
    """Local variable reference."""
    index: int  # 0 = accumulator, 1 = counter/working var


@dataclass(frozen=True)
class Slot:
    """Operand slot - filled with extracted number from NL input."""
    index: int  # SLOT_0, SLOT_1, etc.


@dataclass(frozen=True)
class BinOp:
    """Binary operation."""
    op: BinOpType
    left: 'Expr'
    right: 'Expr'


# Expression is one of these types
Expr = Union[Const, Local, Slot, BinOp]


# =============================================================================
# STATEMENT NODES
# =============================================================================

@dataclass
class SetLocal:
    """Set a local variable."""
    index: int
    value: Expr


@dataclass
class If:
    """Conditional branch."""
    cond: Expr
    then_block: 'Block'
    else_block: 'Block'


@dataclass
class Loop:
    """Loop with continue condition."""
    body: 'Block'
    continue_cond: Expr  # Continue while this is true
    return_expr: Expr | None = None  # Optional return value after loop


@dataclass
class Return:
    """Return a value."""
    value: Expr


# Statement is one of these types
Stmt = Union[SetLocal, If, Loop, Return]

# Block is a list of statements
Block = List[Stmt]


# =============================================================================
# PROGRAM NODE
# =============================================================================

@dataclass
class Program:
    """
    A complete program.

    Structure:
      init: Initialize variables
      body: Main computation (usually a loop or if)
      return_expr: Final return value
    """
    init: Block
    body: Stmt
    return_expr: Expr


# =============================================================================
# HOLE MARKERS (for templates)
# =============================================================================

@dataclass(frozen=True)
class Hole:
    """
    A hole in a template that gets filled during slot filling.

    Different hole types constrain what can fill them:
    - 'expr': any expression
    - 'const': constant value
    - 'local': local variable index
    - 'op': binary operation type
    - 'slot': operand slot index
    """
    name: str
    hole_type: str = 'expr'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_hole(node) -> bool:
    """Check if a node is a hole."""
    return isinstance(node, Hole)


def collect_holes(node) -> list[Hole]:
    """Collect all holes in an AST node."""
    holes = []

    def visit(n):
        if isinstance(n, Hole):
            holes.append(n)
        elif isinstance(n, (BinOp,)):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, SetLocal):
            visit(n.value)
        elif isinstance(n, If):
            visit(n.cond)
            for stmt in n.then_block:
                visit(stmt)
            for stmt in n.else_block:
                visit(stmt)
        elif isinstance(n, Loop):
            for stmt in n.body:
                visit(stmt)
            visit(n.continue_cond)
            if n.return_expr:
                visit(n.return_expr)
        elif isinstance(n, Program):
            for stmt in n.init:
                visit(stmt)
            visit(n.body)
            visit(n.return_expr)

    visit(node)
    return holes


def substitute_holes(node, substitutions: dict[str, Expr]):
    """
    Substitute holes in an AST with concrete values.

    Args:
        node: AST node (possibly containing holes)
        substitutions: Map from hole names to values

    Returns:
        New AST node with holes replaced
    """
    def subst(n):
        if isinstance(n, Hole):
            if n.name in substitutions:
                return substitutions[n.name]
            raise ValueError(f"No substitution for hole: {n.name}")

        elif isinstance(n, (Const, Local, Slot)):
            return n

        elif isinstance(n, BinOp):
            return BinOp(
                op=n.op if not isinstance(n.op, Hole) else substitutions[n.op.name],
                left=subst(n.left),
                right=subst(n.right)
            )

        elif isinstance(n, SetLocal):
            return SetLocal(
                index=n.index if not isinstance(n.index, Hole) else substitutions[n.index.name],
                value=subst(n.value)
            )

        elif isinstance(n, If):
            return If(
                cond=subst(n.cond),
                then_block=[subst(s) for s in n.then_block],
                else_block=[subst(s) for s in n.else_block]
            )

        elif isinstance(n, Loop):
            return Loop(
                body=[subst(s) for s in n.body],
                continue_cond=subst(n.continue_cond),
                return_expr=subst(n.return_expr) if n.return_expr else None
            )

        elif isinstance(n, Return):
            return Return(value=subst(n.value))

        elif isinstance(n, Program):
            return Program(
                init=[subst(s) for s in n.init],
                body=subst(n.body),
                return_expr=subst(n.return_expr)
            )

        elif isinstance(n, list):
            return [subst(item) for item in n]

        else:
            return n

    return subst(node)


def pretty_print(node, indent: int = 0) -> str:
    """Pretty print an AST node."""
    pad = "  " * indent

    if isinstance(node, Const):
        return f"{node.value}"
    elif isinstance(node, Local):
        return f"local_{node.index}"
    elif isinstance(node, Slot):
        return f"SLOT_{node.index}"
    elif isinstance(node, Hole):
        return f"<{node.name}>"
    elif isinstance(node, BinOp):
        op_str = node.op.name if isinstance(node.op, BinOpType) else f"<{node.op.name}>"
        return f"({pretty_print(node.left)} {op_str} {pretty_print(node.right)})"
    elif isinstance(node, SetLocal):
        return f"{pad}local_{node.index} = {pretty_print(node.value)}"
    elif isinstance(node, If):
        lines = [f"{pad}if {pretty_print(node.cond)}:"]
        for stmt in node.then_block:
            lines.append(pretty_print(stmt, indent + 1))
        if node.else_block:
            lines.append(f"{pad}else:")
            for stmt in node.else_block:
                lines.append(pretty_print(stmt, indent + 1))
        return "\n".join(lines)
    elif isinstance(node, Loop):
        lines = [f"{pad}while {pretty_print(node.continue_cond)}:"]
        for stmt in node.body:
            lines.append(pretty_print(stmt, indent + 1))
        if node.return_expr:
            lines.append(f"{pad}return {pretty_print(node.return_expr)}")
        return "\n".join(lines)
    elif isinstance(node, Return):
        return f"{pad}return {pretty_print(node.value)}"
    elif isinstance(node, Program):
        lines = ["Program:"]
        lines.append(f"{pad}  init:")
        for stmt in node.init:
            lines.append(pretty_print(stmt, indent + 2))
        lines.append(f"{pad}  body:")
        lines.append(pretty_print(node.body, indent + 2))
        lines.append(f"{pad}  return: {pretty_print(node.return_expr)}")
        return "\n".join(lines)
    else:
        return str(node)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # Test creating a simple program: sum 1 to n
    # acc = 0; i = 1; while i <= n: acc += i; i++; return acc

    sum_program = Program(
        init=[
            SetLocal(0, Const(0)),  # acc = 0
            SetLocal(1, Const(1)),  # i = 1
        ],
        body=Loop(
            body=[
                SetLocal(0, BinOp(BinOpType.ADD, Local(0), Local(1))),  # acc += i
                SetLocal(1, BinOp(BinOpType.ADD, Local(1), Const(1))),  # i++
            ],
            continue_cond=BinOp(BinOpType.LE, Local(1), Slot(0)),  # while i <= n
        ),
        return_expr=Local(0),  # return acc
    )

    print("Sum 1 to N program:")
    print(pretty_print(sum_program))
    print()

    # Test with holes
    template = Program(
        init=[
            SetLocal(0, Hole("init_acc", "const")),
            SetLocal(1, Hole("init_counter", "expr")),
        ],
        body=Loop(
            body=[
                SetLocal(0, BinOp(
                    Hole("acc_op", "op"),
                    Local(0),
                    Hole("acc_update", "expr")
                )),
                SetLocal(1, BinOp(BinOpType.ADD, Local(1), Const(1))),
            ],
            continue_cond=BinOp(
                Hole("cond_op", "op"),
                Local(1),
                Hole("cond_bound", "expr")
            ),
        ),
        return_expr=Local(0),
    )

    print("Template with holes:")
    print(pretty_print(template))
    print()

    holes = collect_holes(template)
    print(f"Holes found: {[h.name for h in holes]}")
