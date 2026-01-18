"""
AST to IR Linearizer

Compiles AST programs to IR opcode sequences, which can then be
compiled to WASM for execution.

This is a deterministic compiler - given an AST, the IR sequence
is fully determined. The "learning" happens in template classification,
not here.
"""

from typing import List

import sys
from pathlib import Path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

from ir_emission.shared import IROpcode, encode_i32_const, OPCODE_TO_WASM, WASMRuntime

from ast_nodes import (
    Program, Loop, If, SetLocal, Return, Block,
    BinOp, BinOpType, Const, Local, Slot, Expr,
    pretty_print,
)


# =============================================================================
# OPERATION MAPPING
# =============================================================================

BINOP_TO_IR = {
    BinOpType.ADD: IROpcode.I32_ADD,
    BinOpType.SUB: IROpcode.I32_SUB,
    BinOpType.MUL: IROpcode.I32_MUL,
    BinOpType.DIV: IROpcode.I32_DIV_S,
    BinOpType.REM: IROpcode.I32_REM_S,
    BinOpType.EQ: IROpcode.I32_EQ,
    BinOpType.NE: IROpcode.I32_NE,
    BinOpType.LT: IROpcode.I32_LT_S,
    BinOpType.GT: IROpcode.I32_GT_S,
    BinOpType.LE: IROpcode.I32_LE_S,
    BinOpType.GE: IROpcode.I32_GE_S,
}

LOCAL_GET = {
    0: IROpcode.LOCAL_GET_0,
    1: IROpcode.LOCAL_GET_1,
}

LOCAL_SET = {
    0: IROpcode.LOCAL_SET_0,
    1: IROpcode.LOCAL_SET_1,
}

SLOT_IR = {
    0: IROpcode.SLOT_0,
    1: IROpcode.SLOT_1,
    2: IROpcode.SLOT_2,
    3: IROpcode.SLOT_3,
}

CONST_IR = {
    0: IROpcode.CONST_0,
    1: IROpcode.CONST_1,
    2: IROpcode.CONST_2,
    10: IROpcode.CONST_10,
}


# =============================================================================
# COMPILER
# =============================================================================

class IRCompiler:
    """Compiles AST to IR opcode sequence."""

    def __init__(self):
        self.ir: List[int] = []

    def compile(self, program: Program) -> List[int]:
        """Compile a Program AST to IR opcodes."""
        self.ir = [IROpcode.START]

        # Compile init block
        for stmt in program.init:
            self.compile_stmt(stmt)

        # Compile body
        self.compile_stmt(program.body)

        # Compile return expression
        self.compile_expr(program.return_expr)

        self.ir.append(IROpcode.END)
        return self.ir

    def compile_stmt(self, stmt) -> None:
        """Compile a statement."""
        if isinstance(stmt, SetLocal):
            self.compile_expr(stmt.value)
            if stmt.index in LOCAL_SET:
                self.ir.append(LOCAL_SET[stmt.index])
            else:
                raise ValueError(f"Unknown local index: {stmt.index}")

        elif isinstance(stmt, If):
            # Compile condition
            self.compile_expr(stmt.cond)
            self.ir.append(IROpcode.IF_BEGIN)

            # Then block
            for s in stmt.then_block:
                self.compile_stmt(s)

            # Else block
            self.ir.append(IROpcode.ELSE)
            for s in stmt.else_block:
                self.compile_stmt(s)

            self.ir.append(IROpcode.IF_END)

        elif isinstance(stmt, Loop):
            self.ir.append(IROpcode.LOOP_BEGIN)

            # Loop body
            for s in stmt.body:
                self.compile_stmt(s)

            # Continue condition
            self.compile_expr(stmt.continue_cond)
            self.ir.append(IROpcode.BR_IF)

            self.ir.append(IROpcode.LOOP_END)

        elif isinstance(stmt, Return):
            self.compile_expr(stmt.value)
            # Return is implicit (value on stack at end)

        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def compile_expr(self, expr) -> None:
        """Compile an expression."""
        if isinstance(expr, Const):
            if expr.value in CONST_IR:
                self.ir.append(CONST_IR[expr.value])
            else:
                # For constants not in our small set, we'd need CONST_10
                # or extend the codebook. For now, handle small constants.
                self.ir.append(IROpcode.CONST_10)  # Placeholder

        elif isinstance(expr, Local):
            if expr.index in LOCAL_GET:
                self.ir.append(LOCAL_GET[expr.index])
            else:
                raise ValueError(f"Unknown local index: {expr.index}")

        elif isinstance(expr, Slot):
            if expr.index in SLOT_IR:
                self.ir.append(SLOT_IR[expr.index])
            else:
                raise ValueError(f"Unknown slot index: {expr.index}")

        elif isinstance(expr, BinOp):
            # Compile left operand
            self.compile_expr(expr.left)
            # Compile right operand
            self.compile_expr(expr.right)
            # Emit operation
            if expr.op in BINOP_TO_IR:
                self.ir.append(BINOP_TO_IR[expr.op])
            else:
                raise ValueError(f"Unknown binary operation: {expr.op}")

        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")


def linearize(program: Program) -> List[int]:
    """Compile Program AST to IR opcodes."""
    compiler = IRCompiler()
    return compiler.compile(program)


# =============================================================================
# IR TO WASM
# =============================================================================

def ir_to_wasm(ir_opcodes: List[int], operands: List[int]) -> bytes:
    """
    Convert IR opcodes to WASM bytecode.

    This is based on the existing IRCodebook.indices_to_wasm()
    but works directly with our opcode list.
    """
    wasm_bytes = bytearray()

    for opcode in ir_opcodes:
        op = IROpcode(opcode)

        if op in (IROpcode.PAD, IROpcode.START, IROpcode.END):
            continue

        elif op == IROpcode.SLOT_0 and len(operands) > 0:
            wasm_bytes.extend(encode_i32_const(operands[0]))
        elif op == IROpcode.SLOT_1 and len(operands) > 1:
            wasm_bytes.extend(encode_i32_const(operands[1]))
        elif op == IROpcode.SLOT_2 and len(operands) > 2:
            wasm_bytes.extend(encode_i32_const(operands[2]))
        elif op == IROpcode.SLOT_3 and len(operands) > 3:
            wasm_bytes.extend(encode_i32_const(operands[3]))

        elif op == IROpcode.CONST_0:
            wasm_bytes.extend(encode_i32_const(0))
        elif op == IROpcode.CONST_1:
            wasm_bytes.extend(encode_i32_const(1))
        elif op == IROpcode.CONST_2:
            wasm_bytes.extend(encode_i32_const(2))
        elif op == IROpcode.CONST_10:
            wasm_bytes.extend(encode_i32_const(10))

        elif op in OPCODE_TO_WASM:
            wasm_bytes.extend(OPCODE_TO_WASM[op])

        elif op == IROpcode.DUP:
            # Duplicate: local.tee 0, local.get 0
            wasm_bytes.extend(bytes([0x22, 0x00, 0x20, 0x00]))

    return bytes(wasm_bytes)


def execute_ir(ir_opcodes: List[int], operands: List[int]) -> tuple[bool, int, str]:
    """
    Execute IR opcodes with given operands.

    Returns:
        (success, result, error_message)
    """
    try:
        wasm_bytes = ir_to_wasm(ir_opcodes, operands)
        runtime = WASMRuntime(use_native=True)
        result = runtime.execute(wasm_bytes)
        return result.success, result.result, result.error
    except Exception as e:
        return False, None, str(e)


def compile_and_execute(program: Program, operands: List[int]) -> tuple[bool, int, str]:
    """
    Compile AST and execute with operands.

    Returns:
        (success, result, error_message)
    """
    ir_opcodes = linearize(program)
    return execute_ir(ir_opcodes, operands)


# =============================================================================
# VISUALIZATION
# =============================================================================

def ir_to_string(ir_opcodes: List[int]) -> str:
    """Convert IR opcodes to human-readable string."""
    names = []
    for op in ir_opcodes:
        names.append(IROpcode(op).name)
    return " → ".join(names)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from slot_filler import fill_slots
    from templates import TemplateID

    print("=" * 60)
    print("AST Linearizer Test")
    print("=" * 60)

    # Test cases: (template, program, operands, expected_result)
    test_cases = [
        (TemplateID.IF_BRANCH, "max_of_two", [5, 3], 5),
        (TemplateID.IF_BRANCH, "max_of_two", [3, 7], 7),
        (TemplateID.IF_BRANCH, "abs_diff", [10, 3], 7),
        (TemplateID.IF_BRANCH, "abs_diff", [3, 10], 7),
        (TemplateID.LOOP_ACCUMULATE, "sum_1_to_n", [10], 55),
        (TemplateID.LOOP_ACCUMULATE, "sum_1_to_n", [100], 5050),
        (TemplateID.LOOP_ACCUMULATE, "factorial", [5], 120),
        (TemplateID.LOOP_ACCUMULATE, "power", [2, 10], 1024),
        (TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "sum_even", [10], 30),
        (TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "collatz_length", [27], 111),
        (TemplateID.LOOP_CONDITIONAL_ACCUMULATE, "collatz_length", [7], 16),
    ]

    passed = 0
    failed = 0

    for template_id, program_name, operands, expected in test_cases:
        # Fill template
        ast = fill_slots(template_id, program_name, operands)
        if ast is None:
            print(f"FAIL: {program_name}({operands}) - no slot rules")
            failed += 1
            continue

        # Linearize to IR
        ir_opcodes = linearize(ast)

        # Execute
        success, result, error = execute_ir(ir_opcodes, operands)

        if success and result == expected:
            print(f"PASS: {program_name}({operands}) = {result}")
            passed += 1
        else:
            print(f"FAIL: {program_name}({operands}) = {result} (expected {expected})")
            if error:
                print(f"      Error: {error}")
            print(f"      IR: {ir_to_string(ir_opcodes)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)
