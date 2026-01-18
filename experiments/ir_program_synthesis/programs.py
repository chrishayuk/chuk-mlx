"""
IR Program Library

Defines correct IR opcode sequences for various algorithms.
These compile to WASM via IRCodebook.indices_to_wasm().

Each program is a list of IROpcode values that implement the algorithm.
SLOT_0, SLOT_1, etc. are filled with extracted operands at compile time.

Local variable convention:
- LOCAL_0: accumulator / result
- LOCAL_1: loop counter / working variable
"""

from dataclasses import dataclass
from enum import IntEnum

# Import from shared codebook
import sys
from pathlib import Path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))
from ir_emission.shared import IROpcode, encode_i32_const, OPCODE_TO_WASM, WASMRuntime


# =============================================================================
# PROGRAM DEFINITIONS
# =============================================================================

@dataclass
class IRProgram:
    """An IR program with metadata."""
    name: str
    description: str
    opcodes: list[int]
    num_operands: int  # How many SLOT_N values are used
    test_cases: list[tuple[list[int], int]]  # (operands, expected_result)


# Sum from 1 to N
# acc = 0; for i = 1 to N: acc += i; return acc
# Note: We need LOCAL_TEE_1 but don't have it, so we use a workaround
SUM_1_TO_N = IRProgram(
    name="sum_1_to_n",
    description="Sum integers from 1 to N",
    opcodes=[
        IROpcode.START,
        # acc = 0
        IROpcode.CONST_0,
        IROpcode.LOCAL_SET_0,
        # i = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_1,
        # loop
        IROpcode.LOOP_BEGIN,
        # acc += i
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_0,
        # i++ and store (using local.tee workaround)
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        # local.tee 1 would be 0x22 0x01 - need to add this
        IROpcode.LOCAL_SET_1,
        # if i <= N: continue
        IROpcode.LOCAL_GET_1,
        IROpcode.SLOT_0,  # N
        IROpcode.I32_LE_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return acc
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([10], 55),
        ([100], 5050),
        ([1000], 500500),
    ],
)


# Sum from A to B
# acc = 0; for i = A to B: acc += i; return acc
SUM_A_TO_B = IRProgram(
    name="sum_a_to_b",
    description="Sum integers from A to B",
    opcodes=[
        IROpcode.START,
        # acc = 0
        IROpcode.CONST_0,
        IROpcode.LOCAL_SET_0,
        # i = A
        IROpcode.SLOT_0,
        IROpcode.LOCAL_SET_1,
        # loop
        IROpcode.LOOP_BEGIN,
        # acc += i
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_0,
        # i++
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_1,
        # if i <= B: continue
        IROpcode.LOCAL_GET_1,
        IROpcode.SLOT_1,  # B
        IROpcode.I32_LE_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return acc
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=2,
    test_cases=[
        ([1, 10], 55),
        ([5, 15], 110),
        ([1, 100], 5050),
    ],
)


# Factorial: N!
# acc = 1; for i = 2 to N: acc *= i; return acc
FACTORIAL = IRProgram(
    name="factorial",
    description="Compute N factorial (N!)",
    opcodes=[
        IROpcode.START,
        # acc = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_0,
        # i = 2
        IROpcode.CONST_2,
        IROpcode.LOCAL_SET_1,
        # loop
        IROpcode.LOOP_BEGIN,
        # acc *= i
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_MUL,
        IROpcode.LOCAL_SET_0,
        # i++
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_1,
        # if i <= N: continue
        IROpcode.LOCAL_GET_1,
        IROpcode.SLOT_0,  # N
        IROpcode.I32_LE_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return acc
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([5], 120),
        ([6], 720),
        ([10], 3628800),
    ],
)


# Power: base^exp
# acc = 1; for i = 1 to exp: acc *= base; return acc
POWER = IRProgram(
    name="power",
    description="Compute base raised to exponent",
    opcodes=[
        IROpcode.START,
        # acc = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_0,
        # i = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_1,
        # loop
        IROpcode.LOOP_BEGIN,
        # acc *= base
        IROpcode.LOCAL_GET_0,
        IROpcode.SLOT_0,  # base
        IROpcode.I32_MUL,
        IROpcode.LOCAL_SET_0,
        # i++
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_1,
        # if i <= exp: continue
        IROpcode.LOCAL_GET_1,
        IROpcode.SLOT_1,  # exp
        IROpcode.I32_LE_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return acc
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=2,
    test_cases=[
        ([2, 10], 1024),
        ([3, 4], 81),
        ([2, 20], 1048576),
    ],
)


# Fibonacci: fib(N)
# a = 0, b = 1; for i = 2 to N: c = a + b; a = b; b = c; return b
# Using: LOCAL_0 = a, LOCAL_1 = b (need to be creative with limited locals)
FIBONACCI = IRProgram(
    name="fibonacci",
    description="Compute Nth Fibonacci number",
    opcodes=[
        IROpcode.START,
        # Handle N <= 1 case would need IF, skip for now
        # a (on stack) = 0
        IROpcode.CONST_0,
        IROpcode.LOCAL_SET_0,  # a = 0
        # b = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_1,  # b = 1
        # i = 2 (we'll decrement N instead)
        # Actually, let's count down from N to 2
        IROpcode.SLOT_0,  # N on stack, will be our counter
        # loop N-1 times
        IROpcode.LOOP_BEGIN,
        # new_b = a + b
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_ADD,
        # a = b
        IROpcode.LOCAL_GET_1,
        IROpcode.LOCAL_SET_0,
        # b = new_b (still on stack)
        IROpcode.LOCAL_SET_1,
        # decrement counter
        IROpcode.CONST_1,
        IROpcode.I32_SUB,
        IROpcode.DUP,  # keep counter on stack
        IROpcode.CONST_2,  # compare with 2
        IROpcode.I32_GE_S,
        IROpcode.BR_IF,
        IROpcode.DROP,  # drop counter
        IROpcode.LOOP_END,
        # return b
        IROpcode.LOCAL_GET_1,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([10], 55),
        ([20], 6765),
        ([30], 832040),
    ],
)


# Collatz length: count steps until N reaches 1
# count = 0; while n > 1: if n % 2 != 0: n = 3n+1 else n = n/2; count++; return count
# Note: IF_BEGIN branches when condition is TRUE (non-zero), so we check n%2 != 0 for odd case
COLLATZ_LENGTH = IRProgram(
    name="collatz_length",
    description="Count Collatz sequence steps until reaching 1",
    opcodes=[
        IROpcode.START,
        # count = 0
        IROpcode.CONST_0,
        IROpcode.LOCAL_SET_0,
        # n = input
        IROpcode.SLOT_0,
        IROpcode.LOCAL_SET_1,
        # while n > 1
        IROpcode.LOOP_BEGIN,
        # n % 2
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_2,
        IROpcode.I32_REM_S,
        # if n % 2 != 0 (odd)
        IROpcode.IF_BEGIN,
        # n = 3n + 1
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_2,   # We don't have CONST_3, so use 2+1
        IROpcode.CONST_1,
        IROpcode.I32_ADD,   # Now we have 3 on stack
        IROpcode.I32_MUL,   # 3 * n
        IROpcode.CONST_1,
        IROpcode.I32_ADD,   # 3n + 1
        IROpcode.LOCAL_SET_1,
        IROpcode.ELSE,
        # n = n / 2 (even case)
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_2,
        IROpcode.I32_DIV_S,
        IROpcode.LOCAL_SET_1,
        IROpcode.IF_END,
        # count++
        IROpcode.LOCAL_GET_0,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_0,
        # if n > 1: continue loop
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_GT_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return count
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([27], 111),
        ([7], 16),
        ([1], 0),
    ],
)


# GCD using Euclidean algorithm
# while b != 0: t = b; b = a % b; a = t; return a
GCD = IRProgram(
    name="gcd",
    description="Greatest common divisor of A and B",
    opcodes=[
        IROpcode.START,
        # a = SLOT_0
        IROpcode.SLOT_0,
        IROpcode.LOCAL_SET_0,
        # b = SLOT_1
        IROpcode.SLOT_1,
        IROpcode.LOCAL_SET_1,
        # while b != 0
        IROpcode.LOOP_BEGIN,
        # check b == 0
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_0,
        IROpcode.I32_EQ,
        IROpcode.BR_IF,  # exit if b == 0
        # t = b (keep on stack)
        IROpcode.LOCAL_GET_1,
        # b = a % b
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_REM_S,
        IROpcode.LOCAL_SET_1,
        # a = t (from stack)
        IROpcode.LOCAL_SET_0,
        # continue loop
        IROpcode.BR,
        IROpcode.LOOP_END,
        # return a
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=2,
    test_cases=[
        ([48, 18], 6),
        ([100, 35], 5),
        ([17, 13], 1),
    ],
)


# Is Prime (trial division up to sqrt(n))
# for i = 2; i * i <= n; i++: if n % i == 0: return 0; return 1
IS_PRIME = IRProgram(
    name="is_prime",
    description="Check if N is prime (returns 1 if prime, 0 if not)",
    opcodes=[
        IROpcode.START,
        # n = input
        IROpcode.SLOT_0,
        IROpcode.LOCAL_SET_1,
        # i = 2
        IROpcode.CONST_2,
        IROpcode.LOCAL_SET_0,
        # default result = 1 (prime)
        # loop
        IROpcode.LOOP_BEGIN,
        # check i * i <= n
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_0,
        IROpcode.I32_MUL,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_GT_S,
        IROpcode.BR_IF,  # if i*i > n, exit (it's prime)
        # check n % i == 0
        IROpcode.LOCAL_GET_1,
        IROpcode.LOCAL_GET_0,
        IROpcode.I32_REM_S,
        IROpcode.CONST_0,
        IROpcode.I32_EQ,
        IROpcode.IF_BEGIN,
        # divisible - not prime, return 0
        IROpcode.CONST_0,
        IROpcode.END,  # early return
        IROpcode.IF_END,
        # i++
        IROpcode.LOCAL_GET_0,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_0,
        # continue
        IROpcode.BR,
        IROpcode.LOOP_END,
        # return 1 (prime)
        IROpcode.CONST_1,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([7], 1),
        ([11], 1),
        ([12], 0),
        ([97], 1),
    ],
)


# Max of two numbers (uses IF/ELSE with locals)
# result = b; if a > b: result = a; return result
MAX_OF_TWO = IRProgram(
    name="max_of_two",
    description="Return the maximum of two numbers",
    opcodes=[
        IROpcode.START,
        # result = b (default)
        IROpcode.SLOT_1,
        IROpcode.LOCAL_SET_0,
        # Compare a > b
        IROpcode.SLOT_0,
        IROpcode.SLOT_1,
        IROpcode.I32_GT_S,
        IROpcode.IF_BEGIN,
        # true branch: result = a
        IROpcode.SLOT_0,
        IROpcode.LOCAL_SET_0,
        IROpcode.ELSE,
        # false branch: keep result = b (nop)
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_SET_0,
        IROpcode.IF_END,
        # return result
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=2,
    test_cases=[
        ([5, 3], 5),
        ([3, 7], 7),
        ([10, 10], 10),
    ],
)


# Absolute difference: |a - b|
# result = a - b; if a < b: result = b - a; return result
ABS_DIFF = IRProgram(
    name="abs_diff",
    description="Absolute difference between two numbers",
    opcodes=[
        IROpcode.START,
        # result = a - b (default)
        IROpcode.SLOT_0,
        IROpcode.SLOT_1,
        IROpcode.I32_SUB,
        IROpcode.LOCAL_SET_0,
        # Compare a < b
        IROpcode.SLOT_0,
        IROpcode.SLOT_1,
        IROpcode.I32_LT_S,
        IROpcode.IF_BEGIN,
        # true: result = b - a
        IROpcode.SLOT_1,
        IROpcode.SLOT_0,
        IROpcode.I32_SUB,
        IROpcode.LOCAL_SET_0,
        IROpcode.ELSE,
        # false: keep result (nop)
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_SET_0,
        IROpcode.IF_END,
        # return result
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=2,
    test_cases=[
        ([10, 3], 7),
        ([3, 10], 7),
        ([5, 5], 0),
    ],
)


# Count down with conditional (loop + if/else)
# count = 0; for i = n down to 1: if i % 2 == 0: count += i; return count
# (Sum of even numbers from 1 to n)
SUM_EVEN = IRProgram(
    name="sum_even",
    description="Sum of even numbers from 1 to N",
    opcodes=[
        IROpcode.START,
        # count = 0
        IROpcode.CONST_0,
        IROpcode.LOCAL_SET_0,
        # i = 1
        IROpcode.CONST_1,
        IROpcode.LOCAL_SET_1,
        # loop
        IROpcode.LOOP_BEGIN,
        # check if i is even: i % 2 == 0
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_2,
        IROpcode.I32_REM_S,
        IROpcode.CONST_0,
        IROpcode.I32_EQ,
        IROpcode.IF_BEGIN,
        # if even: count += i
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_GET_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_0,
        IROpcode.ELSE,
        # if odd: do nothing (need something for else branch)
        IROpcode.LOCAL_GET_0,
        IROpcode.LOCAL_SET_0,  # nop: count = count
        IROpcode.IF_END,
        # i++
        IROpcode.LOCAL_GET_1,
        IROpcode.CONST_1,
        IROpcode.I32_ADD,
        IROpcode.LOCAL_SET_1,
        # if i <= N: continue
        IROpcode.LOCAL_GET_1,
        IROpcode.SLOT_0,
        IROpcode.I32_LE_S,
        IROpcode.BR_IF,
        IROpcode.LOOP_END,
        # return count
        IROpcode.LOCAL_GET_0,
        IROpcode.END,
    ],
    num_operands=1,
    test_cases=[
        ([10], 30),   # 2 + 4 + 6 + 8 + 10 = 30
        ([7], 12),    # 2 + 4 + 6 = 12
        ([1], 0),     # no evens
    ],
)


# =============================================================================
# PROGRAM REGISTRY
# =============================================================================

ALL_PROGRAMS = {
    "sum_1_to_n": SUM_1_TO_N,
    "sum_a_to_b": SUM_A_TO_B,
    "factorial": FACTORIAL,
    "power": POWER,
    "fibonacci": FIBONACCI,
    "collatz_length": COLLATZ_LENGTH,
    "gcd": GCD,
    "is_prime": IS_PRIME,
    "max_of_two": MAX_OF_TWO,
    "abs_diff": ABS_DIFF,
    "sum_even": SUM_EVEN,
}

# Training set (model learns these)
TRAIN_PROGRAMS = ["sum_1_to_n", "sum_a_to_b", "factorial", "power"]

# Test set (model must generalize to these)
TEST_PROGRAMS = ["fibonacci", "collatz_length", "gcd", "is_prime"]


# =============================================================================
# COMPILER
# =============================================================================

def compile_program(program: IRProgram, operands: list[int]) -> bytes:
    """
    Compile IR program to WASM bytecode.

    This is a simplified version of IRCodebook.indices_to_wasm()
    that handles our specific opcode set.
    """
    wasm_bytes = bytearray()

    for opcode in program.opcodes:
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


def test_program(program: IRProgram) -> dict:
    """Test a program against its test cases."""
    runtime = WASMRuntime(use_native=True)  # Use wasmtime for loops
    results = []

    for operands, expected in program.test_cases:
        wasm_bytes = compile_program(program, operands)
        result = runtime.execute(wasm_bytes)

        success = result.success and result.result == expected
        results.append({
            "operands": operands,
            "expected": expected,
            "actual": result.result if result.success else None,
            "success": success,
            "error": result.error,
        })

    passed = sum(1 for r in results if r["success"])
    return {
        "program": program.name,
        "passed": passed,
        "total": len(results),
        "accuracy": passed / len(results) if results else 0,
        "details": results,
    }


def test_all_programs() -> dict:
    """Test all programs in the registry."""
    results = {}
    for name, program in ALL_PROGRAMS.items():
        results[name] = test_program(program)
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IR Program Library - Testing All Programs")
    print("=" * 60)

    results = test_all_programs()

    total_passed = 0
    total_tests = 0

    for name, result in results.items():
        status = "PASS" if result["accuracy"] == 1.0 else "FAIL"
        print(f"\n{name}: {result['passed']}/{result['total']} [{status}]")

        for detail in result["details"]:
            mark = "✓" if detail["success"] else "✗"
            print(f"  {mark} {detail['operands']} → {detail['actual']} (expected {detail['expected']})")
            if detail["error"]:
                print(f"    Error: {detail['error']}")

        total_passed += result["passed"]
        total_tests += result["total"]

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} = {total_passed/total_tests:.1%}")
    print("=" * 60)
