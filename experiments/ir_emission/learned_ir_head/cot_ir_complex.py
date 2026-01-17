"""
Complex Expression CoT Pipeline.

Handles:
- Chained operations: "12 times 20 and then minus 5" → ((12 * 20) - 5)
- Nested expressions: "12 * (5 + 2)" → 12 * (5 + 2)

The LLM normalizes to a stack-based format that WASM can execute directly.
"""

import sys
from pathlib import Path
import re

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from archive.codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from archive.wasm_runtime import WASMRuntime


def parse_to_ir_sequence(text: str) -> list[dict]:
    """
    Parse normalized expression to IR sequence.

    Handles:
    - Simple: "12 * 9 =" → [push 12, push 9, mul]
    - Chained: "12 * 20 - 5 =" → [push 12, push 20, mul, push 5, sub]
    - Nested: "12 * (5 + 2) =" → [push 12, push 5, push 2, add, mul]
    """
    text = text.strip().rstrip("=").strip()

    # Tokenize: numbers, operators, parens
    tokens = re.findall(r'\d+|[+\-*/()]', text)

    # Convert to postfix (Reverse Polish Notation) using shunting yard
    output = []
    op_stack = []

    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    op_map = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

    for token in tokens:
        if token.isdigit():
            output.append({'type': 'push', 'value': int(token)})
        elif token in precedence:
            while (op_stack and
                   op_stack[-1] != '(' and
                   op_stack[-1] in precedence and
                   precedence[op_stack[-1]] >= precedence[token]):
                output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
            op_stack.append(token)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
            if op_stack:
                op_stack.pop()  # Remove '('

    while op_stack:
        if op_stack[-1] != '(':
            output.append({'type': 'op', 'op': op_map[op_stack.pop()]})
        else:
            op_stack.pop()

    return output


def ir_to_wasm(ir_sequence: list[dict]) -> bytes:
    """Convert IR sequence to WASM bytecode."""
    OP_TO_WASM = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    body = bytearray()
    for instr in ir_sequence:
        if instr['type'] == 'push':
            body.extend(encode_i32_const(instr['value']))
        elif instr['type'] == 'op':
            body.extend(OPCODE_TO_WASM[OP_TO_WASM[instr['op']]])

    return bytes(body)


def normalize_complex(model, tokenizer, text: str, max_tokens: int = 30) -> str:
    """
    Normalize complex NL to math expression.

    "12 times 20 and then minus 5" → "12 * 20 - 5 ="
    "twelve times the sum of 5 and 2" → "12 * (5 + 2) ="
    """
    prompt = f"""Convert to math expression with = at end:
"five plus three" → 5 + 3 =
"twelve times nine" → 12 * 9 =
"10 times 5 minus 3" → 10 * 5 - 3 =
"20 plus 10 then divided by 5" → (20 + 10) / 5 =
"3 times the sum of 4 and 5" → 3 * (4 + 5) =
"12 times 20 and then minus 5" → 12 * 20 - 5 =
"100 minus 50 plus 25" → 100 - 50 + 25 =
"{text}" →"""

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if "=" in decoded:
            break

    result = tokenizer.decode(generated).strip()
    # Clean up - take only up to first =
    if "=" in result:
        result = result[:result.index("=") + 1]
    else:
        result = result + " ="

    return result


class ComplexCoTPipeline:
    """Pipeline for complex expressions."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = WASMRuntime()

    def execute(self, text: str) -> dict:
        # Check if already math-like
        if any(c in text for c in ['+', '-', '*', '/', '(', ')']):
            has_numbers = bool(re.search(r'\d', text))
            if has_numbers:
                canonical = text.strip()
                if not canonical.endswith("="):
                    canonical += " ="
            else:
                canonical = normalize_complex(self.model, self.tokenizer, text)
        else:
            canonical = normalize_complex(self.model, self.tokenizer, text)

        # Parse to IR sequence
        ir_seq = parse_to_ir_sequence(canonical)

        if not ir_seq:
            return {
                "input": text,
                "canonical": canonical,
                "ir": None,
                "result": None,
                "success": False,
                "error": "Parse failed",
            }

        # Build and execute WASM
        wasm_bytes = ir_to_wasm(ir_seq)
        result = self.runtime.execute(wasm_bytes)

        return {
            "input": text,
            "canonical": canonical,
            "ir": ir_seq,
            "result": result.result if result.success else None,
            "success": result.success,
            "error": result.error,
        }


def format_ir(ir_seq: list[dict]) -> str:
    """Format IR sequence for display."""
    parts = []
    for instr in ir_seq:
        if instr['type'] == 'push':
            parts.append(f"push({instr['value']})")
        else:
            parts.append(instr['op'])
    return " → ".join(parts)


def main():
    print("=" * 80)
    print("  COMPLEX EXPRESSION COT PIPELINE")
    print("  NL → Normalize → Parse (shunting yard) → WASM stack → Result")
    print("=" * 80)

    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    pipeline = ComplexCoTPipeline(model, tokenizer)

    # Test cases - simple to complex
    test_cases = [
        # Simple (baseline)
        ("12 times 9", 108),
        ("25 plus 17", 42),

        # Chained operations
        ("12 * 20 - 5", 235),  # (12 * 20) - 5 = 240 - 5 = 235
        ("10 + 5 * 2", 20),    # 10 + (5 * 2) = 10 + 10 = 20 (precedence!)
        ("100 - 50 + 25", 75), # left-to-right: 50 + 25 = 75

        # NL chained
        ("12 times 20 and then minus 5", 235),
        ("10 plus 5 times 2", 20),

        # Nested with parens
        ("(5 + 2) * 3", 21),   # 7 * 3 = 21
        ("12 * (5 + 2)", 84),  # 12 * 7 = 84
        ("(10 + 20) / 5", 6),  # 30 / 5 = 6

        # NL nested
        ("3 times the sum of 4 and 5", 27),  # 3 * (4 + 5) = 27
    ]

    print(f"\n{'Input':<40} {'Canonical':<20} {'Result':<8} {'Expected':<8} {'Status'}")
    print("-" * 100)

    correct = 0
    for text, expected in test_cases:
        result = pipeline.execute(text)

        if result["success"] and result["result"] == expected:
            status = "✓"
            correct += 1
        elif result["success"]:
            status = f"✗ ({result['result']})"
        else:
            status = f"✗ {result['error']}"

        print(f"{text:<40} {result['canonical']:<20} {result['result'] or 'ERR':<8} {expected:<8} {status}")

        # Show IR for interesting cases
        if result["ir"] and len(result["ir"]) > 3:
            print(f"    IR: {format_ir(result['ir'])}")

    print("-" * 100)
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")

    # Architecture
    print("\n" + "=" * 80)
    print("  HOW IT WORKS")
    print("=" * 80)
    print("""
    Example: "12 times 20 and then minus 5"

    Stage 1: CoT Normalization
    ─────────────────────────────────────────
    "12 times 20 and then minus 5" → "12 * 20 - 5 ="

    Stage 2: Shunting Yard → Postfix IR
    ─────────────────────────────────────────
    Tokens: [12, *, 20, -, 5]
    Postfix: [push(12), push(20), mul, push(5), sub]

    Stage 3: WASM Stack Execution
    ─────────────────────────────────────────
    push(12)  → stack: [12]
    push(20)  → stack: [12, 20]
    mul       → stack: [240]      (12 * 20)
    push(5)   → stack: [240, 5]
    sub       → stack: [235]      (240 - 5)

    Result: 235

    ═══════════════════════════════════════════════════════════════════

    Example: "12 * (5 + 2)"

    Stage 1: Already canonical
    ─────────────────────────────────────────
    "12 * (5 + 2) ="

    Stage 2: Shunting Yard (handles parens!)
    ─────────────────────────────────────────
    Tokens: [12, *, (, 5, +, 2, )]
    Postfix: [push(12), push(5), push(2), add, mul]

    Stage 3: WASM Stack Execution
    ─────────────────────────────────────────
    push(12)  → stack: [12]
    push(5)   → stack: [12, 5]
    push(2)   → stack: [12, 5, 2]
    add       → stack: [12, 7]     (5 + 2)
    mul       → stack: [84]        (12 * 7)

    Result: 84
    """)


if __name__ == "__main__":
    main()
