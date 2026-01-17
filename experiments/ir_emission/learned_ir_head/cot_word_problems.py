"""
Word Problem CoT Pipeline.

"Jenny has 5 apples. She gives 2 to Bob. How many does she have?"
→ "5 - 2 =" → 3

The LLM must:
1. Extract relevant quantities
2. Understand the operation implied by verbs (gives → subtract)
3. Emit canonical math expression
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
    """Parse normalized expression to IR sequence."""
    text = text.strip().rstrip("=").strip()
    tokens = re.findall(r'\d+|[+\-*/()]', text)

    output = []
    op_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    op_map = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

    for token in tokens:
        if token.isdigit():
            output.append({'type': 'push', 'value': int(token)})
        elif token in precedence:
            while (op_stack and op_stack[-1] != '(' and
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
                op_stack.pop()

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


def normalize_word_problem(model, tokenizer, text: str, max_tokens: int = 30) -> str:
    """
    Normalize word problem to math expression.

    This is the key challenge - understanding that:
    - "gives away" → subtraction
    - "receives" / "gets" → addition
    - "each" / "per" → multiplication
    - "split" / "shared equally" → division
    """
    prompt = f"""Convert word problem to math expression with = at end:

"Jenny has 5 apples. She gives 2 to Bob. How many left?" → 5 - 2 =
"Tom has 3 cookies. He gets 4 more. How many total?" → 3 + 4 =
"There are 6 bags with 4 oranges each. How many oranges?" → 6 * 4 =
"12 candies split between 3 kids. How many each?" → 12 / 3 =
"Sam had 10 marbles, lost 3, then found 5. How many now?" → 10 - 3 + 5 =
"A box has 8 toys. 2 boxes. How many toys?" → 8 * 2 =
"Lisa has 20 stickers. She gives 5 to Ann and 3 to Bob. How many left?" → 20 - 5 - 3 =
"10 cookies. 2 kids each eat 3. How many left?" → 10 - 2 * 3 =
"A jar has 20 marbles. 4 people each take 3. How many remain?" → 20 - 4 * 3 =
"There are 15 apples. 3 friends each take 2. How many are left?" → 15 - 3 * 2 =

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
    if "=" in result:
        result = result[:result.index("=") + 1]
    else:
        result = result + " ="

    return result


class WordProblemPipeline:
    """Pipeline for word problems."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = WASMRuntime()

    def execute(self, text: str) -> dict:
        canonical = normalize_word_problem(self.model, self.tokenizer, text)

        ir_seq = parse_to_ir_sequence(canonical)

        if not ir_seq:
            return {
                "input": text,
                "canonical": canonical,
                "ir": ir_seq,
                "result": None,
                "success": False,
                "error": "Parse failed",
            }

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


def main():
    print("=" * 90)
    print("  WORD PROBLEM COT PIPELINE")
    print("  'Jenny has 5 apples...' → '5 - 2 =' → WASM → 3")
    print("=" * 90)

    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    pipeline = WordProblemPipeline(model, tokenizer)

    # Word problems - testing semantic understanding
    test_cases = [
        # Basic subtraction (giving away)
        ("Jenny has 5 apples. She gives 2 to Bob. How many does she have?", 3),

        # Basic addition (receiving)
        ("Tom has 3 cookies. He gets 4 more. How many does he have?", 7),

        # Multiplication (groups)
        ("There are 6 bags with 4 oranges in each bag. How many oranges total?", 24),

        # Division (sharing)
        ("12 candies are split equally between 3 kids. How many does each kid get?", 4),

        # Chained operations
        ("Sam had 10 marbles. He lost 3 but then found 5. How many does he have now?", 12),

        # Multiple subtractions
        ("Lisa has 20 stickers. She gives 5 to Ann and 3 to Bob. How many does she have left?", 12),

        # Multiplication context
        ("A classroom has 5 rows of desks with 6 desks in each row. How many desks?", 30),

        # Real world
        ("A pizza has 8 slices. 3 people each eat 2 slices. How many slices are left?", 2),
    ]

    print(f"\n{'Word Problem':<70} {'Math':<15} {'Result':<6} {'Exp':<6}")
    print("-" * 100)

    correct = 0
    for text, expected in test_cases:
        result = pipeline.execute(text)

        if result["success"] and result["result"] == expected:
            status = "✓"
            correct += 1
        elif result["success"]:
            status = f"✗ got {result['result']}"
        else:
            status = f"✗ {result['error']}"

        # Truncate long problems for display
        display_text = text[:67] + "..." if len(text) > 70 else text
        print(f"{display_text:<70} {result['canonical']:<15} {result['result'] or 'ERR':<6} {expected:<6} {status}")

    print("-" * 100)
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")

    # Show the semantic mapping
    print("\n" + "=" * 90)
    print("  SEMANTIC UNDERSTANDING REQUIRED")
    print("=" * 90)
    print("""
    The LLM must learn these semantic mappings:

    VERBS → OPERATIONS
    ─────────────────────────────────────────────────────────────────
    "gives", "loses", "eats", "spends"        →  SUBTRACT
    "gets", "finds", "receives", "earns"      →  ADD
    "each", "per", "rows of", "groups of"     →  MULTIPLY
    "split", "shared equally", "divided"      →  DIVIDE

    ENTITIES → OPERANDS
    ─────────────────────────────────────────────────────────────────
    "Jenny has 5 apples"                      →  Start with 5
    "gives 2 to Bob"                          →  Subtract 2
    "3 kids"                                  →  Divide by 3

    This is the HARD part that the LLM learns through training.
    The math execution is trivial once you have the expression.
    """)


if __name__ == "__main__":
    main()
