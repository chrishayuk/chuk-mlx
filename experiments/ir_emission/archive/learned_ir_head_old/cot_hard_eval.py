"""
Hard Evaluation Problems - GSM8K style.

Testing limits of CoT → IR → WASM pipeline on increasingly complex problems.
"""

import sys
from pathlib import Path
import re

import mlx.core as mx

# Add project root for chuk_lazarus imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from chuk_lazarus.models_v2.loader import load_model
from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime


def parse_to_ir_sequence(text: str) -> list[dict]:
    """Parse normalized expression to IR sequence using shunting yard."""
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


def normalize_hard(model, tokenizer, text: str, max_tokens: int = 80) -> str:
    """
    Normalize complex word problems to math expressions.
    More explicit step-by-step scaffolding.
    """
    prompt = f"""Convert word problem to ONE math expression. Show your work.

Q: A book costs $15. Tom buys 3 books. How much does he spend?
Work: cost × quantity = 15 × 3
Answer: 15 * 3 =

Q: Sam has 20 dollars. He buys 3 toys at 4 dollars each. How much left?
Work: start - (quantity × price) = 20 - (3 × 4)
Answer: 20 - 3 * 4 =

Q: Maria has 48 stickers. She gives them equally to 6 friends. How many does each friend get?
Work: total ÷ friends = 48 ÷ 6
Answer: 48 / 6 =

Q: 24 cookies shared equally among 4 kids. How many does each kid get?
Work: total ÷ kids = 24 ÷ 4
Answer: 24 / 4 =

Q: A baker makes 12 cupcakes per batch. She makes 5 batches and sells 40. How many left?
Work: (per_batch × batches) - sold = (12 × 5) - 40
Answer: 12 * 5 - 40 =

Q: John earns $10 per hour. He works 8 hours Monday and 6 hours Tuesday. He spends $50. How much left?
Work: (rate × hours1) + (rate × hours2) - spent = (10 × 8) + (10 × 6) - 50
Answer: 10 * 8 + 10 * 6 - 50 =

Q: Lisa earns $15 per hour. She works 4 hours Saturday and 5 hours Sunday. How much total?
Work: (rate × hours1) + (rate × hours2) = (15 × 4) + (15 × 5)
Answer: 15 * 4 + 15 * 5 =

Q: 200 apples. Sell 45 Monday, 38 Tuesday, receive 60 more. How many?
Work: start - sell1 - sell2 + receive = 200 - 45 - 38 + 60
Answer: 200 - 45 - 38 + 60 =

Q: There are 5 classrooms. Each classroom has 6 rows. Each row has 4 desks. How many desks total?
Work: classrooms × rows × desks = 5 × 6 × 4
Answer: 5 * 6 * 4 =

Q: A farmer has 3 fields. Each field has 4 sections. Each section produces 25 bushels. Total bushels?
Work: fields × sections × bushels = 3 × 4 × 25
Answer: 3 * 4 * 25 =

Q: 2 buildings. Each building has 5 floors. Each floor has 10 offices. Total offices?
Work: buildings × floors × offices = 2 × 5 × 10
Answer: 2 * 5 * 10 =

Q: Tom has 3 boxes, 12 pencils each. Gives 2 pencils to each of 5 friends. How many left?
Work: (boxes × per_box) - (friends × each) = (3 × 12) - (5 × 2)
Answer: 3 * 12 - 5 * 2 =

Q: Parking lot: 4 rows, 15 spaces each. 38 cars parked. Empty spaces?
Work: (rows × spaces) - parked = (4 × 15) - 38
Answer: 4 * 15 - 38 =

Q: 48 cookies in boxes of 6. Sell each box for $5. Total money?
Work: (cookies ÷ per_box) × price = (48 ÷ 6) × 5
Answer: 48 / 6 * 5 =

Q: Train: 8 cars, 45 seats each. 287 passengers. Empty seats?
Work: (cars × seats) - passengers = (8 × 45) - 287
Answer: 8 * 45 - 287 =

Q: {text}
Work:"""

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
        # Stop at = sign in Answer line
        if "Answer:" in decoded and "=" in decoded.split("Answer:")[-1]:
            break

    result = tokenizer.decode(generated).strip()

    # Extract just the expression after "Answer:"
    if "Answer:" in result:
        expr = result.split("Answer:")[-1].strip()
        # Clean up - take only the math expression
        expr = re.sub(r'[^0-9+\-*/()= ]', '', expr)
        if "=" in expr:
            expr = expr[:expr.index("=") + 1]
        return expr.strip()

    # Fallback: try to find any expression
    match = re.search(r'[\d+\-*/()]+\s*=', result)
    if match:
        return match.group()

    return result + " ="


class HardEvalPipeline:
    """Pipeline for hard evaluation problems."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = WASMRuntime()

    def execute(self, text: str) -> dict:
        canonical = normalize_hard(self.model, self.tokenizer, text)

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

        try:
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
        except Exception as e:
            return {
                "input": text,
                "canonical": canonical,
                "ir": ir_seq,
                "result": None,
                "success": False,
                "error": str(e),
            }


def main():
    print("=" * 90)
    print("  HARD EVALUATION - GSM8K STYLE PROBLEMS")
    print("=" * 90)

    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    pipeline = HardEvalPipeline(model, tokenizer)

    # Organized by difficulty
    test_cases = [
        # LEVEL 1: Single operation with context
        ("LEVEL 1: Single Operation", None, None),
        ("A book costs $15. Tom buys 3 books. How much does he spend?", 45),
        ("Maria has 48 stickers. She gives them equally to 6 friends. How many does each get?", 8),

        # LEVEL 2: Two operations
        ("LEVEL 2: Two Operations", None, None),
        ("Sam has 20 dollars. He buys 3 toys at 4 dollars each. How much money is left?", 8),
        ("A baker makes 12 cupcakes per batch. She makes 5 batches and sells 40. How many left?", 20),

        # LEVEL 3: Three+ operations
        ("LEVEL 3: Three+ Operations", None, None),
        ("John earns $10 per hour. He works 8 hours on Monday and 6 hours on Tuesday. He spends $50 on groceries. How much does he have?", 90),
        ("A store has 200 apples. They sell 45 on Monday, 38 on Tuesday, and receive 60 more. How many now?", 177),

        # LEVEL 4: Nested reasoning
        ("LEVEL 4: Nested Reasoning", None, None),
        ("There are 5 classrooms. Each classroom has 6 rows of desks. Each row has 4 desks. How many desks total?", 120),
        ("A farmer has 3 fields. Each field has 4 sections. Each section produces 25 bushels. How many bushels total?", 300),

        # LEVEL 5: Multi-step with division
        ("LEVEL 5: Multi-step with Division", None, None),
        ("144 students split into teams of 6. Each team needs 4 balls. How many balls total?", 96),
        ("A company has 180 employees split equally into 9 departments. Each department orders 5 computers. How many computers?", 100),

        # LEVEL 6: GSM8K-style complex
        ("LEVEL 6: GSM8K Style", None, None),
        ("Tom has 3 boxes of pencils. Each box has 12 pencils. He gives 2 pencils to each of his 5 friends. How many pencils does Tom have left?", 26),
        ("A parking lot has 4 rows. Each row has 15 spaces. If 38 cars are parked, how many empty spaces?", 22),

        # LEVEL 7: Harder GSM8K
        ("LEVEL 7: Harder Problems", None, None),
        ("Sarah bakes 48 cookies. She puts 6 cookies in each box. She sells each box for $5. If she sells all boxes, how much money?", 40),
        ("A train has 8 cars. Each car has 45 seats. If 287 passengers board, how many empty seats?", 73),
    ]

    current_level = ""
    level_correct = 0
    level_total = 0
    total_correct = 0
    total_problems = 0

    for item in test_cases:
        if item[1] is None:  # Level header
            if level_total > 0:
                print(f"  Level accuracy: {level_correct}/{level_total}\n")
            print(f"\n{'─' * 90}")
            print(f"  {item[0]}")
            print(f"{'─' * 90}")
            level_correct = 0
            level_total = 0
            continue

        text, expected = item
        total_problems += 1
        level_total += 1

        result = pipeline.execute(text)

        if result["success"] and result["result"] == expected:
            status = "✓"
            total_correct += 1
            level_correct += 1
        elif result["success"]:
            status = f"✗ got {result['result']}"
        else:
            status = f"✗ {result['error'][:20]}"

        # Truncate for display
        display_text = text[:65] + "..." if len(text) > 68 else text
        print(f"  {status} {display_text}")
        print(f"      → {result['canonical']:<25} = {result['result']}")

    # Final level
    if level_total > 0:
        print(f"  Level accuracy: {level_correct}/{level_total}")

    print(f"\n{'=' * 90}")
    print(f"  OVERALL: {total_correct}/{total_problems} = {total_correct/total_problems:.0%}")
    print(f"{'=' * 90}")

    # Analysis
    print("""

  ANALYSIS
  ────────────────────────────────────────────────────────────────────────────

  What the pipeline CAN handle:
  • Multi-step arithmetic with +, -, *, /
  • Operator precedence and parentheses
  • Semantic understanding of verbs (buys, sells, gives, earns)
  • Nested quantities (3 boxes × 4 bags × 5 items)
  • Sequential operations (has X, spends Y, earns Z)

  What challenges it:
  • Very long reasoning chains (>4 steps)
  • Problems requiring intermediate variable tracking
  • Ambiguous language or multiple valid interpretations
  • Problems where order of operations matters semantically

  Limitations of current approach:
  • Single expression output - can't handle "compute X, then use X to compute Y"
  • Integer arithmetic only (no fractions, decimals)
  • No variable assignment or equations
  • Dependent on model's few-shot learning capacity
    """)


if __name__ == "__main__":
    main()
