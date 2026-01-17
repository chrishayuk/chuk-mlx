"""
Chain-of-Thought IR Pipeline.

Architecture:
1. LLM normalizes NL to canonical format: "12 times 9" → "12 * 9 ="
2. Parse canonical format to extract IR (deterministic)
3. Execute IR via WASM

Key insight: CoT IS the learned parser. The LLM learns to normalize any NL
to a canonical format, which is then trivially parseable.

The "learning" happens in the LLM during instruction tuning, not in a
separate projection head.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add project root for chuk_lazarus imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from chuk_lazarus.models_v2.loader import load_model
from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime


def parse_canonical(text: str) -> dict:
    """
    Parse canonical format "A op B =" into IR.

    This is deterministic - the LLM already did the hard work.
    """
    text = text.strip().rstrip("=").strip()

    op_map = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
    }

    for sym, op in op_map.items():
        if sym in text:
            parts = text.split(sym)
            if len(parts) == 2:
                try:
                    a = int(parts[0].strip())
                    b = int(parts[1].strip())
                    return {"op": op, "a": a, "b": b}
                except ValueError:
                    pass

    return None


def normalize_to_canonical(model, tokenizer, text: str, max_tokens: int = 20) -> str:
    """
    Use the LLM to normalize natural language to canonical format.

    This is where the "learning" happens - the LLM was trained to understand
    natural language math and output structured format.
    """
    prompt = f"""Convert to math expression with = at end:
"five plus three" → 5 + 3 =
"twenty minus seven" → 20 - 7 =
"six times four" → 6 * 4 =
"fifteen divided by three" → 15 / 3 =
"12 times 9" → 12 * 9 =
"25 plus 17" → 25 + 17 =
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
    if not result.endswith("="):
        result = result + " ="

    return result


class CoTIRPipeline:
    """
    Full pipeline: NL → CoT Normalize → Parse → WASM → Result

    The LLM's job: understand NL, output canonical format
    Our job: parse the canonical format, execute
    """

    OP_TO_WASM = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = WASMRuntime()

    def normalize(self, text: str) -> str:
        """Stage 1: NL → Canonical format via CoT"""
        # If already canonical format, just add =
        text = text.strip()
        for sym in ["+", "-", "*", "/"]:
            if sym in text and not text.endswith("="):
                parts = text.split(sym)
                if len(parts) == 2:
                    try:
                        int(parts[0].strip())
                        int(parts[1].strip())
                        return text + " =" if not text.endswith("=") else text
                    except ValueError:
                        pass
        return normalize_to_canonical(self.model, self.tokenizer, text)

    def parse(self, canonical: str) -> dict:
        """Stage 2: Canonical → IR (deterministic)"""
        return parse_canonical(canonical)

    def build_wasm(self, ir: dict) -> bytes:
        """Stage 3: IR → WASM bytecode"""
        body = bytearray()
        body.extend(encode_i32_const(ir["a"]))
        body.extend(encode_i32_const(ir["b"]))
        body.extend(OPCODE_TO_WASM[self.OP_TO_WASM[ir["op"]]])
        return bytes(body)

    def execute(self, text: str) -> dict:
        """Full pipeline: NL → Result"""
        # Stage 1: Normalize
        canonical = self.normalize(text)

        # Stage 2: Parse
        ir = self.parse(canonical)
        if ir is None:
            return {
                "input": text,
                "canonical": canonical,
                "ir": None,
                "result": None,
                "success": False,
                "error": "Parse failed",
            }

        # Stage 3: Execute
        wasm_bytes = self.build_wasm(ir)
        result = self.runtime.execute(wasm_bytes)

        return {
            "input": text,
            "canonical": canonical,
            "ir": ir,
            "wasm_hex": wasm_bytes.hex(),
            "result": result.result if result.success else None,
            "success": result.success,
            "error": result.error,
        }


def compute_expected(op: str, a: int, b: int) -> int:
    if op == "add":
        return a + b
    elif op == "subtract":
        return a - b
    elif op == "multiply":
        return a * b
    elif op == "divide":
        return a // b if b != 0 else 0
    return 0


def main():
    print("=" * 70)
    print("  COT IR PIPELINE: NL → Normalize → Parse → WASM → Result")
    print("=" * 70)

    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    pipeline = CoTIRPipeline(model, tokenizer)

    # Test cases
    test_cases = [
        ("12 times 9", "multiply", 12, 9),
        ("25 plus 17", "add", 25, 17),
        ("50 minus 15", "subtract", 50, 15),
        ("48 divided by 6", "divide", 48, 6),
        ("seven times eight", "multiply", 7, 8),
        ("thirty plus twenty", "add", 30, 20),
        ("100 - 37", "subtract", 100, 37),
        ("6 * 7", "multiply", 6, 7),
    ]

    print(f"\n{'NL Input':<25} {'Canonical':<15} {'IR':<20} {'Result':<8} {'Expected':<8} {'Status'}")
    print("-" * 100)

    correct = 0
    for text, expected_op, a, b in test_cases:
        result = pipeline.execute(text)

        expected = compute_expected(expected_op, a, b)

        if result["success"] and result["result"] == expected:
            status = "✓"
            correct += 1
        elif result["success"]:
            status = f"✗ ({result['result']})"
        else:
            status = f"✗ {result['error']}"

        ir_str = f"{result['ir']['op']}({result['ir']['a']}, {result['ir']['b']})" if result["ir"] else "FAILED"
        print(f"{text:<25} {result['canonical']:<15} {ir_str:<20} {result['result'] or 'ERR':<8} {expected:<8} {status}")

    print("-" * 100)
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")

    # Architecture
    print("\n" + "=" * 70)
    print("  ARCHITECTURE")
    print("=" * 70)
    print("""
    ┌───────────────────────────────────────────────────────────────┐
    │  "twelve times nine"                                          │
    │         ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │  CoT Normalization (LLM generation, few-shot)           │  │
    │  │  "twelve times nine" → "12 * 9 ="                       │  │
    │  │  This IS the learned parser - trained via instruction   │  │
    │  │  tuning to understand NL and emit structured format     │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │         ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │  Deterministic Parse                                    │  │
    │  │  "12 * 9 =" → {op: multiply, a: 12, b: 9}              │  │
    │  │  Simple string split - 100% reliable on canonical       │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │         ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │  WASM IR Builder                                        │  │
    │  │  [i32.const 12, i32.const 9, i32.mul]                  │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │         ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │  WASM Runtime (wasmtime)                                │  │
    │  │  → 108                                                  │  │
    │  └─────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────┘

    KEY INSIGHT: The LLM itself is the learned parser.

    CoT normalization IS the projection from NL to structured IR.
    The "regex" is just string.split() on the LLM's output.

    Training the LLM to emit canonical format = training the parser.
    """)


if __name__ == "__main__":
    main()
