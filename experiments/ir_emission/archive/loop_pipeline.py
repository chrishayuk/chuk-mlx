#!/usr/bin/env python3
"""
Loop IR Pipeline.

This is where WASM really pays off - the model can't loop, but WASM can.

Examples:
  "Sum 1 to 10"           → loop with accumulator
  "Multiply 1 to 5"       → loop with product (factorial)
  "Count down from 10"    → loop with decrement
"""

import json
import logging
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from codebook import encode_i32_const
from wasm_runtime import WASMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def build_sum_loop_wasm(start: int, end: int) -> bytes:
    """
    Build WASM for: sum from start to end.

    Algorithm:
      local.0 = accumulator (starts at 0)
      local.1 = counter (starts at start)

      loop:
        acc += counter
        counter += 1
        if counter <= end: branch back
      return acc

    WASM bytecode:
      i32.const 0         ; acc = 0
      local.set 0

      i32.const {start}   ; counter = start
      local.set 1

      loop                ; loop label 0
        local.get 0       ; push acc
        local.get 1       ; push counter
        i32.add           ; acc + counter
        local.set 0       ; acc = result

        local.get 1       ; push counter
        i32.const 1
        i32.add           ; counter + 1
        local.tee 1       ; counter = result, keep on stack

        i32.const {end}
        i32.le_s          ; counter <= end?
        br_if 0           ; if true, loop back
      end

      local.get 0         ; return acc
    """
    body = bytearray()

    # Initialize acc = 0
    body.extend(encode_i32_const(0))
    body.append(0x21)  # local.set
    body.append(0x00)  # local 0

    # Initialize counter = start
    body.extend(encode_i32_const(start))
    body.append(0x21)  # local.set
    body.append(0x01)  # local 1

    # loop
    body.append(0x03)  # loop
    body.append(0x40)  # void result

    # acc += counter
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0 (acc)
    body.append(0x20)  # local.get
    body.append(0x01)  # local 1 (counter)
    body.append(0x6a)  # i32.add
    body.append(0x21)  # local.set
    body.append(0x00)  # local 0

    # counter += 1
    body.append(0x20)  # local.get
    body.append(0x01)  # local 1
    body.extend(encode_i32_const(1))
    body.append(0x6a)  # i32.add
    body.append(0x22)  # local.tee
    body.append(0x01)  # local 1

    # counter <= end?
    body.extend(encode_i32_const(end))
    body.append(0x4c)  # i32.le_s

    # br_if 0
    body.append(0x0d)  # br_if
    body.append(0x00)  # label 0

    # end loop
    body.append(0x0b)

    # return acc
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0

    return bytes(body)


def build_product_loop_wasm(start: int, end: int) -> bytes:
    """
    Build WASM for: product from start to end (like factorial).

    Similar to sum but with multiply instead of add.
    """
    body = bytearray()

    # Initialize acc = 1 (multiplicative identity)
    body.extend(encode_i32_const(1))
    body.append(0x21)  # local.set
    body.append(0x00)  # local 0

    # Initialize counter = start
    body.extend(encode_i32_const(start))
    body.append(0x21)  # local.set
    body.append(0x01)  # local 1

    # loop
    body.append(0x03)  # loop
    body.append(0x40)  # void result

    # acc *= counter
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0 (acc)
    body.append(0x20)  # local.get
    body.append(0x01)  # local 1 (counter)
    body.append(0x6c)  # i32.mul
    body.append(0x21)  # local.set
    body.append(0x00)  # local 0

    # counter += 1
    body.append(0x20)  # local.get
    body.append(0x01)  # local 1
    body.extend(encode_i32_const(1))
    body.append(0x6a)  # i32.add
    body.append(0x22)  # local.tee
    body.append(0x01)  # local 1

    # counter <= end?
    body.extend(encode_i32_const(end))
    body.append(0x4c)  # i32.le_s

    # br_if 0
    body.append(0x0d)  # br_if
    body.append(0x00)  # label 0

    # end loop
    body.append(0x0b)

    # return acc
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0

    return bytes(body)


def build_countdown_wasm(start: int) -> bytes:
    """
    Build WASM for countdown - returns final value (0).
    This is mainly to test loop generation.
    """
    body = bytearray()

    # Initialize counter = start
    body.extend(encode_i32_const(start))
    body.append(0x21)  # local.set
    body.append(0x00)  # local 0

    # loop
    body.append(0x03)  # loop
    body.append(0x40)  # void result

    # counter -= 1
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0
    body.extend(encode_i32_const(1))
    body.append(0x6b)  # i32.sub
    body.append(0x22)  # local.tee
    body.append(0x00)  # local 0

    # counter > 0?
    body.extend(encode_i32_const(0))
    body.append(0x4a)  # i32.gt_s

    # br_if 0
    body.append(0x0d)  # br_if
    body.append(0x00)  # label 0

    # end loop
    body.append(0x0b)

    # return counter (should be 0)
    body.append(0x20)  # local.get
    body.append(0x00)  # local 0

    return bytes(body)


class LoopCompiler:
    """
    Compiler for loop constructs.

    Uses few-shot prompting to parse loop intent,
    then generates appropriate WASM loop IR.
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        from chuk_lazarus.models_v2.loader import load_model

        logger.info("Loading model...")
        result = load_model(model_name)
        self.model = result.model
        self.tokenizer = result.tokenizer
        self.model.freeze()

        self.runtime = WASMRuntime()

    def parse_loop(self, nl_input: str) -> dict:
        """Parse loop intent from NL."""
        prompt = f"""<|system|>
Parse the loop instruction. Output: type (sum/product/count), start, end.
</s>
<|user|>
Sum 1 to 10
</s>
<|assistant|>
type: sum
start: 1
end: 10</s>
<|user|>
Multiply 1 to 5
</s>
<|assistant|>
type: product
start: 1
end: 5</s>
<|user|>
Add numbers from 5 to 20
</s>
<|assistant|>
type: sum
start: 5
end: 20</s>
<|user|>
Count down from 10
</s>
<|assistant|>
type: count
start: 10
end: 0</s>
<|user|>
{nl_input}
</s>
<|assistant|>
"""
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids
        for _ in range(40):
            output = self.model(generated_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            decoded = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
            # Stop only when we have </s> or have a complete "end: <number>\n" pattern
            if "</s>" in decoded:
                break
            # Check for complete end pattern: "end:" followed by a number AND newline
            # The newline ensures we don't stop at "end: 1" when it should be "end: 10"
            if re.search(r"end:\s*-?\d+\s*\n", decoded.lower()):
                break

        response = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
        response = response.replace("</s>", "").strip()

        # Parse response
        result = {"type": None, "start": None, "end": None}

        for line in response.split("\n"):
            line = line.strip().lower()
            if line.startswith("type:"):
                result["type"] = line.split(":")[1].strip().split()[0]  # Take first word
            elif line.startswith("start:"):
                try:
                    # Extract number, handle "1</s>" case
                    val = line.split(":")[1].strip()
                    val = ''.join(c for c in val if c.isdigit() or c == '-')
                    result["start"] = int(val)
                except:
                    pass
            elif line.startswith("end:"):
                try:
                    val = line.split(":")[1].strip()
                    val = ''.join(c for c in val if c.isdigit() or c == '-')
                    result["end"] = int(val)
                except:
                    pass

        return result

    def compile_and_run(self, nl_input: str) -> dict:
        """Compile and execute loop."""
        parsed = self.parse_loop(nl_input)

        if not parsed["type"] or parsed["start"] is None or parsed["end"] is None:
            return {
                "input": nl_input,
                "success": False,
                "error": "Failed to parse loop",
                "parsed": parsed,
            }

        try:
            if parsed["type"] == "sum":
                ir_bytes = build_sum_loop_wasm(parsed["start"], parsed["end"])
            elif parsed["type"] == "product":
                ir_bytes = build_product_loop_wasm(parsed["start"], parsed["end"])
            elif parsed["type"] == "count":
                ir_bytes = build_countdown_wasm(parsed["start"])
            else:
                return {
                    "input": nl_input,
                    "success": False,
                    "error": f"Unknown loop type: {parsed['type']}",
                    "parsed": parsed,
                }

            # Execute with 2 locals
            result = self.runtime.execute(ir_bytes, num_locals=2)

            if result.success:
                return {
                    "input": nl_input,
                    "parsed": parsed,
                    "ir_hex": ir_bytes.hex(),
                    "result": result.result,
                    "success": True,
                }
            else:
                return {
                    "input": nl_input,
                    "parsed": parsed,
                    "success": False,
                    "error": result.error,
                }
        except Exception as e:
            return {
                "input": nl_input,
                "parsed": parsed,
                "success": False,
                "error": str(e),
            }


def compute_expected_sum(start: int, end: int) -> int:
    """Compute sum from start to end."""
    return sum(range(start, end + 1))


def compute_expected_product(start: int, end: int) -> int:
    """Compute product from start to end."""
    result = 1
    for i in range(start, end + 1):
        result *= i
    return result


def main():
    compiler = LoopCompiler()

    test_cases = [
        # Sum loops
        ("Sum 1 to 10", compute_expected_sum(1, 10)),        # 55
        ("Sum 1 to 100", compute_expected_sum(1, 100)),      # 5050
        ("Add numbers from 5 to 15", compute_expected_sum(5, 15)),  # 110
        ("Sum from 1 to 5", compute_expected_sum(1, 5)),     # 15

        # Product loops (factorial-like)
        ("Multiply 1 to 5", compute_expected_product(1, 5)),  # 120 (5!)
        ("Product of 1 to 6", compute_expected_product(1, 6)),  # 720 (6!)
        ("Multiply numbers from 2 to 4", compute_expected_product(2, 4)),  # 24

        # Countdown
        ("Count down from 10", 0),
        ("Count from 5 to 0", 0),
    ]

    logger.info("\n" + "=" * 70)
    logger.info("LOOP COMPILER - Turing Completeness via WASM")
    logger.info("=" * 70)

    correct = 0
    for nl_input, expected in test_cases:
        result = compiler.compile_and_run(nl_input)

        if result["success"] and result["result"] == expected:
            status = "OK"
            correct += 1
        elif result["success"]:
            status = f"WRONG (got {result['result']})"
        else:
            status = f"ERROR: {result.get('error', 'unknown')[:30]}"

        logger.info(f"\nInput: {nl_input}")
        if result.get("parsed"):
            p = result["parsed"]
            logger.info(f"  Parsed: type={p['type']}, start={p['start']}, end={p['end']}")
        logger.info(f"  Result: {result.get('result', 'N/A')} (expected {expected}) [{status}]")

    logger.info("\n" + "=" * 70)
    logger.info(f"ACCURACY: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")
    logger.info("=" * 70)

    # Show what WASM can do that the model can't
    logger.info("\n" + "=" * 70)
    logger.info("WHY THIS MATTERS:")
    logger.info("=" * 70)
    logger.info("The transformer CANNOT loop. It processes sequences in one pass.")
    logger.info("But WASM CAN loop. 'Sum 1 to 100' requires 100 iterations.")
    logger.info("The model emits the INTENT, WASM does the COMPUTATION.")
    logger.info("This is Turing completeness via hybrid architecture.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
