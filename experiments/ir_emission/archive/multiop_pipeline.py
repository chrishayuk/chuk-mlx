#!/usr/bin/env python3
"""
Multi-Op Chain Pipeline.

Extends the neural compiler to handle multi-step operations:
  "16 - 3, then multiply by 5" → 65

Uses few-shot prompting to parse chains into sequences of canonical operations.
"""

import json
import logging
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from codebook import IROpcode, encode_i32_const, OPCODE_TO_WASM
from wasm_runtime import WASMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiOpCompiler:
    """
    Multi-operation neural compiler.

    Handles chains like:
      "16 - 3, then multiply by 5"
    By parsing into:
      Step 1: 16 - 3 = 13
      Step 2: 13 * 5 = 65
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        classifier_path: str = "experiments/ir_emission/checkpoints/dual_reward/final/adapters.safetensors",
    ):
        from chuk_lazarus.models_v2.loader import load_model
        from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora

        logger.info("Loading models...")
        result = load_model(model_name)
        self.base_model = result.model
        self.tokenizer = result.tokenizer
        self.config = result.config

        # Classifier model
        cls_result = load_model(model_name)
        self.cls_model = cls_result.model

        lora_config = LoRAConfig(rank=32, alpha=64.0, target_modules=["v_proj", "o_proj"])
        apply_lora(self.cls_model, lora_config)

        with safe_open(classifier_path, framework="numpy") as f:
            lora_weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

        backbone = self.cls_model.model
        for name, param in lora_weights.items():
            if name.startswith("model."):
                name = name[6:]
            parts = name.split(".")
            try:
                obj = backbone
                for p in parts[:-1]:
                    obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
                if parts[-1] == "lora_a":
                    obj.lora_A = param
                elif parts[-1] == "lora_b":
                    obj.lora_B = param
            except:
                pass

        self.base_model.freeze()
        self.cls_model.freeze()
        self.runtime = WASMRuntime()

        self.classifier_tokens = {
            "add": 788, "subtract": 23197, "multiply": 22932, "divide": 16429
        }
        self.class_to_ir = {
            "add": IROpcode.I32_ADD, "subtract": IROpcode.I32_SUB,
            "multiply": IROpcode.I32_MUL, "divide": IROpcode.I32_DIV_S
        }
        self.decision_layer = int(self.config.num_hidden_layers * 0.55)

    def parse_chain(self, nl_input: str) -> list[dict]:
        """Parse multi-op NL into sequence of operations."""
        # Count how many operations are in the input
        # Check for parenthesized expressions like "(a + b) * c"
        paren_match = re.match(r"\(.*?\)\s*[+\-*/]", nl_input)
        if paren_match:
            num_ops = 2
        else:
            op_keywords = ["then", "and then", ","]
            num_ops = 1
            for kw in op_keywords:
                num_ops += nl_input.lower().count(kw)
        num_ops = min(num_ops, 3)  # Cap at 3 operations

        prompt = f"""<|system|>
Parse the math chain into exactly the steps needed. Stop after the last step.
</s>
<|user|>
16 - 3, then multiply by 5
</s>
<|assistant|>
Step 1: 16 - 3 =
Step 2: result * 5 =</s>
<|user|>
Add 10 and 20, then subtract 5
</s>
<|assistant|>
Step 1: 10 + 20 =
Step 2: result - 5 =</s>
<|user|>
(8 + 4) * 3
</s>
<|assistant|>
Step 1: 8 + 4 =
Step 2: result * 3 =</s>
<|user|>
6 * 7
</s>
<|assistant|>
Step 1: 6 * 7 =</s>
<|user|>
{nl_input}
</s>
<|assistant|>
"""
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids
        step_count = 0
        for _ in range(40):
            output = self.base_model(generated_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
            mx.eval(generated_ids)

            decoded = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())

            # Count steps in decoded
            step_count = decoded.count("Step ")

            # Stop if we've generated enough steps or hit end token
            if "</s>" in decoded or step_count >= num_ops:
                break

            # Also stop if we see a complete step pattern ending
            if f"Step {num_ops}:" in decoded and "=" in decoded.split(f"Step {num_ops}:")[-1]:
                break

        response = self.tokenizer.decode(generated_ids[0, prompt_len:].tolist())
        response = response.replace("</s>", "").strip()

        # Only keep lines with expected step numbers
        lines = response.split("\n")
        valid_lines = []
        for i in range(1, num_ops + 1):
            for line in lines:
                if f"Step {i}:" in line:
                    valid_lines.append(line)
                    break
        response = "\n".join(valid_lines)

        # Parse steps
        steps = []
        for line in response.split("\n"):
            line = line.strip()
            if not line.startswith("Step"):
                continue

            # Extract: "Step N: expr = "
            match = re.search(r"Step \d+:\s*(.+?)\s*=", line)
            if not match:
                continue

            expr = match.group(1).strip()

            # Parse expression
            # First step: "num op num"
            # Later steps: "result op num"
            num_match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)", expr)
            result_match = re.match(r"result\s*([+\-*/])\s*(\d+)", expr)

            if num_match:
                a, op, b = num_match.groups()
                steps.append({
                    "a": int(a),
                    "b": int(b),
                    "op": op,
                    "use_result": False,
                })
            elif result_match:
                op, b = result_match.groups()
                steps.append({
                    "a": None,  # Will use previous result
                    "b": int(b),
                    "op": op,
                    "use_result": True,
                })

        return steps

    def classify(self, canonical: str) -> str:
        """Classify operation from canonical form."""
        backbone = self.cls_model.model
        tokens = self.tokenizer.encode(canonical)
        input_ids = mx.array([tokens])

        h = backbone.embed_tokens(input_ids)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(backbone.layers):
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output
            if i == self.decision_layer:
                break

        h_normed = backbone.norm(h)
        head_output = self.cls_model.lm_head(h_normed)
        logits = head_output.logits if hasattr(head_output, "logits") else head_output
        probs = mx.softmax(logits[0, -1, :])

        best_class, best_prob = None, 0
        for cls, tid in self.classifier_tokens.items():
            prob = float(probs[tid].item())
            if prob > best_prob:
                best_prob, best_class = prob, cls
        return best_class

    def build_chain_ir(self, steps: list[dict]) -> bytes:
        """Build WASM IR for multi-op chain."""
        body = bytearray()

        for i, step in enumerate(steps):
            if i == 0:
                # First step: a op b
                body.extend(encode_i32_const(step["a"]))
                body.extend(encode_i32_const(step["b"]))
            else:
                # Later steps: result is on stack, push b
                body.extend(encode_i32_const(step["b"]))

            # Get operation from classifier
            if step["use_result"]:
                canonical = f"1 {step['op']} {step['b']} = "  # Dummy for classification
            else:
                canonical = f"{step['a']} {step['op']} {step['b']} = "

            op_class = self.classify(canonical)
            ir_op = self.class_to_ir[op_class]
            body.extend(OPCODE_TO_WASM[ir_op])

        return bytes(body)

    def compile_and_run(self, nl_input: str) -> dict:
        """Compile and execute multi-op chain."""
        # Parse into steps
        steps = self.parse_chain(nl_input)

        if not steps:
            return {
                "input": nl_input,
                "success": False,
                "error": "Failed to parse chain",
            }

        # Build IR
        try:
            ir_bytes = self.build_chain_ir(steps)
            result = self.runtime.execute(ir_bytes)

            if result.success:
                return {
                    "input": nl_input,
                    "steps": steps,
                    "ir_hex": ir_bytes.hex(),
                    "result": result.result,
                    "success": True,
                }
            else:
                return {
                    "input": nl_input,
                    "steps": steps,
                    "success": False,
                    "error": result.error,
                }
        except Exception as e:
            return {
                "input": nl_input,
                "steps": steps,
                "success": False,
                "error": str(e),
            }


def main():
    compiler = MultiOpCompiler()

    test_cases = [
        # Two-op chains
        ("16 - 3, then multiply by 5", (16 - 3) * 5),        # 65
        ("Add 10 and 20, then subtract 5", (10 + 20) - 5),   # 25
        ("Multiply 4 by 7, then add 8", (4 * 7) + 8),        # 36
        ("Start with 50, subtract 20, then divide by 3", (50 - 20) // 3),  # 10
        ("(8 + 4) * 3", (8 + 4) * 3),                        # 36
        ("(20 - 5) * 2", (20 - 5) * 2),                      # 30
        ("6 * 7, then add 10", (6 * 7) + 10),                # 52
        ("100 - 40, then divide by 2", (100 - 40) // 2),     # 30
    ]

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-OP CHAIN COMPILER")
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
        if result.get("steps"):
            logger.info(f"  Steps: {len(result['steps'])}")
            for i, s in enumerate(result["steps"]):
                a = "result" if s.get("use_result") else s["a"]
                logger.info(f"    {i+1}: {a} {s['op']} {s['b']}")
        logger.info(f"  Result: {result.get('result', 'N/A')} (expected {expected}) [{status}]")

    logger.info("\n" + "=" * 70)
    logger.info(f"ACCURACY: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
