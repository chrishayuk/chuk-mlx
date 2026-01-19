"""
Full Learned IR Pipeline.

End-to-end: NL → L12 Hidden States → IR Head → WASM → Result

No text generation. No regex. Just learned structure extraction and execution.
"""

import sys
from pathlib import Path
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add project root for chuk_lazarus imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from chuk_lazarus.models_v2.loader import load_model
from experiments.ir_emission.shared import OPCODE_TO_WASM, IROpcode, encode_i32_const, WASMRuntime


class LearnedIRHead(nn.Module):
    """
    IR Head that extracts (operation, operand_a, operand_b) from hidden states.

    Uses classification for operation, regression for operands.
    (Regression worked better in quick tests than 256-bin classification)
    """

    OP_NAMES = ["add", "subtract", "multiply", "divide"]
    OP_TO_IDX = {op: i for i, op in enumerate(OP_NAMES)}

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Operation classifier
        self.op_head = nn.Linear(256, 4)

        # Operand regressors
        self.a_head = nn.Linear(256, 1)
        self.b_head = nn.Linear(256, 1)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Forward pass: hidden states → (op_logits, a_pred, b_pred)"""
        encoded = self.encoder(h)
        return (
            self.op_head(encoded),
            self.a_head(encoded).squeeze(-1),
            self.b_head(encoded).squeeze(-1),
        )

    def decode(self, op_logits, a_pred, b_pred) -> dict:
        """Decode outputs to IR instruction."""
        mx.eval(op_logits, a_pred, b_pred)
        return {
            "op": self.OP_NAMES[int(mx.argmax(op_logits).item())],
            "a": max(0, int(round(float(a_pred.item())))),
            "b": max(0, int(round(float(b_pred.item())))),
        }


class LearnedIRPipeline:
    """
    Complete pipeline: NL → Learned IR Head → WASM → Result

    Architecture:
        1. Tokenize NL input
        2. Run through frozen backbone to layer 12
        3. IR head extracts (op, a, b) from hidden states
        4. Build WASM IR from extracted structure
        5. Execute WASM and return result
    """

    OP_TO_WASM = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    def __init__(self, model, tokenizer, ir_head: LearnedIRHead, layer_idx: int = 12):
        self.model = model
        self.tokenizer = tokenizer
        self.ir_head = ir_head
        self.layer_idx = layer_idx
        self.runtime = WASMRuntime()

    def get_hidden_states(self, text: str) -> mx.array:
        """Extract normalized L12 hidden states from text."""
        tokens = mx.array([self.tokenizer.encode(text)])
        backbone = self.model.model

        h = backbone.embed_tokens(tokens)
        seq_len = tokens.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(backbone.layers):
            if i >= self.layer_idx:
                break
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output

        h = backbone.norm(h)
        return h[:, -1, :]  # Last token

    def extract_ir(self, text: str) -> dict:
        """Extract IR instruction from natural language."""
        h = self.get_hidden_states(text)
        mx.eval(h)

        op_logits, a_logits, b_logits = self.ir_head(h)
        return self.ir_head.decode(op_logits[0], a_logits[0], b_logits[0])

    def build_wasm(self, ir: dict) -> bytes:
        """Build WASM bytecode from IR instruction."""
        body = bytearray()
        body.extend(encode_i32_const(ir["a"]))
        body.extend(encode_i32_const(ir["b"]))
        body.extend(OPCODE_TO_WASM[self.OP_TO_WASM[ir["op"]]])
        return bytes(body)

    def execute(self, text: str) -> dict:
        """
        Full pipeline: NL → IR → WASM → Result

        Returns dict with:
            - input: Original text
            - ir: Extracted IR instruction
            - wasm_hex: WASM bytecode as hex
            - result: Execution result
            - success: Whether execution succeeded
        """
        ir = self.extract_ir(text)
        wasm_bytes = self.build_wasm(ir)
        result = self.runtime.execute(wasm_bytes)

        return {
            "input": text,
            "ir": ir,
            "wasm_hex": wasm_bytes.hex(),
            "result": result.result if result.success else None,
            "success": result.success,
            "error": result.error,
        }


def generate_training_data(n: int = 500, max_val: int = 100) -> list[dict]:
    """Generate training data with simple canonical format."""
    random.seed(42)
    data = []

    # Simple format that we know works: "A op B"
    ops = [
        ("+", "add"),
        ("-", "subtract"),
        ("*", "multiply"),
        ("/", "divide"),
    ]

    for _ in range(n):
        sym, op = random.choice(ops)
        if op == "divide":
            b = random.randint(1, 20)
            a = b * random.randint(1, max_val // max(1, b))
        else:
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)

        text = f"{a} {sym} {b}"
        data.append({"text": text, "op": op, "a": a, "b": b})

    return data


def compute_expected(op: str, a: int, b: int) -> int:
    """Compute expected result."""
    if op == "add":
        return a + b
    elif op == "subtract":
        return a - b
    elif op == "multiply":
        return a * b
    elif op == "divide":
        return a // b if b != 0 else 0
    return 0


def train_ir_head(model, tokenizer, num_epochs: int = 20, num_examples: int = 500):
    """Train the IR head."""
    print("Generating training data...")
    train_data = generate_training_data(num_examples)

    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    ir_head = LearnedIRHead(hidden_dim=hidden_dim)

    optimizer = optim.Adam(learning_rate=0.002)

    def get_hidden(text):
        tokens = mx.array([tokenizer.encode(text)])
        backbone = model.model
        h = backbone.embed_tokens(tokens)
        seq_len = tokens.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)
        for i, layer in enumerate(backbone.layers):
            if i >= 12:
                break
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output
        return backbone.norm(h)[:, -1, :]

    def loss_fn(ir_head, h, op_target, a_target, b_target):
        op_logits, a_pred, b_pred = ir_head(h)
        op_loss = nn.losses.cross_entropy(op_logits, op_target, reduction="mean")
        # MSE for regression, scaled down
        a_loss = mx.mean((a_pred - a_target.astype(mx.float32)) ** 2) / 10000
        b_loss = mx.mean((b_pred - b_target.astype(mx.float32)) ** 2) / 10000
        return 0.5 * op_loss + 0.25 * a_loss + 0.25 * b_loss

    loss_and_grad = nn.value_and_grad(ir_head, loss_fn)

    print(f"Training IR head ({num_examples} examples, {num_epochs} epochs)...")
    print("-" * 50)

    for epoch in range(num_epochs):
        random.shuffle(train_data)
        total_loss = 0

        for ex in train_data:
            h = get_hidden(ex["text"])
            mx.eval(h)

            op_idx = LearnedIRHead.OP_TO_IDX[ex["op"]]
            loss, grads = loss_and_grad(
                ir_head, h,
                mx.array([op_idx]),
                mx.array([ex["a"]]),
                mx.array([ex["b"]]),
            )

            optimizer.update(ir_head, grads)
            mx.eval(ir_head.parameters())
            total_loss += float(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:2d}: loss={total_loss/len(train_data):.4f}")

    return ir_head


def main():
    print("=" * 60)
    print("  FULL LEARNED IR PIPELINE")
    print("  NL → L12 Hidden → IR Head → WASM → Result")
    print("=" * 60)

    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    model.freeze()

    # Train IR head
    ir_head = train_ir_head(model, tokenizer, num_epochs=20, num_examples=500)

    # Create pipeline
    pipeline = LearnedIRPipeline(model, tokenizer, ir_head)

    # Test cases
    print("\n" + "=" * 60)
    print("  END-TO-END EXECUTION")
    print("=" * 60)

    # Test with same format as training to verify pipeline works
    test_cases = [
        ("25 + 17", "add", 25, 17),
        ("50 - 15", "subtract", 50, 15),
        ("7 * 8", "multiply", 7, 8),
        ("48 / 6", "divide", 48, 6),
        ("33 + 19", "add", 33, 19),
        ("100 - 37", "subtract", 100, 37),
        ("12 * 9", "multiply", 12, 9),
        ("144 / 12", "divide", 144, 12),
    ]

    correct = 0
    print(f"\n{'Input':<30} {'IR':<25} {'Result':<10} {'Expected':<10} {'Status'}")
    print("-" * 90)

    for text, expected_op, a, b in test_cases:
        expected = compute_expected(expected_op, a, b)
        result = pipeline.execute(text)

        ir_str = f"{result['ir']['op']}({result['ir']['a']}, {result['ir']['b']})"

        if result["success"] and result["result"] == expected:
            status = "✓"
            correct += 1
        elif result["success"]:
            status = f"✗ (got {result['result']})"
        else:
            status = f"✗ error: {result['error']}"

        print(f"{text:<30} {ir_str:<25} {result['result'] or 'ERR':<10} {expected:<10} {status}")

    print("-" * 90)
    print(f"\nAccuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")

    # Show architecture
    print("\n" + "=" * 60)
    print("  ARCHITECTURE SUMMARY")
    print("=" * 60)
    print("""
    ┌─────────────────────────────────────────────────────┐
    │  "Add 25 and 17"                                    │
    │         ↓                                           │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  Frozen Backbone (TinyLlama L0-L12)         │    │
    │  │  → Normalized hidden states                 │    │
    │  └─────────────────────────────────────────────┘    │
    │         ↓                                           │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  Learned IR Head (trained, ~500K params)    │    │
    │  │  → op=add, a=25, b=17                       │    │
    │  └─────────────────────────────────────────────┘    │
    │         ↓                                           │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  WASM Builder (deterministic)               │    │
    │  │  → [i32.const 25, i32.const 17, i32.add]    │    │
    │  └─────────────────────────────────────────────┘    │
    │         ↓                                           │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  WASM Runtime (wasmtime)                    │    │
    │  │  → 42                                       │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

    NO TEXT GENERATION. NO REGEX. JUST LEARNED PROJECTION.
    """)


if __name__ == "__main__":
    main()
