"""
RL with Verifiable Rewards - Train Model to Emit CoT Directly.

Goal: Train the model so it naturally outputs "5 - 2 =" from word problems,
WITHOUT needing few-shot prompting at inference time.

Training: Q: <problem>\nA: → model samples → parse → WASM → reward
Inference: Same prompt, model directly emits expression (no few-shot needed)

The model learns to BE the CoT normalizer.
"""

import sys
from pathlib import Path
import json
import re

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from archive.codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from archive.wasm_runtime import WASMRuntime


# =============================================================================
# VERIFIABLE REWARD (via WASM)
# =============================================================================

def parse_expression(text: str) -> list[dict] | None:
    """Parse expression to IR using shunting yard."""
    text = text.strip().rstrip("=").strip()
    text = re.sub(r'[^0-9+\-*/() ]', '', text)
    tokens = re.findall(r'\d+|[+\-*/()]', text)

    if not tokens:
        return None

    output, op_stack = [], []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    op_map = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}

    try:
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

        return output if output else None
    except:
        return None


def wasm_execute(ir: list[dict], runtime: WASMRuntime) -> int | None:
    """Execute IR via WASM."""
    OP_MAP = {
        "add": IROpcode.I32_ADD, "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL, "divide": IROpcode.I32_DIV_S,
    }
    try:
        body = bytearray()
        for instr in ir:
            if instr['type'] == 'push':
                body.extend(encode_i32_const(instr['value']))
            else:
                body.extend(OPCODE_TO_WASM[OP_MAP[instr['op']]])
        result = runtime.execute(bytes(body))
        return result.result if result.success else None
    except:
        return None


def verifiable_reward(expression: str, expected: int, runtime: WASMRuntime) -> tuple[float, str]:
    """
    Verifiable reward via WASM execution.
    Returns (reward, reason).
    """
    ir = parse_expression(expression)

    if ir is None:
        return 0.0, "parse_fail"

    result = wasm_execute(ir, runtime)

    if result is None:
        return 0.1, "exec_fail"

    if result == expected:
        return 1.0, "correct"

    return 0.3, f"wrong:{result}"


# =============================================================================
# SAMPLE FROM MODEL
# =============================================================================

def sample_from_model(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 30):
    """
    Sample from model. Returns (text, list of log_probs).
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    log_probs = []

    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        logits = logits[:, -1, :] / temperature

        probs = mx.softmax(logits, axis=-1)
        next_token = int(mx.random.categorical(logits).item())

        log_prob = mx.log(probs[0, next_token] + 1e-10)
        log_probs.append(log_prob)

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if "=" in decoded or "\n" in decoded:
            break

    text = tokenizer.decode(generated).strip()
    if "=" in text:
        text = text[:text.index("=") + 1]
    if "\n" in text:
        text = text[:text.index("\n")]

    return text, log_probs


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 30) -> str:
    """Greedy generation (for evaluation)."""
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
        if "=" in decoded or "\n" in decoded:
            break

    text = tokenizer.decode(generated).strip()
    if "=" in text:
        text = text[:text.index("=") + 1]
    if "\n" in text:
        text = text[:text.index("\n")]

    return text


# =============================================================================
# SAMPLE DATA
# =============================================================================

SAMPLE_DATA = [
    # Simple operations
    {"question": "Tom has 5 apples and gets 3 more. How many total?", "answer": 8, "expr": "5 + 3 ="},
    {"question": "Sarah has 20 cookies and eats 7. How many left?", "answer": 13, "expr": "20 - 7 ="},
    {"question": "A box has 6 rows with 4 pencils each. Total pencils?", "answer": 24, "expr": "6 * 4 ="},
    {"question": "Split 36 marbles among 6 friends. Each gets?", "answer": 6, "expr": "36 / 6 ="},
    {"question": "Buy 4 books at $12 each. Total cost?", "answer": 48, "expr": "4 * 12 ="},
    {"question": "Start with 100 points, lose 35. How many left?", "answer": 65, "expr": "100 - 35 ="},
    {"question": "3 bags with 15 candies each. Total candies?", "answer": 45, "expr": "3 * 15 ="},
    {"question": "Divide 48 stickers among 8 kids. Each gets?", "answer": 6, "expr": "48 / 8 ="},
    {"question": "Have 25 dollars, earn 17 more. Total?", "answer": 42, "expr": "25 + 17 ="},
    {"question": "50 students, 12 leave. How many remain?", "answer": 38, "expr": "50 - 12 ="},
    # Slightly harder
    {"question": "Jenny has 8 apples. She gives 3 to Bob. How many left?", "answer": 5, "expr": "8 - 3 ="},
    {"question": "A store has 5 shelves with 9 books each. Total books?", "answer": 45, "expr": "5 * 9 ="},
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  RL TRAINING: Model Learns to Emit CoT Directly")
    print("  No few-shot prompting at inference time")
    print("=" * 70)

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    runtime = WASMRuntime()

    # -------------------------------------------------------------------------
    # PHASE 1: Baseline - what does raw model output?
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 1: Baseline (no training, no few-shot)")
    print("Prompt format: 'Q: <problem>\\nA:'")
    print("-" * 70)

    model.freeze()

    def format_prompt(question: str) -> str:
        return f"Q: {question}\nA:"

    print(f"\n{'Question':<50} {'Model Output':<25} {'Expected':<15}")
    print("-" * 90)

    baseline_correct = 0
    for item in SAMPLE_DATA[:6]:
        prompt = format_prompt(item["question"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = verifiable_reward(output, item["answer"], runtime)

        status = "✓" if reward == 1.0 else reason
        if reward == 1.0:
            baseline_correct += 1

        q_short = item["question"][:47] + "..." if len(item["question"]) > 50 else item["question"]
        print(f"{q_short:<50} {output:<25} {item['expr']:<15} {status}")

    print(f"\nBaseline accuracy (no training): {baseline_correct}/6")

    # -------------------------------------------------------------------------
    # PHASE 2: RL Training Loop (mini version)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 2: RL Training (5 iterations)")
    print("-" * 70)

    # Unfreeze for training
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()

    optimizer = optim.Adam(learning_rate=5e-6)

    # Training loop
    baseline_reward = 0.3  # Initial baseline estimate

    for iteration in range(5):
        # Sample batch
        batch_samples = []  # (question, output, log_probs, reward)

        for item in SAMPLE_DATA[:8]:
            prompt = format_prompt(item["question"])
            output, log_probs = sample_from_model(model, tokenizer, prompt, temperature=0.8)
            reward, reason = verifiable_reward(output, item["answer"], runtime)
            batch_samples.append((item, output, log_probs, reward))

        avg_reward = sum(s[3] for s in batch_samples) / len(batch_samples)

        # REINFORCE update
        def compute_loss(model):
            """Policy gradient loss."""
            total_loss = mx.array(0.0)

            for item, target_output, _, reward in batch_samples:
                prompt = format_prompt(item["question"])
                tokens = tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

                # Forward pass to compute log probs
                gen_log_prob = mx.array(0.0)
                for _ in range(20):
                    output = model(input_ids)
                    logits = output.logits if hasattr(output, 'logits') else output
                    logits = logits[:, -1, :]

                    probs = mx.softmax(logits, axis=-1)
                    next_token = mx.argmax(logits, axis=-1)
                    log_prob = mx.log(mx.max(probs) + 1e-10)
                    gen_log_prob = gen_log_prob + log_prob

                    input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

                # Weight by advantage
                advantage = reward - baseline_reward
                total_loss = total_loss - gen_log_prob * advantage

            return total_loss / len(batch_samples)

        loss_and_grad = nn.value_and_grad(model, compute_loss)
        loss, grads = loss_and_grad(model)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Update baseline
        baseline_reward = 0.9 * baseline_reward + 0.1 * avg_reward

        # Show samples
        correct_count = sum(1 for s in batch_samples if s[3] == 1.0)
        print(f"Iter {iteration+1}: avg_reward={avg_reward:.2f}, correct={correct_count}/8, loss={loss.item():.4f}")

        # Show one sample
        sample = batch_samples[0]
        print(f"  Sample: '{sample[0]['question'][:40]}...' → '{sample[1]}' (r={sample[3]:.1f})")

    # -------------------------------------------------------------------------
    # PHASE 3: Evaluation after training
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3: Evaluation after RL training")
    print("-" * 70)

    print(f"\n{'Question':<50} {'Model Output':<25} {'Expected':<15}")
    print("-" * 90)

    post_correct = 0
    for item in SAMPLE_DATA:
        prompt = format_prompt(item["question"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = verifiable_reward(output, item["answer"], runtime)

        status = "✓" if reward == 1.0 else reason
        if reward == 1.0:
            post_correct += 1

        q_short = item["question"][:47] + "..." if len(item["question"]) > 50 else item["question"]
        print(f"{q_short:<50} {output:<25} {item['expr']:<15} {status}")

    print(f"\nPost-training accuracy: {post_correct}/{len(SAMPLE_DATA)}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
    Architecture:
      Training:  Q: <problem>\\nA: → sample → parse → WASM → reward
      Inference: Q: <problem>\\nA: → model emits expression (NO few-shot)

    The model learns to BE the CoT normalizer:
      "Tom has 5 apples and gets 3 more" → "5 + 3 ="

    Reward is VERIFIABLE via WASM execution:
      1.0 = WASM exec matches expected answer
      0.3 = valid parse, wrong answer
      0.0 = parse failure

    Results:
      Baseline (no training): {baseline_correct}/6
      After RL training: {post_correct}/{len(SAMPLE_DATA)}
    """)


if __name__ == "__main__":
    main()
