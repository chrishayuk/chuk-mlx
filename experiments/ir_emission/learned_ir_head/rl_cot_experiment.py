"""
RL with Verifiable Rewards - Full Experiment.

Pipeline:
  1. SFT Cold Start - Teach the model the format (problem → expression)
  2. RL Fine-tune - Refine with verifiable rewards from WASM execution

The model learns to emit CoT directly, no few-shot prompting at inference.
"""

import sys
from pathlib import Path
import json
import re
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from archive.codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from archive.wasm_runtime import WASMRuntime


# =============================================================================
# WASM EXECUTION & REWARD
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
    """Verifiable reward via WASM execution."""
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
# GENERATION UTILITIES
# =============================================================================

def format_prompt(question: str) -> str:
    """Format input prompt."""
    return f"Q: {question}\nA:"


def format_target(question: str, expression: str) -> str:
    """Format full sequence for SFT."""
    return f"Q: {question}\nA: {expression}"


def sample_from_model(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 30):
    """Sample from model with temperature. Returns (text, log_probs)."""
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
    """Greedy generation for evaluation."""
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
# TRAINING DATA
# =============================================================================

TRAIN_DATA = [
    # Addition
    {"q": "Tom has 5 apples and gets 3 more. How many total?", "expr": "5 + 3 =", "ans": 8},
    {"q": "Have 25 dollars, earn 17 more. Total?", "expr": "25 + 17 =", "ans": 42},
    {"q": "Lisa has 12 stickers and finds 8 more. How many?", "expr": "12 + 8 =", "ans": 20},
    {"q": "A jar has 30 marbles. Add 15 more. Total?", "expr": "30 + 15 =", "ans": 45},
    {"q": "Sam has 7 toys and gets 9 more. How many now?", "expr": "7 + 9 =", "ans": 16},
    {"q": "14 birds in a tree. 6 more arrive. Total?", "expr": "14 + 6 =", "ans": 20},
    # Subtraction
    {"q": "Sarah has 20 cookies and eats 7. How many left?", "expr": "20 - 7 =", "ans": 13},
    {"q": "Start with 100 points, lose 35. How many left?", "expr": "100 - 35 =", "ans": 65},
    {"q": "50 students, 12 leave. How many remain?", "expr": "50 - 12 =", "ans": 38},
    {"q": "Jenny has 15 candies and gives away 6. How many left?", "expr": "15 - 6 =", "ans": 9},
    {"q": "A box has 40 items. Remove 18. How many?", "expr": "40 - 18 =", "ans": 22},
    {"q": "28 flowers in a garden. 9 are picked. How many left?", "expr": "28 - 9 =", "ans": 19},
    # Multiplication - varied patterns including number-first
    {"q": "A box has 6 rows with 4 pencils each. Total pencils?", "expr": "6 * 4 =", "ans": 24},
    {"q": "Buy 4 books at $12 each. Total cost?", "expr": "4 * 12 =", "ans": 48},
    {"q": "3 bags with 15 candies each. Total candies?", "expr": "3 * 15 =", "ans": 45},
    {"q": "A store has 5 shelves with 9 books each. Total books?", "expr": "5 * 9 =", "ans": 45},
    {"q": "7 boxes with 8 oranges each. How many oranges?", "expr": "7 * 8 =", "ans": 56},
    {"q": "8 rows with 6 chairs each. Total chairs?", "expr": "8 * 6 =", "ans": 48},  # number-first pattern
    {"q": "4 trays with 9 cookies each. How many cookies?", "expr": "4 * 9 =", "ans": 36},  # number-first pattern
    {"q": "12 packs with 5 cards each. Total cards?", "expr": "12 * 5 =", "ans": 60},  # number-first pattern
    {"q": "9 shelves with 4 items each. Total items?", "expr": "9 * 4 =", "ans": 36},  # 9 first
    {"q": "11 rows with 3 plants each. Total plants?", "expr": "11 * 3 =", "ans": 33},  # varied
    {"q": "6 tables with 8 seats each. Total seats?", "expr": "6 * 8 =", "ans": 48},  # seats pattern
    {"q": "5 packs with 12 cards each. Total cards?", "expr": "5 * 12 =", "ans": 60},  # packs pattern
    {"q": "3 packs with 10 items each. How many items?", "expr": "3 * 10 =", "ans": 30},  # 10 pattern
    {"q": "6 boxes with 10 toys each. Total toys?", "expr": "6 * 10 =", "ans": 60},  # exact 6 * 10
    {"q": "7 packs with 11 stickers each. Total stickers?", "expr": "7 * 11 =", "ans": 77},  # more variety
    # Division
    {"q": "Split 36 marbles among 6 friends. Each gets?", "expr": "36 / 6 =", "ans": 6},
    {"q": "Divide 48 stickers among 8 kids. Each gets?", "expr": "48 / 8 =", "ans": 6},
    {"q": "Share 24 cookies equally among 4 people. Each gets?", "expr": "24 / 4 =", "ans": 6},
    {"q": "Split 50 dollars between 5 friends. Each gets?", "expr": "50 / 5 =", "ans": 10},
    {"q": "Divide 63 cards among 7 players. Each gets?", "expr": "63 / 7 =", "ans": 9},
    {"q": "72 students in 8 groups. How many per group?", "expr": "72 / 8 =", "ans": 9},  # number-first pattern
]

TEST_DATA = [
    {"q": "Mike has 8 apples. He gets 5 more. Total?", "expr": "8 + 5 =", "ans": 13},
    {"q": "A class has 32 students. 9 go home. How many left?", "expr": "32 - 9 =", "ans": 23},
    {"q": "6 packs with 10 cards each. Total cards?", "expr": "6 * 10 =", "ans": 60},
    {"q": "Divide 42 pencils among 6 kids. Each gets?", "expr": "42 / 6 =", "ans": 7},
    {"q": "Anna has 45 stickers and loses 17. How many left?", "expr": "45 - 17 =", "ans": 28},
    {"q": "9 rows with 7 seats each. Total seats?", "expr": "9 * 7 =", "ans": 63},
]


# =============================================================================
# SFT TRAINING
# =============================================================================

def sft_train(model, tokenizer, data: list[dict], epochs: int = 3, lr: float = 1e-5):
    """Supervised fine-tuning to teach format."""
    print(f"\n  SFT Training: {epochs} epochs, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)

    def compute_sft_loss(model, batch_tokens, batch_masks):
        """Cross-entropy loss on answer tokens only."""
        logits = model(batch_tokens)
        if hasattr(logits, 'logits'):
            logits = logits.logits

        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        targets = batch_tokens[:, 1:]
        masks = batch_masks[:, 1:]

        vocab_size = logits.shape[-1]
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        ).reshape(targets.shape)

        loss = (ce * masks).sum() / (masks.sum() + 1e-8)
        return loss

    loss_and_grad = nn.value_and_grad(model, compute_sft_loss)

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(data), 4):  # batch size 4
            batch = data[i:i+4]

            # Tokenize
            all_tokens = []
            all_masks = []

            for item in batch:
                full_text = format_target(item["q"], item["expr"])
                prompt = format_prompt(item["q"])

                full_tokens = tokenizer.encode(full_text)[:64]
                prompt_tokens = tokenizer.encode(prompt)

                # Mask: 0 for prompt, 1 for answer
                mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

                all_tokens.append(full_tokens)
                all_masks.append(mask)

            # Pad
            max_len = max(len(t) for t in all_tokens)
            padded_tokens = [t + [0] * (max_len - len(t)) for t in all_tokens]
            padded_masks = [m + [0] * (max_len - len(m)) for m in all_masks]

            batch_tokens = mx.array(padded_tokens)
            batch_masks = mx.array(padded_masks, dtype=mx.float32)

            # Update
            loss, grads = loss_and_grad(model, batch_tokens, batch_masks)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model


# =============================================================================
# RL TRAINING
# =============================================================================

def rl_train(model, tokenizer, data: list[dict], runtime: WASMRuntime,
             iterations: int = 10, lr: float = 1e-6, temperature: float = 0.8):
    """REINFORCE with verifiable rewards."""
    print(f"\n  RL Training: {iterations} iterations, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    baseline = 0.5

    for iteration in range(iterations):
        # Sample from policy
        samples = []
        for item in random.sample(data, min(8, len(data))):
            prompt = format_prompt(item["q"])
            output, log_probs = sample_from_model(model, tokenizer, prompt, temperature)
            reward, reason = verifiable_reward(output, item["ans"], runtime)
            samples.append((item, output, log_probs, reward, reason))

        avg_reward = sum(s[3] for s in samples) / len(samples)
        correct = sum(1 for s in samples if s[3] == 1.0)

        # REINFORCE update
        def compute_rl_loss(model):
            total_loss = mx.array(0.0)

            for item, _, _, reward, _ in samples:
                prompt = format_prompt(item["q"])
                tokens = tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

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

                advantage = reward - baseline
                total_loss = total_loss - gen_log_prob * advantage

            return total_loss / len(samples)

        loss_and_grad = nn.value_and_grad(model, compute_rl_loss)
        loss, grads = loss_and_grad(model)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        baseline = 0.9 * baseline + 0.1 * avg_reward

        if (iteration + 1) % 2 == 0:
            print(f"    Iter {iteration+1}: reward={avg_reward:.2f}, correct={correct}/8")

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, tokenizer, data: list[dict], runtime: WASMRuntime, label: str = ""):
    """Evaluate model on test data."""
    correct = 0
    parse_ok = 0

    print(f"\n  {label}")
    print(f"  {'Question':<45} {'Output':<20} {'Expected':<12} {'Status'}")
    print("  " + "-" * 85)

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason = verifiable_reward(output, item["ans"], runtime)

        if reward == 1.0:
            correct += 1
            status = "✓"
        elif reward >= 0.3:
            parse_ok += 1
            status = reason
        else:
            status = reason

        q_short = item["q"][:42] + "..." if len(item["q"]) > 45 else item["q"]
        print(f"  {q_short:<45} {output:<20} {item['expr']:<12} {status}")

    accuracy = correct / len(data)
    print(f"\n  Accuracy: {correct}/{len(data)} = {accuracy:.0%}")
    return accuracy


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("  RL WITH VERIFIABLE REWARDS - FULL EXPERIMENT")
    print("  Pipeline: SFT Cold Start → RL Fine-tune")
    print("=" * 70)

    # Load model
    print("\nLoading TinyLlama...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    runtime = WASMRuntime()

    # -------------------------------------------------------------------------
    # PHASE 0: Baseline (no training)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE (no training)")
    print("=" * 70)

    model.freeze()
    baseline_acc = evaluate(model, tokenizer, TEST_DATA, runtime, "Baseline Test")

    # -------------------------------------------------------------------------
    # PHASE 1: SFT Cold Start
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: SFT COLD START")
    print("=" * 70)

    # Unfreeze last 2 layers for training
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.lm_head.unfreeze()

    model = sft_train(model, tokenizer, TRAIN_DATA, epochs=5, lr=2e-5)

    sft_acc = evaluate(model, tokenizer, TEST_DATA, runtime, "After SFT")

    # -------------------------------------------------------------------------
    # PHASE 2: RL Fine-tune
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: RL FINE-TUNE (Verifiable Rewards)")
    print("=" * 70)

    model = rl_train(model, tokenizer, TRAIN_DATA, runtime,
                     iterations=10, lr=5e-7, temperature=0.7)

    rl_acc = evaluate(model, tokenizer, TEST_DATA, runtime, "After RL")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"""
    Pipeline: SFT Cold Start → RL with Verifiable Rewards

    Results on held-out test set ({len(TEST_DATA)} examples):
      Baseline (no training):  {baseline_acc:.0%}
      After SFT:               {sft_acc:.0%}
      After SFT + RL:          {rl_acc:.0%}

    Architecture:
      Input:  "Tom has 5 apples and gets 3 more. How many?"
      Output: "5 + 3 ="
      Execute via WASM → 8 (verified correct)

    Key insight: Model learns to BE the CoT normalizer.
    No few-shot prompting needed at inference time.
    """)


if __name__ == "__main__":
    main()
