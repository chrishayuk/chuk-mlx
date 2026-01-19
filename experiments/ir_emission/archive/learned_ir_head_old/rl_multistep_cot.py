"""
Multi-Step CoT with Per-Step Verification.

Format:
  Step 1: 16 - 3 = 13
  Step 2: 13 - 4 = 9
  Step 3: 9 * 2 = 18

Each step is parseable. Each step is verifiable via WASM.
Reward = all steps execute correctly AND final answer matches.
"""

import sys
from pathlib import Path
import re
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


# =============================================================================
# EXPRESSION PARSER & WASM EXECUTION (same as before)
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


# =============================================================================
# MULTI-STEP COT PARSER
# =============================================================================

def parse_cot_steps(text: str) -> list[tuple[str, int]] | None:
    """
    Parse multi-step CoT output.

    Input: "Step 1: 16 - 3 = 13\nStep 2: 13 - 4 = 9\nStep 3: 9 * 2 = 18"
    Output: [("16 - 3", 13), ("13 - 4", 9), ("9 * 2", 18)]

    Also handles simpler formats like:
    "16 - 3 = 13\n13 - 4 = 9\n9 * 2 = 18"
    """
    steps = []

    # Try to extract step patterns
    # Pattern 1: "Step N: expr = result"
    # Pattern 2: "expr = result"

    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove "Step N:" prefix if present
        line = re.sub(r'^Step\s*\d+[:\.]?\s*', '', line, flags=re.IGNORECASE)

        # Look for "expr = result" pattern
        match = re.match(r'^(.+?)\s*=\s*(-?\d+)\s*$', line)
        if match:
            expr = match.group(1).strip()
            result = int(match.group(2))
            steps.append((expr, result))

    return steps if steps else None


def verify_cot_steps(steps: list[tuple[str, int]], runtime: WASMRuntime) -> dict:
    """
    Verify each step of CoT output.

    Returns:
        {
            "steps_correct": [True, True, False, ...],
            "all_correct": bool,
            "final_result": int or None,
            "errors": [...]
        }
    """
    results = {
        "steps_correct": [],
        "all_correct": True,
        "final_result": None,
        "errors": [],
        "step_details": []
    }

    for i, (expr, claimed_result) in enumerate(steps):
        ir = parse_expression(expr)

        if ir is None:
            results["steps_correct"].append(False)
            results["all_correct"] = False
            results["errors"].append(f"Step {i+1}: parse failed for '{expr}'")
            results["step_details"].append({"expr": expr, "claimed": claimed_result, "actual": None, "correct": False})
            continue

        actual_result = wasm_execute(ir, runtime)

        if actual_result is None:
            results["steps_correct"].append(False)
            results["all_correct"] = False
            results["errors"].append(f"Step {i+1}: execution failed for '{expr}'")
            results["step_details"].append({"expr": expr, "claimed": claimed_result, "actual": None, "correct": False})
            continue

        is_correct = actual_result == claimed_result
        results["steps_correct"].append(is_correct)
        results["step_details"].append({"expr": expr, "claimed": claimed_result, "actual": actual_result, "correct": is_correct})

        if not is_correct:
            results["all_correct"] = False
            results["errors"].append(f"Step {i+1}: {expr} = {actual_result}, not {claimed_result}")

    if steps:
        results["final_result"] = steps[-1][1]

    return results


def multistep_reward(cot_output: str, expected_answer: int, runtime: WASMRuntime) -> tuple[float, str, dict]:
    """
    Compute reward for multi-step CoT output.

    Returns: (reward, reason, details)

    Reward levels:
        1.0 = all steps correct, final answer correct
        0.7 = all steps correct, wrong final (rare - means wrong problem setup)
        0.5 = some steps correct, final answer correct (lucky)
        0.3 = some steps correct, wrong final
        0.1 = parseable but all steps wrong
        0.0 = parse failure
    """
    steps = parse_cot_steps(cot_output)

    if steps is None or len(steps) == 0:
        return 0.0, "parse_fail", {}

    verification = verify_cot_steps(steps, runtime)

    final_correct = verification["final_result"] == expected_answer
    steps_correct_count = sum(verification["steps_correct"])
    total_steps = len(verification["steps_correct"])

    if verification["all_correct"] and final_correct:
        return 1.0, "perfect", verification

    if verification["all_correct"] and not final_correct:
        return 0.7, f"steps_ok_wrong_final:{verification['final_result']}", verification

    if final_correct:
        return 0.5, f"partial_steps_lucky:{steps_correct_count}/{total_steps}", verification

    if steps_correct_count > 0:
        return 0.3, f"partial:{steps_correct_count}/{total_steps}", verification

    return 0.1, "all_steps_wrong", verification


# =============================================================================
# GENERATION
# =============================================================================

def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def format_target(question: str, steps: str) -> str:
    return f"Q: {question}\nA:\n{steps}"


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Greedy generation for evaluation."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        # Stop after seeing answer pattern or double newline
        if decoded.count('\n') >= 5 or decoded.endswith('\n\n'):
            break

    return tokenizer.decode(generated).strip()


def sample_from_model(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 100):
    """Sample with temperature for RL."""
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

        if next_token == tokenizer.eos_token_id:
            break

        log_prob = mx.log(probs[0, next_token] + 1e-10)
        log_probs.append(log_prob)

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if decoded.count('\n') >= 5 or decoded.endswith('\n\n'):
            break

    return tokenizer.decode(generated).strip(), log_probs


# =============================================================================
# TRAINING DATA - Multi-step problems
# =============================================================================

TRAIN_DATA = [
    # Two-step problems
    {
        "q": "Tom has 20 apples. He eats 5 and gives 3 away. How many left?",
        "steps": "Step 1: 20 - 5 = 15\nStep 2: 15 - 3 = 12",
        "ans": 12
    },
    {
        "q": "A store has 50 books. They sell 12 in the morning and 8 in the afternoon. How many left?",
        "steps": "Step 1: 50 - 12 = 38\nStep 2: 38 - 8 = 30",
        "ans": 30
    },
    {
        "q": "Lisa has 15 stickers. She gets 10 more, then gives 7 to her friend. How many now?",
        "steps": "Step 1: 15 + 10 = 25\nStep 2: 25 - 7 = 18",
        "ans": 18
    },
    {
        "q": "A farmer has 30 chickens. He buys 15 more, then sells 20. How many chickens?",
        "steps": "Step 1: 30 + 15 = 45\nStep 2: 45 - 20 = 25",
        "ans": 25
    },
    # Three-step problems
    {
        "q": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for baking. She sells the rest at $2 each. How much money?",
        "steps": "Step 1: 16 - 3 = 13\nStep 2: 13 - 4 = 9\nStep 3: 9 * 2 = 18",
        "ans": 18
    },
    {
        "q": "A bakery makes 100 cookies. They put 24 in boxes of 6. They sell 40 individual cookies. How many are left?",
        "steps": "Step 1: 100 - 24 = 76\nStep 2: 76 - 40 = 36",
        "ans": 36
    },
    {
        "q": "Sam has $50. He spends $12 on lunch, $8 on a book, and $15 on a gift. How much left?",
        "steps": "Step 1: 50 - 12 = 38\nStep 2: 38 - 8 = 30\nStep 3: 30 - 15 = 15",
        "ans": 15
    },
    {
        "q": "A bus has 45 passengers. At the first stop, 12 get off and 8 get on. At the second stop, 15 get off. How many on the bus?",
        "steps": "Step 1: 45 - 12 = 33\nStep 2: 33 + 8 = 41\nStep 3: 41 - 15 = 26",
        "ans": 26
    },
    # Multiplication chains
    {
        "q": "A factory makes 8 boxes per hour. Each box has 12 items. How many items in 3 hours?",
        "steps": "Step 1: 8 * 12 = 96\nStep 2: 96 * 3 = 288",
        "ans": 288
    },
    {
        "q": "There are 5 classrooms. Each has 6 rows of desks. Each row has 4 desks. Total desks?",
        "steps": "Step 1: 6 * 4 = 24\nStep 2: 5 * 24 = 120",
        "ans": 120
    },
    # Mixed operations
    {
        "q": "Tom buys 3 books at $15 each. He has a $10 coupon. How much does he pay?",
        "steps": "Step 1: 3 * 15 = 45\nStep 2: 45 - 10 = 35",
        "ans": 35
    },
    {
        "q": "A garden has 4 rows with 8 tomato plants each. 5 plants die. How many left?",
        "steps": "Step 1: 4 * 8 = 32\nStep 2: 32 - 5 = 27",
        "ans": 27
    },
    {
        "q": "Maria earns $12 per hour. She works 8 hours and spends $36 on groceries. How much left?",
        "steps": "Step 1: 12 * 8 = 96\nStep 2: 96 - 36 = 60",
        "ans": 60
    },
    {
        "q": "A parking lot has 6 rows with 15 spaces each. 47 cars are parked. How many empty spaces?",
        "steps": "Step 1: 6 * 15 = 90\nStep 2: 90 - 47 = 43",
        "ans": 43
    },
    # Division involved
    {
        "q": "48 students split into 4 equal groups. Each group gets 3 pizzas. Total pizzas needed?",
        "steps": "Step 1: 48 / 4 = 12\nStep 2: 12 * 3 = 36",
        "ans": 36
    },
    {
        "q": "A store has 120 apples. They make bags of 10 apples each. They sell 8 bags. How many apples left?",
        "steps": "Step 1: 8 * 10 = 80\nStep 2: 120 - 80 = 40",
        "ans": 40
    },
]

TEST_DATA = [
    {
        "q": "Emma has 25 candies. She eats 7 and gives 5 to her brother. How many left?",
        "steps": "Step 1: 25 - 7 = 18\nStep 2: 18 - 5 = 13",
        "ans": 13
    },
    {
        "q": "A shop has 80 shirts. They sell 25 in the morning and 18 in the afternoon. How many left?",
        "steps": "Step 1: 80 - 25 = 55\nStep 2: 55 - 18 = 37",
        "ans": 37
    },
    {
        "q": "Jake buys 4 packs of cards at $8 each. He uses a $5 discount. How much does he pay?",
        "steps": "Step 1: 4 * 8 = 32\nStep 2: 32 - 5 = 27",
        "ans": 27
    },
    {
        "q": "A library has 3 floors. Each floor has 12 shelves. Each shelf has 25 books. Total books?",
        "steps": "Step 1: 12 * 25 = 300\nStep 2: 3 * 300 = 900",
        "ans": 900
    },
]


# =============================================================================
# SFT TRAINING
# =============================================================================

def sft_train(model, tokenizer, data: list[dict], epochs: int = 5, lr: float = 2e-5):
    """Supervised fine-tuning for multi-step CoT."""
    print(f"\n  SFT Training: {epochs} epochs, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)

    def compute_loss(model, batch_tokens, batch_masks):
        logits = model(batch_tokens)
        if hasattr(logits, 'logits'):
            logits = logits.logits

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

    loss_and_grad = nn.value_and_grad(model, compute_loss)

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(data), 2):  # batch size 2
            batch = data[i:i+2]

            all_tokens = []
            all_masks = []

            for item in batch:
                full_text = format_target(item["q"], item["steps"])
                prompt = format_prompt(item["q"])

                full_tokens = tokenizer.encode(full_text)[:200]
                prompt_tokens = tokenizer.encode(prompt)

                mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

                all_tokens.append(full_tokens)
                all_masks.append(mask)

            max_len = max(len(t) for t in all_tokens)
            padded_tokens = [t + [0] * (max_len - len(t)) for t in all_tokens]
            padded_masks = [m + [0] * (max_len - len(m)) for m in all_masks]

            batch_tokens = mx.array(padded_tokens)
            batch_masks = mx.array(padded_masks, dtype=mx.float32)

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
             iterations: int = 10, lr: float = 1e-6, temperature: float = 0.7):
    """REINFORCE with per-step verification."""
    print(f"\n  RL Training: {iterations} iterations, lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    baseline = 0.5

    for iteration in range(iterations):
        samples = []
        for item in random.sample(data, min(4, len(data))):
            prompt = format_prompt(item["q"])
            output, log_probs = sample_from_model(model, tokenizer, prompt, temperature)
            reward, reason, details = multistep_reward(output, item["ans"], runtime)
            samples.append((item, output, log_probs, reward, reason))

        avg_reward = sum(s[3] for s in samples) / len(samples)
        perfect = sum(1 for s in samples if s[3] == 1.0)

        def compute_loss(model):
            total_loss = mx.array(0.0)

            for item, _, _, reward, _ in samples:
                prompt = format_prompt(item["q"])
                tokens = tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

                gen_log_prob = mx.array(0.0)
                for _ in range(50):
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

        loss_and_grad = nn.value_and_grad(model, compute_loss)
        loss, grads = loss_and_grad(model)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        baseline = 0.9 * baseline + 0.1 * avg_reward

        if (iteration + 1) % 2 == 0:
            print(f"    Iter {iteration+1}: reward={avg_reward:.2f}, perfect={perfect}/4")

    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, tokenizer, data: list[dict], runtime: WASMRuntime, label: str = ""):
    """Evaluate with per-step verification."""
    print(f"\n  {label}")
    print("-" * 90)

    perfect = 0
    partial = 0

    for item in data:
        prompt = format_prompt(item["q"])
        output = greedy_generate(model, tokenizer, prompt)
        reward, reason, details = multistep_reward(output, item["ans"], runtime)

        if reward == 1.0:
            perfect += 1
            status = "✓ perfect"
        elif reward >= 0.5:
            partial += 1
            status = f"~ {reason}"
        else:
            status = f"✗ {reason}"

        q_short = item["q"][:50] + "..." if len(item["q"]) > 50 else item["q"]
        print(f"\n  Q: {q_short}")
        print(f"  Expected: {item['ans']}")
        print(f"  Output:\n    {output.replace(chr(10), chr(10) + '    ')}")
        print(f"  Status: {status}")

    accuracy = perfect / len(data)
    print(f"\n  Perfect: {perfect}/{len(data)} = {accuracy:.0%}")
    print(f"  Partial: {partial}/{len(data)}")
    return accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  MULTI-STEP COT WITH PER-STEP VERIFICATION")
    print("  Each step is parseable → executable → verifiable")
    print("=" * 70)

    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer
    runtime = WASMRuntime()

    # Baseline
    print("\n" + "=" * 70)
    print("PHASE 0: BASELINE")
    print("=" * 70)
    model.freeze()
    baseline_acc = evaluate(model, tokenizer, TEST_DATA[:2], runtime, "Baseline (2 examples)")

    # SFT
    print("\n" + "=" * 70)
    print("PHASE 1: SFT COLD START")
    print("=" * 70)
    model.model.layers[-1].unfreeze()
    model.model.layers[-2].unfreeze()
    model.lm_head.unfreeze()

    model = sft_train(model, tokenizer, TRAIN_DATA, epochs=8, lr=2e-5)
    sft_acc = evaluate(model, tokenizer, TEST_DATA, runtime, "After SFT")

    # RL
    print("\n" + "=" * 70)
    print("PHASE 2: RL FINE-TUNE")
    print("=" * 70)
    model = rl_train(model, tokenizer, TRAIN_DATA, runtime, iterations=10, lr=5e-7)
    rl_acc = evaluate(model, tokenizer, TEST_DATA, runtime, "After RL")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
    Multi-Step CoT with Per-Step Verification

    Format:
      Step 1: 16 - 3 = 13
      Step 2: 13 - 4 = 9
      Step 3: 9 * 2 = 18

    Each step verified via WASM execution.

    Results:
      Baseline: {baseline_acc:.0%}
      After SFT: {sft_acc:.0%}
      After RL:  {rl_acc:.0%}
    """)


if __name__ == "__main__":
    main()
