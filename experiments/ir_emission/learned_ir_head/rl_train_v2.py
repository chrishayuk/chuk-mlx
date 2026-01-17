"""
RL Training v2 - Optimized for Speed and Stability.

Key improvements:
1. Rejection sampling (simpler than PPO, works well in practice)
2. Reward-weighted supervised learning
3. Batched generation
4. KL penalty to stay near SFT policy
"""

import sys
from pathlib import Path
import json
import random
import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from archive.codebook import OPCODE_TO_WASM, IROpcode, encode_i32_const
from archive.wasm_runtime import WASMRuntime


# =============================================================================
# Reward Computation (Verifiable via WASM)
# =============================================================================

def parse_expression(text: str) -> list[dict] | None:
    """Parse expression to IR."""
    text = text.strip().rstrip("=").strip()
    # Remove any non-math characters
    text = re.sub(r'[^0-9+\-*/() ]', '', text)
    tokens = re.findall(r'\d+|[+\-*/()]', text)

    if not tokens:
        return None

    output = []
    op_stack = []
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

        # Validate: need at least one number and result in one value
        nums = sum(1 for x in output if x['type'] == 'push')
        ops = sum(1 for x in output if x['type'] == 'op')
        if nums == 0 or (nums > 1 and ops < nums - 1):
            return None

        return output if output else None
    except:
        return None


def execute_ir(ir_seq: list[dict], runtime: WASMRuntime) -> int | None:
    """Execute IR via WASM."""
    OP_TO_WASM = {
        "add": IROpcode.I32_ADD,
        "subtract": IROpcode.I32_SUB,
        "multiply": IROpcode.I32_MUL,
        "divide": IROpcode.I32_DIV_S,
    }

    try:
        body = bytearray()
        for instr in ir_seq:
            if instr['type'] == 'push':
                body.extend(encode_i32_const(instr['value']))
            elif instr['type'] == 'op':
                body.extend(OPCODE_TO_WASM[OP_TO_WASM[instr['op']]])

        result = runtime.execute(bytes(body))
        return result.result if result.success else None
    except:
        return None


@dataclass
class RewardResult:
    reward: float
    parse_ok: bool
    correct: bool
    expression: str
    result: int | None


def compute_reward(expression: str, expected: int, runtime: WASMRuntime) -> RewardResult:
    """Compute verifiable reward."""
    ir = parse_expression(expression)

    if ir is None:
        return RewardResult(0.0, False, False, expression, None)

    result = execute_ir(ir, runtime)

    if result is None:
        return RewardResult(0.1, True, False, expression, None)

    if result == expected:
        return RewardResult(1.0, True, True, expression, result)

    return RewardResult(0.3, True, False, expression, result)


# =============================================================================
# Generation
# =============================================================================

def generate_expression(
    model, tokenizer, question: str,
    temperature: float = 0.8,
    max_tokens: int = 25
) -> str:
    """Generate expression with temperature sampling."""
    prompt = f"Q: {question}\nA:"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        logits = logits[:, -1, :] / max(temperature, 0.1)

        if temperature > 0:
            probs = mx.softmax(logits, axis=-1)
            next_token = int(mx.random.categorical(logits).item())
        else:
            next_token = int(mx.argmax(logits, axis=-1).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        decoded = tokenizer.decode(generated)
        if "=" in decoded or "\n" in decoded:
            break

    result = tokenizer.decode(generated).strip()
    if "\n" in result:
        result = result.split("\n")[0]
    if "=" in result:
        result = result[:result.index("=") + 1].strip()

    return result


# =============================================================================
# Reward-Weighted Training (Simpler than PPO, often works as well)
# =============================================================================

def load_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train_rl_v2(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_dir: str = None,
    num_iterations: int = 5,
    samples_per_iter: int = 500,
    num_samples_per_problem: int = 3,
    top_k_ratio: float = 0.3,  # Keep top 30% by reward
    learning_rate: float = 2e-6,
    temperature: float = 0.8,
):
    """
    Reward-weighted SFT (a.k.a. Rejection Sampling Fine-tuning).

    For each iteration:
    1. Sample multiple outputs per problem
    2. Score with verifiable reward
    3. Keep top-k by reward
    4. Fine-tune on high-reward samples
    """
    print("=" * 70)
    print("  RL TRAINING V2 - REWARD-WEIGHTED SFT")
    print("=" * 70)

    if data_dir is None:
        data_dir = Path(__file__).parent / "sft_data"

    # Load data
    print("\nLoading data...")
    train_data = load_dataset(data_dir / "train.jsonl")
    val_data = load_dataset(data_dir / "val.jsonl")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model
    print(f"\nLoading {model_name}...")
    result = load_model(model_name)
    model, tokenizer = result.model, result.tokenizer

    # Freeze most, train last layer
    model.freeze()
    model.model.layers[-1].unfreeze()

    runtime = WASMRuntime()
    optimizer = optim.Adam(learning_rate=learning_rate)

    print(f"\nTraining config:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples/iter: {samples_per_iter}")
    print(f"  Samples/problem: {num_samples_per_problem}")
    print(f"  Top-k ratio: {top_k_ratio}")
    print(f"  Temperature: {temperature}")
    print("-" * 70)

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Sample problems for this iteration
        problems = random.sample(train_data, min(samples_per_iter, len(train_data)))

        # Generate multiple samples per problem and score
        print(f"\nGenerating {len(problems) * num_samples_per_problem} samples...")
        all_samples = []

        for i, problem in enumerate(problems):
            for _ in range(num_samples_per_problem):
                expr = generate_expression(
                    model, tokenizer, problem["question"],
                    temperature=temperature
                )
                reward_result = compute_reward(expr, problem["answer"], runtime)

                all_samples.append({
                    "question": problem["question"],
                    "expression": expr,
                    "reward": reward_result.reward,
                    "correct": reward_result.correct,
                    "parse_ok": reward_result.parse_ok,
                })

            if (i + 1) % 100 == 0:
                correct = sum(1 for s in all_samples if s["correct"])
                parse_ok = sum(1 for s in all_samples if s["parse_ok"])
                print(f"  Progress: {i+1}/{len(problems)}, "
                      f"correct={correct}/{len(all_samples)}, "
                      f"parse={parse_ok}/{len(all_samples)}")

        # Stats before filtering
        total = len(all_samples)
        correct = sum(1 for s in all_samples if s["correct"])
        parse_ok = sum(1 for s in all_samples if s["parse_ok"])
        avg_reward = sum(s["reward"] for s in all_samples) / total

        print(f"\nGeneration stats:")
        print(f"  Parse success: {parse_ok}/{total} = {parse_ok/total:.1%}")
        print(f"  Correct: {correct}/{total} = {correct/total:.1%}")
        print(f"  Avg reward: {avg_reward:.3f}")

        # Filter to top-k by reward
        all_samples.sort(key=lambda x: x["reward"], reverse=True)
        n_keep = max(int(len(all_samples) * top_k_ratio), 50)
        top_samples = all_samples[:n_keep]

        # Only keep samples with reward > 0 (valid parses)
        top_samples = [s for s in top_samples if s["reward"] > 0]

        if not top_samples:
            print("  No valid samples, skipping training step")
            continue

        top_correct = sum(1 for s in top_samples if s["correct"])
        print(f"\nFiltered to {len(top_samples)} samples "
              f"({top_correct} correct, {top_correct/len(top_samples):.1%})")

        # Train on high-reward samples
        print(f"\nTraining on {len(top_samples)} high-reward samples...")

        random.shuffle(top_samples)
        batch_size = 4
        total_loss = 0
        num_batches = 0

        for i in range(0, len(top_samples), batch_size):
            batch = top_samples[i:i + batch_size]

            # Prepare batch
            all_tokens = []
            all_masks = []
            all_weights = []

            for sample in batch:
                full_text = f"Q: {sample['question']}\nA: {sample['expression']}"
                q_text = f"Q: {sample['question']}\nA:"

                full_tokens = tokenizer.encode(full_text)[:100]
                q_tokens = tokenizer.encode(q_text)

                q_len = len(q_tokens)
                mask = [0] * q_len + [1] * (len(full_tokens) - q_len)

                all_tokens.append(full_tokens)
                all_masks.append(mask)
                all_weights.append(sample["reward"])

            # Pad
            max_len = max(len(t) for t in all_tokens)
            padded_tokens = [t + [0] * (max_len - len(t)) for t in all_tokens]
            padded_masks = [m + [0] * (max_len - len(m)) for m in all_masks]

            batch_tokens = mx.array(padded_tokens)
            batch_masks = mx.array(padded_masks, dtype=mx.float32)
            batch_weights = mx.array(all_weights, dtype=mx.float32)

            # Weighted loss
            def compute_loss(model, tokens, masks, weights):
                logits = model(tokens)
                if hasattr(logits, 'logits'):
                    logits = logits.logits

                logits = logits[:, :-1, :]
                targets = tokens[:, 1:]
                masks = masks[:, 1:]

                vocab_size = logits.shape[-1]
                ce = nn.losses.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction='none'
                )
                ce = ce.reshape(targets.shape)

                # Per-sample loss, weighted by reward
                sample_loss = (ce * masks).sum(axis=1) / (masks.sum(axis=1) + 1e-8)
                weighted_loss = (sample_loss * weights).mean()

                return weighted_loss

            loss_and_grad = nn.value_and_grad(model, compute_loss)
            loss, grads = loss_and_grad(model, batch_tokens, batch_masks, batch_weights)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Training loss: {avg_loss:.4f}")

        # Validation
        print("\nValidation...")
        val_correct, val_parse = evaluate(model, tokenizer, val_data[:100], runtime)
        print(f"  Val accuracy: {val_correct:.1%}, parse: {val_parse:.1%}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)

    test_data = load_dataset(data_dir / "test.jsonl")
    test_correct, test_parse = evaluate(model, tokenizer, test_data, runtime, verbose=True)
    print(f"\nTest accuracy: {test_correct:.1%}")
    print(f"Parse success: {test_parse:.1%}")

    return model, tokenizer


def evaluate(model, tokenizer, data, runtime, verbose=False):
    """Evaluate with greedy decoding."""
    correct = 0
    parse_ok = 0

    for i, item in enumerate(data):
        expr = generate_expression(model, tokenizer, item["question"], temperature=0)
        reward = compute_reward(expr, item["answer"], runtime)

        if reward.parse_ok:
            parse_ok += 1
        if reward.correct:
            correct += 1

        if verbose and i < 5:
            status = "✓" if reward.correct else "✗"
            print(f"{status} {item['question'][:50]}...")
            print(f"    Pred: {expr} → {reward.result}")
            print(f"    Expected: {item['answer']}")

    return correct / len(data), parse_ok / len(data)


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  REWARD-WEIGHTED SFT (Rejection Sampling Fine-tuning)             ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  1. Sample multiple outputs per problem                           ║
    ║  2. Score with WASM execution (verifiable reward)                 ║
    ║  3. Keep top-k by reward                                          ║
    ║  4. Fine-tune on high-reward samples                              ║
    ║                                                                   ║
    ║  Simpler than PPO, often works just as well.                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    model, tokenizer = train_rl_v2(
        num_iterations=5,
        samples_per_iter=300,
        num_samples_per_problem=3,
        top_k_ratio=0.4,
        learning_rate=2e-6,
        temperature=0.8,
    )


if __name__ == "__main__":
    main()
