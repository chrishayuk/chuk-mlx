"""
RL with Verifiable Rewards - REINFORCE with WASM Execution.

The reward is VERIFIABLE:
  reward = 1.0 if WASM_execute(parse(output)) == expected_answer
  reward = 0.3 if parse succeeds but wrong answer
  reward = 0.0 if parse fails

No human labels. No reward model. Just execution correctness.

Policy gradient: ∇J = E[R * ∇log π(a|s)]
"""

import sys
from pathlib import Path
import json
import random
import re

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
# VERIFIABLE REWARD FUNCTION (via WASM)
# =============================================================================

def parse_to_ir(text: str) -> list[dict] | None:
    """Parse expression to IR sequence."""
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
    """Execute IR via WASM - this is ground truth."""
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


def verifiable_reward(expression: str, expected: int, runtime: WASMRuntime) -> float:
    """
    THE VERIFIABLE REWARD FUNCTION.

    This is the key insight: WASM execution IS the reward.
    No human labels needed. No reward model. Just execution.
    """
    ir = parse_to_ir(expression)

    if ir is None:
        return 0.0  # Parse failure

    result = wasm_execute(ir, runtime)

    if result is None:
        return 0.1  # Execution failure (valid parse though)

    if result == expected:
        return 1.0  # CORRECT - verifiably correct!

    return 0.3  # Valid structure, wrong answer


# =============================================================================
# REINFORCE POLICY GRADIENT
# =============================================================================

def sample_with_log_probs(model, tokenizer, prompt: str, temperature: float = 0.8, max_len: int = 25):
    """
    Sample from policy, collecting log probabilities for each token.

    Returns: (generated_text, total_log_prob)
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated_tokens = []
    log_probs = []

    for _ in range(max_len):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        logits = logits[:, -1, :] / temperature

        # Sample from policy
        probs = mx.softmax(logits, axis=-1)
        next_token = int(mx.random.categorical(logits).item())

        # Log probability of chosen action
        log_prob = mx.log(probs[0, next_token] + 1e-10)
        log_probs.append(log_prob)

        generated_tokens.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        # Stop at = or newline
        decoded = tokenizer.decode(generated_tokens)
        if "=" in decoded or "\n" in decoded:
            break

    text = tokenizer.decode(generated_tokens).strip()
    if "=" in text:
        text = text[:text.index("=") + 1]

    # Sum log probs for REINFORCE
    total_log_prob = sum(log_probs) if log_probs else mx.array(0.0)

    return text, total_log_prob


def reinforce_step(model, optimizer, samples: list[tuple], baseline: float):
    """
    REINFORCE policy gradient update.

    samples: list of (log_prob, reward) tuples
    baseline: running average reward for variance reduction

    Policy gradient: ∇J = E[(R - baseline) * ∇log π]
    """
    if not samples:
        return 0.0

    # Compute policy gradient loss
    # Loss = -E[(R - baseline) * log_prob]  (negative because we minimize)
    total_loss = mx.array(0.0)

    for log_prob, reward in samples:
        advantage = reward - baseline
        total_loss = total_loss - log_prob * advantage

    total_loss = total_loss / len(samples)

    return total_loss


# =============================================================================
# TRAINING LOOP
# =============================================================================

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train_rl_verifiable(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_dir: str = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-6,
    temperature: float = 0.8,
    baseline_decay: float = 0.95,
):
    """
    REINFORCE training with verifiable rewards from WASM execution.
    """
    print("=" * 70)
    print("  RL WITH VERIFIABLE REWARDS")
    print("  Reward = WASM_execute(parse(output)) == expected")
    print("=" * 70)

    if data_dir is None:
        data_dir = Path(__file__).parent / "sft_data"

    train_data = load_data(data_dir / "train.jsonl")
    val_data = load_data(data_dir / "val.jsonl")
    print(f"\nData: train={len(train_data)}, val={len(val_data)}")

    print(f"\nLoading {model_name}...")
    result = load_model(model_name)
    model, tokenizer = result.model, result.tokenizer

    # Only train last layer for efficiency
    model.freeze()
    model.model.layers[-1].unfreeze()

    runtime = WASMRuntime()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Running baseline for variance reduction
    baseline = 0.5

    print(f"\nConfig: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}, temp={temperature}")
    print("-" * 70)

    for epoch in range(num_epochs):
        random.shuffle(train_data)

        epoch_rewards = []
        epoch_correct = 0
        epoch_parse_ok = 0
        num_samples = 0

        for batch_idx in range(0, min(len(train_data), 500), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]

            # Collect samples for this batch
            batch_samples = []  # (log_prob, reward)

            for item in batch:
                prompt = f"Q: {item['question']}\nA:"

                # Sample from policy
                expression, log_prob = sample_with_log_probs(
                    model, tokenizer, prompt, temperature
                )

                # GET VERIFIABLE REWARD
                reward = verifiable_reward(expression, item["answer"], runtime)

                batch_samples.append((log_prob, reward))
                epoch_rewards.append(reward)
                num_samples += 1

                if reward == 1.0:
                    epoch_correct += 1
                if reward >= 0.3:
                    epoch_parse_ok += 1

            # Compute REINFORCE loss
            def policy_loss_fn(model):
                # Recompute forward pass for gradients
                loss = mx.array(0.0)
                for i, item in enumerate(batch):
                    prompt = f"Q: {item['question']}\nA:"
                    tokens = tokenizer.encode(prompt)
                    input_ids = mx.array([tokens])

                    # Forward pass
                    for _ in range(15):
                        output = model(input_ids)
                        logits = output.logits if hasattr(output, 'logits') else output
                        logits = logits[:, -1, :] / temperature
                        probs = mx.softmax(logits, axis=-1)

                        # Take greedy action for gradient
                        next_token = mx.argmax(logits, axis=-1)
                        log_prob = mx.log(mx.max(probs) + 1e-10)

                        # Weight by advantage
                        _, reward = batch_samples[i]
                        advantage = reward - baseline
                        loss = loss - log_prob * advantage

                        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

                return loss / len(batch)

            # Update
            loss_and_grad = nn.value_and_grad(model, policy_loss_fn)
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            # Update baseline
            batch_avg_reward = sum(r for _, r in batch_samples) / len(batch_samples)
            baseline = baseline_decay * baseline + (1 - baseline_decay) * batch_avg_reward

            if (batch_idx // batch_size + 1) % 10 == 0:
                avg_r = sum(epoch_rewards) / len(epoch_rewards)
                acc = epoch_correct / num_samples
                parse = epoch_parse_ok / num_samples
                print(f"  Batch {batch_idx//batch_size + 1}: "
                      f"reward={avg_r:.3f}, acc={acc:.1%}, parse={parse:.1%}")

        # Epoch summary
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        accuracy = epoch_correct / num_samples
        parse_rate = epoch_parse_ok / num_samples

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Avg reward: {avg_reward:.3f}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Parse rate: {parse_rate:.1%}")

        # Validation
        val_acc, val_parse = evaluate(model, tokenizer, val_data[:100], runtime)
        print(f"  Val: acc={val_acc:.1%}, parse={val_parse:.1%}")

    # Final test
    print("\n" + "=" * 70)
    print("  FINAL TEST")
    print("=" * 70)

    test_data = load_data(data_dir / "test.jsonl")
    test_acc, test_parse = evaluate(model, tokenizer, test_data, runtime, verbose=True)
    print(f"\nTest accuracy: {test_acc:.1%}")
    print(f"Parse rate: {test_parse:.1%}")

    return model, tokenizer


def evaluate(model, tokenizer, data, runtime, verbose=False):
    """Evaluate with greedy decoding."""
    correct = 0
    parse_ok = 0

    for i, item in enumerate(data):
        prompt = f"Q: {item['question']}\nA:"
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(25):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
            if "=" in tokenizer.decode(generated):
                break

        expr = tokenizer.decode(generated).strip()
        if "=" in expr:
            expr = expr[:expr.index("=") + 1]

        reward = verifiable_reward(expr, item["answer"], runtime)

        if reward >= 0.3:
            parse_ok += 1
        if reward == 1.0:
            correct += 1
            if verbose and correct <= 3:
                print(f"✓ Q: {item['question'][:50]}...")
                print(f"  A: {expr}")

    return correct / len(data), parse_ok / len(data)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   RL WITH VERIFIABLE REWARDS                                      ║
    ║                                                                   ║
    ║   reward = WASM_execute(parse(model_output)) == expected          ║
    ║                                                                   ║
    ║   • No human labels                                               ║
    ║   • No reward model                                               ║
    ║   • Just execution correctness                                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    train_rl_verifiable(
        num_epochs=3,
        batch_size=8,
        learning_rate=1e-6,
        temperature=0.8,
    )
