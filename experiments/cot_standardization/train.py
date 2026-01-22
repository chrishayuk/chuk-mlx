#!/usr/bin/env python3
"""
Train Standardized CoT for Virtual Expert Routing.

Unified CoT format that works across ALL expert types:

    <expert>: <spec> -> <result>

Examples:
    multiply: 256 * 4 -> 1024
    add: 100 + 200 -> 300
    word_problem: eggs:16 | -3 -4 *2 -> 18
    schedule: Alice:2hr,Bob:1hr | no_overlap -> Alice:9-11,Bob:11-12
    time: Asia/Tokyo -> 14:30 JST
    chat: -> [passthrough]

The model learns to:
1. Identify the expert from the query
2. Extract the spec in the expert's format
3. The hidden states at "<expert>:" trigger MoE routing
4. Expert executes the spec and produces result

Training:
1. SFT: Train on query -> expert: spec -> result format
2. RL: Verified rewards (expert actually executes spec)

Usage:
    python train_cot_router.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python train_cot_router.py --model HuggingFaceTB/SmolLM-1.7B
"""

from __future__ import annotations

import argparse
import functools
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model


# =============================================================================
# UNIFIED COT FORMAT
# =============================================================================
#
# Format: <expert>: <spec> -> <result>
#
# Expert Types:
#   - multiply, add, subtract, divide: Simple arithmetic
#   - word_problem: GSM8K-style word problems
#   - schedule: CSP scheduling
#   - time: Timezone queries
#   - chat: Passthrough to base model
#
# The expert token triggers MoE routing.
# The spec is parsed by the expert.
# The result is produced by expert execution.


@dataclass
class Example:
    """Training example."""
    query: str
    expert: str           # Expert token: multiply, add, time, schedule, chat, etc.
    spec: str             # Specification in expert's format
    result: str           # Expected result
    expert_class: str     # Broad class: math, word_problem, csp, time, none
    ground_truth: Any = None  # For verification


# =============================================================================
# EXPERT EXECUTION (verified rewards)
# =============================================================================

def execute_math(expr: str) -> tuple[bool, Any]:
    """Execute math expression."""
    try:
        import ast
        import operator

        OPS = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return OPS[type(node.op)](_eval(node.operand))
            raise ValueError(f"Unsupported: {type(node)}")

        tree = ast.parse(expr, mode="eval")
        return True, _eval(tree)
    except:
        return False, None


def execute_word_problem(spec: str) -> tuple[bool, Any]:
    """
    Execute word problem spec.

    Format: entity:value | op1 op2 op3 ...
    Example: eggs:16 | -3 -4 *2
    """
    try:
        if "|" not in spec:
            return False, None

        entity_part, ops_part = spec.split("|", 1)

        # Parse entity: "eggs:16" -> 16
        _, value_str = entity_part.strip().split(":")
        value = float(value_str)

        # Parse operations: "-3 -4 *2" -> [(-,3), (-,4), (*,2)]
        ops = re.findall(r'([+\-*/])(\d+\.?\d*)', ops_part)

        for op, num in ops:
            num = float(num)
            if op == '+':
                value += num
            elif op == '-':
                value -= num
            elif op == '*':
                value *= num
            elif op == '/':
                value /= num

        # Return as int if whole number
        result = int(value) if value == int(value) else value
        return True, result
    except:
        return False, None


def execute_schedule(spec: str) -> tuple[bool, str]:
    """
    Execute schedule spec (simplified).

    Format: Task1:dur,Task2:dur | constraint1,constraint2
    Example: Alice:2hr,Bob:1hr | no_overlap

    For now, just validate format and return a simple schedule.
    """
    try:
        if "|" not in spec:
            return False, ""

        tasks_part, constraints_part = spec.split("|", 1)
        tasks = []

        for task_str in tasks_part.strip().split(","):
            name, dur = task_str.strip().split(":")
            tasks.append((name.strip(), dur.strip()))

        # Simple greedy schedule
        time = 9  # Start at 9am
        schedule = []
        for name, dur in tasks:
            # Parse duration
            dur_hours = float(re.search(r'(\d+\.?\d*)', dur).group(1))
            end_time = time + dur_hours
            schedule.append(f"{name}:{int(time)}-{int(end_time)}")
            time = end_time

        return True, ",".join(schedule)
    except:
        return False, ""


def execute_time(tz: str) -> tuple[bool, str]:
    """Execute time query."""
    try:
        from datetime import datetime
        import zoneinfo

        TZ_MAP = {
            "tokyo": "Asia/Tokyo", "asia/tokyo": "Asia/Tokyo",
            "london": "Europe/London", "europe/london": "Europe/London",
            "new york": "America/New_York", "america/new_york": "America/New_York",
            "paris": "Europe/Paris", "europe/paris": "Europe/Paris",
            "sydney": "Australia/Sydney", "australia/sydney": "Australia/Sydney",
            "berlin": "Europe/Berlin", "europe/berlin": "Europe/Berlin",
            "utc": "UTC",
        }

        tz_clean = tz.lower().strip()
        tz_name = TZ_MAP.get(tz_clean, tz)
        zone = zoneinfo.ZoneInfo(tz_name)
        now = datetime.now(zone)
        return True, now.strftime("%H:%M %Z")
    except:
        return False, ""


def compute_reward(output: str, example: Example) -> tuple[float, str]:
    """
    Compute verified reward.

    Parses: <expert>: <spec> -> <result>
    """
    # Parse output
    match = re.match(r'(\w+):\s*(.+?)\s*->\s*(.+)', output.strip())
    if not match:
        # Try simpler format: expert: spec
        match2 = re.match(r'(\w+):\s*(.+)', output.strip())
        if match2:
            got_expert = match2.group(1).lower()
            got_spec = match2.group(2).strip()
            got_result = ""
        else:
            return 0.0, "parse_fail"
    else:
        got_expert = match.group(1).lower()
        got_spec = match.group(2).strip()
        got_result = match.group(3).strip()

    # Check expert match
    expected_experts = {
        "math": ["multiply", "add", "subtract", "divide", "math"],
        "word_problem": ["word_problem", "wp"],
        "csp": ["schedule", "csp"],
        "time": ["time", "clock"],
        "none": ["chat", "none", "passthrough"],
    }

    correct_experts = expected_experts.get(example.expert_class, [example.expert])
    if got_expert not in correct_experts:
        return 0.3, f"wrong_expert:{got_expert}!={example.expert}"

    # Verify execution based on expert class
    if example.expert_class == "math":
        success, result = execute_math(got_spec)
        if not success:
            return 0.5, f"math_exec_fail:{got_spec}"
        if example.ground_truth is not None:
            if abs(result - example.ground_truth) < 0.01:
                return 1.0, "correct"
            return 0.7, f"math_wrong:{result}!={example.ground_truth}"
        return 0.9, "math_format_ok"

    elif example.expert_class == "word_problem":
        success, result = execute_word_problem(got_spec)
        if not success:
            return 0.5, f"wp_exec_fail:{got_spec}"
        if example.ground_truth is not None:
            if abs(result - example.ground_truth) < 0.01:
                return 1.0, "correct"
            return 0.7, f"wp_wrong:{result}!={example.ground_truth}"
        return 0.9, "wp_format_ok"

    elif example.expert_class == "csp":
        success, result = execute_schedule(got_spec)
        if not success:
            return 0.5, f"csp_exec_fail:{got_spec}"
        return 1.0, "correct"

    elif example.expert_class == "time":
        success, result = execute_time(got_spec)
        if not success:
            return 0.5, f"time_exec_fail:{got_spec}"
        return 1.0, "correct"

    elif example.expert_class == "none":
        # Chat/passthrough is always correct if format is right
        return 1.0, "correct"

    return 0.5, "unknown_class"


# =============================================================================
# TRAINING DATA
# =============================================================================

# Simple math
MATH_DATA = [
    ("256 * 4", "multiply", "256 * 4", 1024),
    ("127 * 89", "multiply", "127 * 89", 11303),
    ("99 * 99", "multiply", "99 * 99", 9801),
    ("15 * 8", "multiply", "15 * 8", 120),
    ("456 + 789", "add", "456 + 789", 1245),
    ("100 + 200", "add", "100 + 200", 300),
    ("1000 - 250", "subtract", "1000 - 250", 750),
    ("50 - 17", "subtract", "50 - 17", 33),
    ("144 / 12", "divide", "144 / 12", 12),
    ("72 / 9", "divide", "72 / 9", 8),
]

MATH_QUERIES = [
    "{expr}",
    "What is {expr}?",
    "Calculate {expr}",
    "{expr} = ?",
    # With extra instructions
    "What is {expr}? Answer briefly.",
    "Calculate {expr}, show work",
]

# Word problems
WORD_PROBLEMS = [
    ("Janet has 16 eggs. She eats 3 and bakes 4. She sells the rest for $2 each. How much does she make?",
     "word_problem", "eggs:16 | -3 -4 *2", 18),
    ("Sam has 10 marbles. He loses 3, then finds 5. How many does he have?",
     "word_problem", "marbles:10 | -3 +5", 12),
    ("6 bags have 4 oranges each. How many oranges total?",
     "word_problem", "oranges:6 | *4", 24),
    ("A shirt costs $40. It's 25% off. What's the sale price?",
     "word_problem", "price:40 | *0.75", 30),
    ("12 cookies shared among 4 kids. How many each?",
     "word_problem", "cookies:12 | /4", 3),
]

# CSP scheduling
CSP_PROBLEMS = [
    ("Schedule Alice (2hr) and Bob (1hr) with no overlap",
     "schedule", "Alice:2hr,Bob:1hr | no_overlap", "Alice:9-11,Bob:11-12"),
    ("Schedule gym (1hr), lunch (1.5hr), meeting (2hr)",
     "schedule", "gym:1hr,lunch:1.5hr,meeting:2hr | sequential", "gym:9-10,lunch:10-12,meeting:12-14"),
]

# Time queries
TIME_DATA = [
    ("Tokyo", "Asia/Tokyo"),
    ("London", "Europe/London"),
    ("New York", "America/New_York"),
    ("Paris", "Europe/Paris"),
    ("Sydney", "Australia/Sydney"),
]

TIME_QUERIES = [
    "What time is it in {loc}?",
    "Current time in {loc}",
    "Time in {loc}",
    # With extra instructions
    "What time in {loc}? Be brief",
    "Time in {loc}, answer in french",
]

# Chat (passthrough)
CHAT_QUERIES = [
    "Tell me a joke",
    "What is the capital of France?",
    "Write a poem",
    "Hello, how are you?",
    "Explain quantum physics",
    # With extra instructions
    "Tell me a joke, make it funny",
    "What's the capital of France? Be brief",
]


def generate_examples(n_per_class: int = 50) -> list[Example]:
    """Generate balanced training examples."""
    examples = []

    # Simple math
    for _ in range(n_per_class):
        expr, expert, spec, result = random.choice(MATH_DATA)
        query_tpl = random.choice(MATH_QUERIES)
        query = query_tpl.format(expr=expr)

        examples.append(Example(
            query=query,
            expert=expert,
            spec=spec,
            result=str(result),
            expert_class="math",
            ground_truth=result,
        ))

    # Word problems
    for _ in range(n_per_class):
        query, expert, spec, result = random.choice(WORD_PROBLEMS)
        examples.append(Example(
            query=query,
            expert=expert,
            spec=spec,
            result=str(result),
            expert_class="word_problem",
            ground_truth=result,
        ))

    # CSP
    for _ in range(n_per_class // 2):
        query, expert, spec, result = random.choice(CSP_PROBLEMS)
        examples.append(Example(
            query=query,
            expert=expert,
            spec=spec,
            result=result,
            expert_class="csp",
        ))

    # Time
    for _ in range(n_per_class):
        loc, tz = random.choice(TIME_DATA)
        query_tpl = random.choice(TIME_QUERIES)
        query = query_tpl.format(loc=loc)
        success, time_str = execute_time(tz)

        examples.append(Example(
            query=query,
            expert="time",
            spec=tz,
            result=time_str if success else "12:00 UTC",
            expert_class="time",
        ))

    # Chat
    for _ in range(n_per_class):
        query = random.choice(CHAT_QUERIES)
        examples.append(Example(
            query=query,
            expert="chat",
            spec="",
            result="[response]",
            expert_class="none",
        ))

    random.shuffle(examples)
    return examples


# =============================================================================
# FORMAT - Using TinyLlama's native chat format
# =============================================================================

SYSTEM_PROMPT = """You are a routing assistant. Convert user queries to expert format.
Format: EXPERT: SPEC -> RESULT
Experts: multiply, add, subtract, divide, word_problem, schedule, time, chat
Examples:
- "256 * 4" -> multiply: 256 * 4 -> 1024
- "Time in Tokyo" -> time: Asia/Tokyo -> 14:30
- "Hello" -> chat: -> [response]"""


def format_target(example: Example) -> str:
    """Format as: expert: spec -> result"""
    if example.spec:
        return f"{example.expert}: {example.spec} -> {example.result}"
    else:
        return f"{example.expert}: -> {example.result}"


def format_chat_prompt(query: str) -> str:
    """Format using TinyLlama's chat template."""
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"


def format_full_prompt(query: str) -> str:
    """Full prompt for generation."""
    return format_chat_prompt(query)


def format_full_target(example: Example) -> str:
    """Full target for training."""
    return format_chat_prompt(example.query) + format_target(example) + "</s>"


# =============================================================================
# GENERATION
# =============================================================================

def generate(model, tokenizer, prompt: str, max_tokens: int = 60, greedy: bool = True, temp: float = 0.7) -> str:
    """Generate completion."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if greedy:
            next_token = int(mx.argmax(logits).item())
        else:
            next_token = int(mx.random.categorical(logits / temp).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        text = tokenizer.decode(generated)
        # Stop at EOS marker or newline
        if '</s>' in text or '\n' in text:
            break
        # Also stop if we have complete format: expert: spec -> result
        if '->' in text and len(text.split('->')) >= 2:
            result_part = text.split('->')[-1].strip()
            if len(result_part) > 2:  # Has some result
                break

    output = tokenizer.decode(generated).strip()
    # Clean up any EOS tokens
    output = output.replace('</s>', '').strip()
    return output.split('\n')[0]


# =============================================================================
# TRAINING
# =============================================================================

def sft_step(model, tokenizer, batch: list[Example], optimizer, max_len: int = 384):
    """One SFT step."""

    def loss_fn(model, tokens_list, masks_list):
        total = mx.array(0.0)
        for tokens, mask in zip(tokens_list, masks_list):
            toks = mx.array([tokens])
            logits = model(toks)
            if hasattr(logits, "logits"):
                logits = logits.logits

            logits = logits[:, :-1, :]
            targets = toks[:, 1:]
            m = mx.array([mask[1:]], dtype=mx.float32)

            ce = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none"
            ).reshape(targets.shape)

            total = total + (ce * m).sum() / (m.sum() + 1e-8)
        return total / len(tokens_list)

    tokens_list, masks_list = [], []
    for ex in batch:
        full = format_full_target(ex)
        prompt = format_full_prompt(ex.query)

        full_toks = tokenizer.encode(full)[:max_len]
        prompt_len = len(tokenizer.encode(prompt))

        mask = [0] * prompt_len + [1] * (len(full_toks) - prompt_len)
        tokens_list.append(full_toks)
        masks_list.append(mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens_list, masks_list)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, tokenizer, data: list[Example], show: int = 0) -> dict:
    """Evaluate examples."""
    stats = {"total": 0, "correct": 0, "by_class": {}}

    for ex in data:
        prompt = format_full_prompt(ex.query)
        output = generate(model, tokenizer, prompt, greedy=True)
        reward, reason = compute_reward(output, ex)

        stats["total"] += 1
        if reward >= 0.7:
            stats["correct"] += 1

        # Track by class
        if ex.expert_class not in stats["by_class"]:
            stats["by_class"][ex.expert_class] = {"total": 0, "correct": 0}
        stats["by_class"][ex.expert_class]["total"] += 1
        if reward >= 0.7:
            stats["by_class"][ex.expert_class]["correct"] += 1

        if show > 0 and stats["total"] <= show:
            mark = "✓" if reward >= 0.7 else "✗"
            q = ex.query[:40] + "..." if len(ex.query) > 40 else ex.query
            print(f"  {mark} [{ex.expert_class:12}] {q}")
            print(f"      Got: {output[:50]}")
            print(f"      Expected: {ex.expert}: {ex.spec[:30]}...")

    return stats


def rl_step(model, tokenizer, data: list[Example], optimizer, baseline: float, batch_size: int = 8, temp: float = 0.7):
    """
    One REINFORCE step with proper policy gradient.

    Key: We need to compute log_prob of the SAMPLED tokens, not greedy.
    """
    batch = random.sample(data, min(batch_size, len(data)))

    # Sample outputs and collect token sequences
    samples = []
    for ex in batch:
        prompt = format_full_prompt(ex.query)
        prompt_tokens = tokenizer.encode(prompt)

        # Generate with sampling, keeping track of chosen tokens
        input_ids = mx.array([prompt_tokens])
        generated_tokens = []

        for _ in range(40):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            logits = logits[:, -1, :] / temp

            # Sample from distribution
            next_token = int(mx.random.categorical(logits).item())
            generated_tokens.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            text = tokenizer.decode(generated_tokens)
            if '</s>' in text or '\n' in text:
                break
            if '->' in text and len(text.split('->')) >= 2:
                if len(text.split('->')[-1].strip()) > 2:
                    break

        output_text = tokenizer.decode(generated_tokens).replace('</s>', '').strip().split('\n')[0]
        reward, reason = compute_reward(output_text, ex)
        samples.append((ex, prompt_tokens, generated_tokens, reward, reason))

    avg_reward = sum(s[3] for s in samples) / len(samples)

    # Compute policy gradient loss
    def loss_fn(model):
        total = mx.array(0.0)
        for ex, prompt_tokens, gen_tokens, reward, _ in samples:
            if not gen_tokens:
                continue

            # Full sequence: prompt + generated
            full_tokens = prompt_tokens + gen_tokens
            input_ids = mx.array([full_tokens[:-1]])  # All but last
            targets = mx.array([full_tokens[1:]])      # All but first

            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            # Only compute loss on generated tokens (not prompt)
            gen_start = len(prompt_tokens) - 1
            gen_logits = logits[:, gen_start:, :]
            gen_targets = targets[:, gen_start:]

            # Log probability of chosen tokens (log_softmax = x - logsumexp(x))
            log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)
            chosen_log_probs = mx.take_along_axis(
                log_probs,
                gen_targets[:, :, None],
                axis=-1
            ).squeeze(-1)

            # Sum log probs for sequence
            seq_log_prob = chosen_log_probs.sum()

            # REINFORCE: -log_prob * advantage
            advantage = reward - baseline
            total = total - seq_log_prob * advantage

        return total / max(len(samples), 1)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    new_baseline = 0.9 * baseline + 0.1 * avg_reward
    correct = sum(1 for s in samples if s[3] >= 0.7)

    return avg_reward, new_baseline, correct


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft-epochs", type=int, default=10)
    parser.add_argument("--rl-iters", type=int, default=20)
    parser.add_argument("--examples", type=int, default=250)
    parser.add_argument("--minimal-sft", action="store_true",
                        help="Use 1 epoch SFT then focus on RL (recommended)")
    args = parser.parse_args()

    # Minimal SFT mode: 1 epoch SFT, more RL
    if args.minimal_sft:
        args.sft_epochs = 1
        if args.rl_iters < 20:
            args.rl_iters = 20

    print("=" * 70)
    print("  STANDARDIZED COT TRAINING FOR VIRTUAL EXPERT ROUTING")
    print("  Format: <expert>: <spec> -> <result>")
    print("=" * 70)
    print("\nExpert types:")
    print("  multiply/add/subtract/divide  → Simple math")
    print("  word_problem                  → GSM8K-style word problems")
    print("  schedule                      → CSP scheduling")
    print("  time                          → Timezone queries")
    print("  chat                          → Passthrough")

    # Data
    print(f"\nGenerating {args.examples} examples...")
    train_data = generate_examples(args.examples // 5)
    eval_data = generate_examples(25)

    print(f"Training: {len(train_data)} examples")
    for cls in ["math", "word_problem", "csp", "time", "none"]:
        count = sum(1 for e in train_data if e.expert_class == cls)
        print(f"  {cls}: {count}")

    # Model
    print(f"\nLoading {args.model}...")
    result = load_model(args.model)
    model, tokenizer = result.model, result.tokenizer

    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE (before training)")
    print("=" * 70)
    model.freeze()
    stats = evaluate(model, tokenizer, eval_data[:10], show=5)
    print(f"\nBaseline: {stats['correct']}/{stats['total']} correct")

    # Unfreeze more layers (6 instead of 3 for better learning)
    for layer in model.model.layers[-6:]:
        layer.unfreeze()
    model.lm_head.unfreeze()

    # SFT
    print("\n" + "=" * 70)
    print(f"SFT ({args.sft_epochs} epochs)")
    print("=" * 70)

    sft_opt = optim.Adam(learning_rate=2e-5)
    for epoch in range(args.sft_epochs):
        random.shuffle(train_data)
        losses = []
        for i in range(0, len(train_data), 4):
            loss = sft_step(model, tokenizer, train_data[i:i+4], sft_opt)
            losses.append(loss)

        if (epoch + 1) % 2 == 0:
            stats = evaluate(model, tokenizer, eval_data)
            acc = stats["correct"] / stats["total"]
            print(f"  Epoch {epoch+1}: loss={sum(losses)/len(losses):.4f} acc={acc:.0%}")

    print("\nAfter SFT:")
    evaluate(model, tokenizer, eval_data[:5], show=5)

    # RL
    print("\n" + "=" * 70)
    print(f"RL ({args.rl_iters} iterations)")
    print("=" * 70)

    rl_opt = optim.Adam(learning_rate=5e-7)
    baseline = 0.5

    for i in range(args.rl_iters):
        avg_r, baseline, correct = rl_step(model, tokenizer, train_data, rl_opt, baseline)
        if (i + 1) % 5 == 0:
            stats = evaluate(model, tokenizer, eval_data)
            print(f"  Iter {i+1}: reward={avg_r:.2f} batch={correct}/8 eval={stats['correct']}/{stats['total']}")

    # Final
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final = evaluate(model, tokenizer, eval_data, show=5)
    print(f"\nOverall: {final['correct']}/{final['total']} ({final['correct']/final['total']:.0%})")
    print("\nBy expert class:")
    for cls, s in final["by_class"].items():
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"  {cls:15} {acc:.0%} ({s['correct']}/{s['total']})")

    # Robustness test
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST")
    print("=" * 70)

    test_cases = [
        ("256 * 4", "math"),
        ("What is 256 * 4? Answer briefly.", "math"),
        ("Janet has 16 eggs, eats 3, bakes 4, sells rest for $2. How much?", "word_problem"),
        ("Schedule Alice (2hr) and Bob (1hr), no overlap", "csp"),
        ("Current time in Tokyo", "time"),
        ("Time in London, answer in french", "time"),
        ("Tell me a joke", "none"),
        ("What's the capital of France? Be brief", "none"),
    ]

    print("\nQuery → Expert detection:")
    for query, expected_class in test_cases:
        prompt = format_full_prompt(query)
        output = generate(model, tokenizer, prompt, greedy=True)

        # Extract expert
        match = re.match(r'(\w+):', output)
        got_expert = match.group(1).lower() if match else "?"

        # Map to class
        expert_to_class = {
            "multiply": "math", "add": "math", "subtract": "math", "divide": "math",
            "word_problem": "word_problem", "wp": "word_problem",
            "schedule": "csp", "csp": "csp",
            "time": "time",
            "chat": "none", "none": "none",
        }
        got_class = expert_to_class.get(got_expert, "?")

        mark = "✓" if got_class == expected_class else "✗"
        print(f"  {mark} [{expected_class:12}] {query[:40]:40} → {got_expert}")


if __name__ == "__main__":
    main()
