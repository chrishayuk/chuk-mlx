#!/usr/bin/env python3
"""
GSM-8K Training with YAML Trace Format and Verifiable Rewards.

Fine-tunes a model to emit symbolic YAML traces for GSM-8K math problems.
Uses chuk-virtual-expert TraceVerifier for execution and graduated rewards.

Reward structure (Rogue-1):
- 1.0: Correct answer (trace valid + answer matches)
- 0.7: Valid trace, wrong answer
- 0.5: Valid trace structure, execution error
- 0.3: Parsed YAML, wrong expert
- 0.0: Parse failure

Usage:
    python train_gsm8k_yaml.py
    python train_gsm8k_yaml.py --minimal-sft
    python train_gsm8k_yaml.py --save-checkpoint ckpt/gsm8k
    python train_gsm8k_yaml.py --load-checkpoint ckpt/gsm8k --eval-only --use-hf
    python train_gsm8k_yaml.py --use-hf  # Load from HuggingFace
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from chuk_virtual_expert import ExpertRegistry, TraceVerifier
from chuk_virtual_expert_arithmetic import (
    ArithmeticExpert,
    ComparisonExpert,
    EntityTrackExpert,
    PercentageExpert,
    RateEquationExpert,
)

from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import (
    get_sample_problems,
    GSM8KProblem,
)


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(model, path: str):
    """Save model weights to directory."""
    from pathlib import Path
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    mx.savez(str(ckpt_dir / "weights.npz"), **weights)
    print(f"Checkpoint saved: {ckpt_dir} ({len(weights)} arrays)")


def load_checkpoint(model, path: str):
    """Load model weights from directory."""
    from pathlib import Path
    ckpt_path = Path(path) / "weights.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    weights = dict(mx.load(str(ckpt_path)))
    # Apply weights to model
    model_params = dict(mlx.utils.tree_flatten(model.parameters()))
    updated = 0
    for key, value in weights.items():
        if key in model_params:
            # Navigate the model tree to set the parameter
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            updated += 1

    mx.eval(model.parameters())
    print(f"Checkpoint loaded: {path} ({updated} arrays updated)")


# =============================================================================
# VERIFIER SETUP
# =============================================================================

def build_registry() -> ExpertRegistry:
    """Build registry with all arithmetic experts."""
    registry = ExpertRegistry()
    registry.register(EntityTrackExpert())
    registry.register(ArithmeticExpert())
    registry.register(PercentageExpert())
    registry.register(RateEquationExpert())
    registry.register(ComparisonExpert())
    return registry


_registry = build_registry()
_verifier = TraceVerifier(_registry)


def verify_yaml_output(yaml_str: str, expected_answer=None, tolerance: float = 0.01) -> dict:
    """Verify YAML trace output."""
    result = asyncio.run(_verifier.verify(yaml_str, expected_answer, tolerance=tolerance))
    return {
        "parsed": result.parsed,
        "expert": result.expert,
        "trace_valid": result.trace_valid,
        "trace_error": result.trace_error,
        "computed_answer": result.computed_answer,
        "answer_correct": result.answer_correct,
        "final_state": result.final_state,
    }


# =============================================================================
# REWARD
# =============================================================================

def compute_reward(output: str, expected_answer: float | int, expected_expert: str = None) -> tuple[float, str]:
    """
    Compute graduated reward from model output.

    Returns:
        (reward, reason)
    """
    # Extract YAML from output
    yaml_str = extract_yaml(output)
    if not yaml_str:
        return 0.0, "parse_fail"

    # Verify
    result = verify_yaml_output(yaml_str, expected_answer=expected_answer)

    if not result["parsed"]:
        return 0.0, "parse_fail"

    # Check for "none" expert (passthrough)
    if result["expert"] == "none":
        if expected_expert == "none":
            return 1.0, "correct_passthrough"
        return 0.0, "wrong_passthrough"

    if expected_expert and result["expert"] != expected_expert:
        return 0.3, f"wrong_expert:{result['expert']}"

    if not result["trace_valid"]:
        return 0.5, f"invalid_trace:{result['trace_error']}"

    if not result["answer_correct"]:
        return 0.7, f"wrong_answer:{result['computed_answer']}"

    return 1.0, "correct"


def extract_yaml(text: str) -> str | None:
    """Extract YAML content from model output."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```yaml"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if "```" in text:
        text = text.split("```")[0]

    text = text.strip()
    if not text:
        return None

    # Validate it looks like YAML
    try:
        parsed = yaml.safe_load(text)
        # Single-expert: dict with "expert" key
        if isinstance(parsed, dict) and "expert" in parsed:
            return text
        # Composed: list of sub-traces, each with "expert" key
        if isinstance(parsed, list) and all(
            isinstance(sub, dict) and "expert" in sub for sub in parsed
        ):
            return text
    except yaml.YAMLError:
        pass

    return None


# =============================================================================
# DATA
# =============================================================================

def load_training_data(n: int = 250, include_composition: bool = True) -> list[dict]:
    """Generate YAML trace training examples.

    Args:
        n: Number of training examples to generate.
        include_composition: Whether to include multi-expert composition examples.
    """
    from chuk_virtual_expert_arithmetic.generators import TraceGenerator
    gen = TraceGenerator()
    examples = gen.generate_balanced(n, include_composition=include_composition)

    result = []
    for ex in examples:
        if isinstance(ex, dict):
            # Dict format (composition or simplified patterns from composition.py)
            result.append(ex)
        else:
            # Single-expert TraceExample — serialize steps
            trace = [
                {k: v for k, v in step.model_dump(mode="json").items() if v is not None}
                for step in ex.trace
            ]
            result.append({
                "expert": ex.expert,
                "query": ex.query,
                "trace": trace,
                "answer": ex.answer,
            })

    random.shuffle(result)
    return result


def gsm8k_to_examples(problems: list[GSM8KProblem]) -> list[dict]:
    """Convert GSM8K problems to evaluation format."""
    return [
        {"query": p.question, "answer": p.answer, "expert": None}
        for p in problems
    ]


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage"""


def format_chat_prompt(question: str) -> str:
    """Format using TinyLlama's chat template."""
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n```yaml\n"


def _format_trace_step(step: dict) -> str:
    """Format a single trace step in consistent flow style."""
    # Always use flow style for trace steps to avoid mixed formatting
    parts = []
    for k, v in step.items():
        if isinstance(v, list):
            # Format arrays inline: [a, b, c]
            parts.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        elif isinstance(v, str):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


def _format_trace(expert: str, trace: list[dict]) -> str:
    """Format a single expert trace with consistent styling."""
    lines = [f"expert: {expert}", "trace:"]
    for step in trace:
        lines.append(f"- {_format_trace_step(step)}")
    return "\n".join(lines)


def format_target(example: dict) -> str:
    """Format target as YAML trace (single-expert dict or composed list)."""
    if example.get("composed"):
        # Composed: list of sub-traces
        parts = []
        for sub in example["steps"]:
            sub_yaml = _format_trace(sub["expert"], sub["trace"])
            # Indent all lines and prefix first with "- "
            sub_lines = sub_yaml.split("\n")
            parts.append("- " + sub_lines[0])
            for line in sub_lines[1:]:
                parts.append("  " + line)
        yaml_str = "\n".join(parts) + "\n"
    else:
        # Single expert
        yaml_str = _format_trace(example["expert"], example["trace"]) + "\n"
    return yaml_str + "```"


def format_full_target(example: dict) -> str:
    """Full training target (prompt + answer)."""
    question = example["query"]
    return format_chat_prompt(question) + format_target(example) + "</s>"


# =============================================================================
# GENERATION
# =============================================================================

def generate(
    model, tokenizer, prompt: str, max_tokens: int = 250, greedy: bool = True, temp: float = 0.7
) -> str:
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
        if "</s>" in text:
            break
        # Stop at end of YAML block
        if "```" in text and text.count("```") >= 1 and not text.strip().endswith("```yaml"):
            break

    output = tokenizer.decode(generated).strip()
    output = output.replace("</s>", "").strip()
    return output


# =============================================================================
# TRAINING
# =============================================================================

def sft_step(model, tokenizer, batch: list[dict], optimizer, max_len: int = 1024):
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
                reduction="none",
            ).reshape(targets.shape)

            total = total + (ce * m).sum() / (m.sum() + 1e-8)
        return total / len(tokens_list)

    tokens_list, masks_list = [], []
    for ex in batch:
        full = format_full_target(ex)
        question = ex["query"]
        prompt = format_chat_prompt(question)

        full_toks = tokenizer.encode(full)[:max_len]
        prompt_len = min(len(tokenizer.encode(prompt)), len(full_toks))

        mask = [0] * prompt_len + [1] * (len(full_toks) - prompt_len)
        tokens_list.append(full_toks)
        masks_list.append(mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens_list, masks_list)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(
    model, tokenizer, data: list[dict],
    show: int = 0, quiet: bool = False, show_wrong_traces: bool = False,
) -> dict:
    """Evaluate model on examples.

    Args:
        show: Number of examples to print (truncated output)
        quiet: Suppress progress output
        show_wrong_traces: Print full trace for wrong answers (for debugging)
    """
    stats = {
        "total": 0, "correct": 0, "parsed": 0, "valid_trace": 0,
        "wrong_answer": 0, "wrong_expert": 0, "by_expert": {},
    }

    for idx, ex in enumerate(data):
        if not quiet and len(data) > 10 and (idx + 1) % 5 == 0:
            print(f"    evaluating {idx+1}/{len(data)}...", end="\r", flush=True)

        question = ex["query"]
        prompt = format_chat_prompt(question)
        output = generate(model, tokenizer, prompt, greedy=True)
        reward, reason = compute_reward(
            output,
            ex.get("answer", 0),
            expected_expert=ex.get("expert"),
        )

        stats["total"] += 1
        if reward >= 1.0:
            stats["correct"] += 1
        if reward > 0.0:
            stats["parsed"] += 1
        if reward >= 0.5:
            stats["valid_trace"] += 1
        if reward == 0.7:
            stats["wrong_answer"] += 1
        if reward == 0.3:
            stats["wrong_expert"] += 1

        # Track by expert
        expert = "composition" if ex.get("composed") else ex.get("expert", "unknown")
        if expert not in stats["by_expert"]:
            stats["by_expert"][expert] = {"total": 0, "correct": 0}
        stats["by_expert"][expert]["total"] += 1
        if reward >= 1.0:
            stats["by_expert"][expert]["correct"] += 1

        if show > 0 and stats["total"] <= show:
            mark = "+" if reward >= 1.0 else ("~" if reward >= 0.5 else "-")
            q = question[:55] + "..." if len(question) > 55 else question
            print(f"  {mark} [{expert or '?':15}] {q}")
            print(f"      Output: {output[:80]}...")
            print(f"      Reward: {reward:.1f} ({reason})")

        # Show full trace for wrong answers (debugging)
        if show_wrong_traces and reward < 1.0 and reward >= 0.5:
            print(f"\n  === WRONG ANSWER DEBUG ===")
            print(f"  Question: {question}")
            print(f"  Expected: {ex.get('answer')}")
            print(f"  Got: {reason}")
            print(f"  Full trace:")
            print(f"  {output}")
            print(f"  ===========================\n")

    if not quiet and len(data) > 10:
        print("                                    ", end="\r")  # clear progress line
    return stats


def rl_step(
    model, tokenizer, data: list[dict], optimizer, baseline: float,
    batch_size: int = 8, temp: float = 0.7, max_tokens: int = 250,
):
    """One REINFORCE step with graduated rewards."""
    batch = random.sample(data, min(batch_size, len(data)))

    # Sample outputs and collect sequences
    samples = []
    for ex in batch:
        question = ex["query"]
        prompt = format_chat_prompt(question)
        prompt_tokens = tokenizer.encode(prompt)

        # Generate with sampling
        input_ids = mx.array([prompt_tokens])
        generated_tokens = []

        for _ in range(max_tokens):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            logits = logits[:, -1, :] / temp

            next_token = int(mx.random.categorical(logits).item())
            generated_tokens.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            text = tokenizer.decode(generated_tokens)
            if "</s>" in text:
                break
            if "```" in text and text.count("```") >= 1 and not text.strip().endswith("```yaml"):
                break

        output_text = tokenizer.decode(generated_tokens).replace("</s>", "").strip()
        reward, reason = compute_reward(
            output_text,
            ex.get("answer", 0),
            expected_expert=ex.get("expert"),
        )
        samples.append((ex, prompt_tokens, generated_tokens, reward, reason))

    avg_reward = sum(s[3] for s in samples) / len(samples)

    # Policy gradient loss
    def loss_fn(model):
        total = mx.array(0.0)
        for ex, prompt_tokens, gen_tokens, reward, _ in samples:
            if not gen_tokens:
                continue

            full_tokens = prompt_tokens + gen_tokens
            input_ids = mx.array([full_tokens[:-1]])
            targets = mx.array([full_tokens[1:]])

            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            gen_start = len(prompt_tokens) - 1
            gen_logits = logits[:, gen_start:, :]
            gen_targets = targets[:, gen_start:]

            log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)
            chosen_log_probs = mx.take_along_axis(
                log_probs,
                gen_targets[:, :, None],
                axis=-1,
            ).squeeze(-1)

            seq_log_prob = chosen_log_probs.sum()
            advantage = reward - baseline
            total = total - seq_log_prob * advantage

        return total / max(len(samples), 1)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    new_baseline = 0.9 * baseline + 0.1 * avg_reward
    correct = sum(1 for s in samples if s[3] >= 1.0)

    return avg_reward, new_baseline, correct


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GSM-8K YAML Trace Training")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--rl-iters", type=int, default=20)
    parser.add_argument("--minimal-sft", action="store_true", help="1 epoch SFT + more RL")
    parser.add_argument("--fast", action="store_true", help="Smaller batches for speed")
    parser.add_argument("--use-hf", action="store_true", help="Load GSM-8K from HuggingFace for eval")
    parser.add_argument("--eval-n", type=int, default=20, help="Number of HF problems to eval")
    parser.add_argument("--n-train", type=int, default=250, help="Number of training examples to generate")
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Save trained weights to directory")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Load weights from checkpoint directory")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--eval-sample", type=int, default=0, help="Limit final eval to N training examples (0=all)")
    parser.add_argument("--show-wrong-traces", action="store_true", help="Print full trace for wrong answers (debugging)")
    args = parser.parse_args()

    # Config
    rl_batch_size = 4 if args.fast else 8
    max_gen_tokens = 150 if args.fast else 250

    if args.minimal_sft:
        args.sft_epochs = 1
        if args.rl_iters < 20:
            args.rl_iters = 20

    print("=" * 70)
    print("  GSM-8K YAML TRACE TRAINING (Rogue-1 Format)")
    print("=" * 70)

    # Load training data (static GSM-8K + synthetic)
    train_data = load_training_data(n=args.n_train)
    print(f"Training examples: {len(train_data)}")

    # Show expert distribution
    expert_counts = {}
    for ex in train_data:
        if ex.get("composed"):
            expert_counts["composition"] = expert_counts.get("composition", 0) + 1
        else:
            expert_counts[ex["expert"]] = expert_counts.get(ex["expert"], 0) + 1
    for expert, count in sorted(expert_counts.items()):
        print(f"  {expert}: {count}")

    # Load evaluation data
    eval_data = get_sample_problems(10)
    eval_examples = gsm8k_to_examples(eval_data)
    print(f"Eval examples (sample): {len(eval_examples)}")

    if args.use_hf:
        try:
            from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import load_gsm8k
            hf_problems = load_gsm8k(n=args.eval_n, split="test", shuffle=True)
            hf_eval = gsm8k_to_examples(hf_problems)
            print(f"HuggingFace eval problems: {len(hf_eval)}")
        except ImportError:
            print("HuggingFace datasets not available, using sample problems only")
            hf_eval = None
    else:
        hf_eval = None

    # Verify training data through verifier
    print("\nVerifying training examples...")
    verified = 0
    for ex in train_data:
        if ex.get("composed"):
            yaml_str = yaml.dump(ex["steps"], default_flow_style=None, sort_keys=False)
        else:
            yaml_str = yaml.dump(
                {"expert": ex["expert"], "trace": ex["trace"]},
                default_flow_style=None, sort_keys=False,
            )
        result = verify_yaml_output(yaml_str, expected_answer=ex["answer"])
        if result["answer_correct"]:
            verified += 1
        else:
            q = ex["query"][:50]
            print(f"  WARN: '{q}...' -> {result['computed_answer']} (expected {ex['answer']})")

    print(f"  Verified: {verified}/{len(train_data)}")

    # Load model
    print(f"\nLoading {args.model}...")
    result = load_model(args.model)
    model, tokenizer = result.model, result.tokenizer

    # Load checkpoint if specified
    if args.load_checkpoint:
        load_checkpoint(model, args.load_checkpoint)

    if not args.eval_only:
        # Baseline evaluation
        print("\n" + "=" * 70)
        print("BASELINE (before training)")
        print("=" * 70)
        model.freeze()
        stats = evaluate(model, tokenizer, train_data[:5], show=3)
        print(f"\nBaseline: {stats['correct']}/{stats['total']} correct, {stats['parsed']}/{stats['total']} parsed")

        # Unfreeze layers
        for layer in model.model.layers[-6:]:
            layer.unfreeze()
        model.lm_head.unfreeze()

        # SFT Phase
        print("\n" + "=" * 70)
        print(f"SFT ({args.sft_epochs} epochs)")
        print("=" * 70)

        sft_opt = optim.Adam(learning_rate=2e-5)
        batch_size = 4

        for epoch in range(args.sft_epochs):
            random.shuffle(train_data)
            losses = []
            n_steps = (len(train_data) + batch_size - 1) // batch_size
            for step_i, i in enumerate(range(0, len(train_data), batch_size)):
                loss = sft_step(model, tokenizer, train_data[i : i + batch_size], sft_opt)
                losses.append(loss)
                if (step_i + 1) % 10 == 0 or step_i == n_steps - 1:
                    avg = sum(losses[-10:]) / len(losses[-10:])
                    print(f"  Epoch {epoch+1} step {step_i+1}/{n_steps}: loss={avg:.4f}", flush=True)

            # Quick eval on subset
            eval_subset = train_data[:20]
            stats = evaluate(model, tokenizer, eval_subset)
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            parse_rate = stats["parsed"] / stats["total"] if stats["total"] else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            print(f"  Epoch {epoch+1} done: loss={avg_loss:.4f} correct={acc:.0%} parsed={parse_rate:.0%} (eval on 20)")

        print("\nAfter SFT (sample outputs):")
        evaluate(model, tokenizer, train_data[:5], show=5)

        # Save SFT checkpoint
        if args.save_checkpoint:
            save_checkpoint(model, args.save_checkpoint + "_sft")

        # RL Phase
        if args.rl_iters > 0:
            print("\n" + "=" * 70)
            print(f"RL ({args.rl_iters} iterations, batch={rl_batch_size})")
            print("=" * 70)

            rl_opt = optim.Adam(learning_rate=5e-7)
            baseline = 0.5

            for i in range(args.rl_iters):
                avg_r, baseline, correct = rl_step(
                    model, tokenizer, train_data, rl_opt, baseline,
                    batch_size=rl_batch_size, max_tokens=max_gen_tokens,
                )
                print(
                    f"  RL iter {i+1:2d}/{args.rl_iters}: reward={avg_r:.2f} "
                    f"batch_correct={correct}/{rl_batch_size} baseline={baseline:.2f}"
                )
                if (i + 1) % 5 == 0:
                    eval_subset = train_data[:20]
                    stats = evaluate(model, tokenizer, eval_subset)
                    print(
                        f"    eval: {stats['correct']}/{stats['total']} correct "
                        f"({stats['parsed']}/{stats['total']} parsed)"
                    )

        # Save checkpoint if specified
        if args.save_checkpoint:
            save_checkpoint(model, args.save_checkpoint)

    # Final Evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print("\nTraining data:")
    eval_data = train_data
    if args.eval_sample > 0:
        eval_data = random.sample(train_data, min(args.eval_sample, len(train_data)))
        print(f"  (sampled {len(eval_data)} of {len(train_data)})")
    final = evaluate(model, tokenizer, eval_data, show=5)
    print(f"\n  Overall: {final['correct']}/{final['total']} ({final['correct']/max(final['total'],1):.0%})")
    print(f"  Parsed: {final['parsed']}/{final['total']}")
    print(f"  Valid traces: {final['valid_trace']}/{final['total']}")
    print(f"  Wrong answer: {final['wrong_answer']}")
    print(f"  Wrong expert: {final['wrong_expert']}")

    if final["by_expert"]:
        print("\n  By expert:")
        for expert, s in sorted(final["by_expert"].items()):
            if s["total"] > 0:
                acc = s["correct"] / s["total"]
                print(f"    {expert:15} {acc:.0%} ({s['correct']}/{s['total']})")

    # Eval on sample GSM-8K problems
    print("\nGSM-8K sample problems:")
    gsm_stats = evaluate(
        model, tokenizer, eval_examples, show=3,
        show_wrong_traces=args.show_wrong_traces,
    )
    print(f"  Correct: {gsm_stats['correct']}/{gsm_stats['total']}")
    print(f"  Valid traces: {gsm_stats['valid_trace']}/{gsm_stats['total']}")

    # Eval on HuggingFace GSM-8K
    if hf_eval:
        print("\nHuggingFace GSM-8K (unseen problems):")
        hf_stats = evaluate(
            model, tokenizer, hf_eval, show=5,
            show_wrong_traces=args.show_wrong_traces,
        )
        print(f"  Correct: {hf_stats['correct']}/{hf_stats['total']}")
        print(f"  Parsed: {hf_stats['parsed']}/{hf_stats['total']}")
        print(f"  Valid traces: {hf_stats['valid_trace']}/{hf_stats['total']}")


if __name__ == "__main__":
    main()
