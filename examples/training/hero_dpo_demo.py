#!/usr/bin/env python3
"""
Hero Demo 3: DPO/GRPO Preference Tuning on Puzzle Outcomes

This demo shows how to:
1. Generate preference pairs from puzzle solving outcomes
2. Create a preference dataset for DPO training
3. Configure and run DPO training
4. Evaluate preference alignment

This demonstrates the "small tired model resurrected" storyline where
we teach models to prefer correct reasoning over incorrect reasoning.

Usage:
    python examples/training/hero_dpo_demo.py

Or via CLI:
    uvx chuk-lazarus train dpo \
        --model ./checkpoints/sft/final \
        --data ./data/preferences.jsonl \
        --use-lora --beta 0.1
"""

import json
import random
import tempfile
from pathlib import Path


def generate_puzzle_outcome() -> dict:
    """Generate a puzzle with correct and incorrect solution attempts."""
    puzzle_type = random.choice(["arithmetic", "sequence", "logic"])

    if puzzle_type == "arithmetic":
        # Simple arithmetic puzzle
        a = random.randint(10, 50)
        b = random.randint(10, 50)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:
            answer = a * b

        wrong_answer = answer + random.choice([-5, -3, -1, 1, 3, 5])

        prompt = f"Calculate: {a} {op} {b}"

        chosen = f"Let me work through this step by step.\n{a} {op} {b} = {answer}\nThe answer is {answer}."

        rejected = f"The answer is {wrong_answer}."  # No reasoning, wrong answer

    elif puzzle_type == "sequence":
        # Number sequence puzzle
        start = random.randint(1, 10)
        step = random.randint(2, 5)
        sequence = [start + step * i for i in range(4)]
        next_num = start + step * 4
        wrong_next = next_num + random.randint(-3, 3)
        if wrong_next == next_num:
            wrong_next += 1

        prompt = f"What comes next in the sequence: {', '.join(map(str, sequence))}, ?"

        chosen = (
            f"I notice each number increases by {step}.\n"
            f"{sequence[-1]} + {step} = {next_num}\n"
            f"The next number is {next_num}."
        )

        rejected = f"The next number is {wrong_next}."

    else:  # logic
        # Simple logic puzzle
        items = ["apples", "oranges", "bananas"]
        item = random.choice(items)
        total = random.randint(5, 15)
        given = random.randint(1, total - 1)
        remaining = total - given

        prompt = f"I have {total} {item}. I give away {given}. How many do I have left?"

        chosen = (
            f"Let me think about this.\n"
            f"Starting with {total} {item}.\n"
            f"After giving away {given}: {total} - {given} = {remaining}\n"
            f"I have {remaining} {item} left."
        )

        rejected = f"You have {remaining + random.randint(1, 3)} {item}."

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "puzzle_type": puzzle_type,
    }


def generate_preference_dataset(
    num_samples: int = 50, output_path: Path | None = None
) -> list[dict]:
    """Generate preference pairs for DPO training."""
    print(f"\nGenerating {num_samples} preference pairs...")

    data = []
    for _ in range(num_samples):
        outcome = generate_puzzle_outcome()

        # Format for DPO training
        entry = {
            "prompt": outcome["prompt"],
            "chosen": outcome["chosen"],
            "rejected": outcome["rejected"],
        }
        data.append(entry)

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"   Saved to: {output_path}")

    return data


def demo_preference_pairs():
    """Demonstrate preference pair generation."""
    print("=" * 60)
    print("Step 1: Generate Preference Pairs from Puzzle Outcomes")
    print("=" * 60)

    print("\nThe key insight: we want the model to prefer:")
    print("  - Responses with step-by-step reasoning")
    print("  - Correct answers over incorrect ones")
    print("  - Detailed explanations over terse responses")

    print("\nSample preference pairs:")
    print("-" * 60)

    for i in range(3):
        outcome = generate_puzzle_outcome()
        print(f"\n--- Pair {i + 1} ({outcome['puzzle_type']}) ---")
        print(f"Prompt: {outcome['prompt']}")
        print(f"\nCHOSEN (preferred):\n{outcome['chosen']}")
        print(f"\nREJECTED:\n{outcome['rejected']}")


def demo_dataset_creation(output_dir: Path):
    """Demonstrate preference dataset creation."""
    print("\n" + "=" * 60)
    print("Step 2: Create DPO Dataset")
    print("=" * 60)

    # Generate training data
    train_data = generate_preference_dataset(
        num_samples=50, output_path=output_dir / "dpo_train.jsonl"
    )

    # Generate eval data
    eval_data = generate_preference_dataset(
        num_samples=10, output_path=output_dir / "dpo_eval.jsonl"
    )

    print("\nDataset Statistics:")
    print(f"  Training pairs: {len(train_data)}")
    print(f"  Evaluation pairs: {len(eval_data)}")

    # Show format
    print("\nData format:")
    sample = train_data[0]
    print(f"  prompt: {sample['prompt'][:50]}...")
    print(f"  chosen: {sample['chosen'][:50]}...")
    print(f"  rejected: {sample['rejected'][:50]}...")


def demo_training_config():
    """Show DPO training configuration."""
    print("\n" + "=" * 60)
    print("Step 3: DPO Training Configuration")
    print("=" * 60)

    config = {
        "model": "./checkpoints/sft/final",
        "ref_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "beta": 0.1,
        "use_lora": True,
        "lora_rank": 8,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-6,
        "max_length": 256,
    }

    print("\nDPO hyperparameters:")
    print(f"  beta: {config['beta']} (controls preference strength)")
    print("    - Lower beta: stronger preference signal")
    print("    - Higher beta: more conservative updates")

    print("\nRecommended config for demo:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nCLI command:")
    print("  uvx chuk-lazarus train dpo \\")
    print("    --model ./checkpoints/sft/final \\")
    print("    --ref-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\")
    print("    --data ./data/preferences/dpo_train.jsonl \\")
    print("    --eval-data ./data/preferences/dpo_eval.jsonl \\")
    print("    --beta 0.1 \\")
    print("    --use-lora --lora-rank 8 \\")
    print("    --epochs 1 --batch-size 2 \\")
    print("    --output ./checkpoints/dpo")


def demo_workflow():
    """Show complete workflow from SFT to DPO."""
    print("\n" + "=" * 60)
    print("Step 4: Complete Training Workflow")
    print("=" * 60)

    print("\nThe 'Small Tired Model Resurrected' Pipeline:")
    print("-" * 60)

    steps = [
        (
            "1. Generate SFT Data",
            "Create math problems with step-by-step solutions",
            "uvx chuk-lazarus generate --type math --output ./data/math",
        ),
        (
            "2. SFT Training",
            "Teach the model to produce reasoning chains",
            "uvx chuk-lazarus train sft --model TinyLlama/... --data ./data/math/sft_train.jsonl --use-lora",
        ),
        (
            "3. Generate Preference Pairs",
            "Create good/bad response pairs from puzzle outcomes",
            "python examples/training/hero_dpo_demo.py --generate",
        ),
        (
            "4. DPO Training",
            "Align model to prefer correct reasoning",
            "uvx chuk-lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences/dpo_train.jsonl",
        ),
        (
            "5. Evaluate",
            "Test preference alignment on held-out puzzles",
            "uvx chuk-lazarus infer --model ./checkpoints/dpo/final --prompt 'What is 25 + 37?'",
        ),
    ]

    for step_name, description, command in steps:
        print(f"\n{step_name}")
        print(f"  Goal: {description}")
        print(f"  Command: {command}")


def demo_grpo_variant():
    """Show GRPO variant for group-based preference optimization."""
    print("\n" + "=" * 60)
    print("Bonus: GRPO (Group Relative Policy Optimization)")
    print("=" * 60)

    print("\nGRPO extends DPO for scenarios with multiple completions:")
    print("  - Sample K completions per prompt")
    print("  - Rank by reward (e.g., correctness, reasoning quality)")
    print("  - Optimize preferences among the group")

    print("\nUseful for:")
    print("  - Puzzle gyms with multiple solution attempts")
    print("  - Code generation with test-based scoring")
    print("  - Math reasoning with verification")

    print("\nPseudo-code for GRPO data generation:")
    print("""
    for puzzle in puzzles:
        completions = []
        for _ in range(K):
            response = model.generate(puzzle.prompt)
            score = evaluate(response, puzzle.answer)
            completions.append((response, score))

        # Create pairwise preferences from rankings
        completions.sort(key=lambda x: x[1], reverse=True)
        for i, (chosen, _) in enumerate(completions[:-1]):
            for rejected, _ in completions[i+1:]:
                preferences.append({
                    'prompt': puzzle.prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("HERO DEMO 3: DPO/GRPO Preference Tuning on Puzzle Outcomes")
    print("=" * 60)

    # Create temp directory for demo data
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "preferences"

        # Run demos
        demo_preference_pairs()
        demo_dataset_creation(output_dir)
        demo_training_config()
        demo_workflow()
        demo_grpo_variant()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. DPO learns from preference pairs (chosen vs rejected)")
    print("  2. Quality of pairs matters more than quantity")
    print("  3. Works best after SFT establishes base capabilities")
    print("  4. GRPO extends to multiple completions per prompt")
    print("\nNext steps:")
    print("  - Generate preference data from your puzzle gym")
    print("  - Run DPO training on SFT checkpoint")
    print("  - Iterate on data quality based on evaluation")
