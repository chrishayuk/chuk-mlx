#!/usr/bin/env python3
"""
Hero Demo 2: SFT TinyLlama on Synthetic Math Steps (Small Run)

This demo shows how to:
1. Generate synthetic math training data with step-by-step reasoning
2. Create an SFT dataset with proper chat formatting
3. Fine-tune TinyLlama using LoRA on the synthetic data
4. Run inference to test the trained model

This is a "small run" designed to complete quickly for demonstration.
Adjust parameters for production training.

Usage:
    python examples/training/hero_sft_math_demo.py

Or via CLI:
    uvx chuk-lazarus generate --type math --output ./data/math
    uvx chuk-lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --data ./data/math/sft_train.jsonl --use-lora --epochs 1
"""

import json
import random
import tempfile
from pathlib import Path


def generate_math_problem() -> dict:
    """Generate a random math problem with step-by-step solution."""
    problem_type = random.choice(["addition", "subtraction", "multiplication", "division"])

    if problem_type == "addition":
        a = random.randint(10, 100)
        b = random.randint(10, 100)
        answer = a + b
        question = f"What is {a} + {b}?"
        steps = [
            f"I need to add {a} and {b}.",
            f"Starting with {a}, I add {b}.",
            f"{a} + {b} = {answer}.",
            f"The answer is {answer}.",
        ]

    elif problem_type == "subtraction":
        a = random.randint(50, 200)
        b = random.randint(10, a - 10)
        answer = a - b
        question = f"What is {a} - {b}?"
        steps = [
            f"I need to subtract {b} from {a}.",
            f"Starting with {a}, I subtract {b}.",
            f"{a} - {b} = {answer}.",
            f"The answer is {answer}.",
        ]

    elif problem_type == "multiplication":
        a = random.randint(2, 15)
        b = random.randint(2, 15)
        answer = a * b
        question = f"What is {a} times {b}?"
        steps = [
            f"I need to multiply {a} by {b}.",
            f"I can think of this as adding {a} to itself {b} times.",
            f"{a} × {b} = {answer}.",
            f"The answer is {answer}.",
        ]

    else:  # division
        b = random.randint(2, 12)
        answer = random.randint(2, 15)
        a = b * answer
        question = f"What is {a} divided by {b}?"
        steps = [
            f"I need to divide {a} by {b}.",
            f"I'm looking for how many times {b} goes into {a}.",
            f"{a} ÷ {b} = {answer}.",
            f"The answer is {answer}.",
        ]

    return {
        "question": question,
        "answer": str(answer),
        "steps": steps,
        "full_response": "\n".join(steps),
    }


def generate_sft_dataset(num_samples: int = 100, output_path: Path | None = None) -> list[dict]:
    """Generate SFT training data in chat format."""
    print(f"\nGenerating {num_samples} math problems...")

    data = []
    for i in range(num_samples):
        problem = generate_math_problem()

        # Format as chat conversation
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Show your reasoning step by step.",
                },
                {"role": "user", "content": problem["question"]},
                {"role": "assistant", "content": problem["full_response"]},
            ]
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


def demo_data_generation():
    """Demonstrate synthetic data generation."""
    print("=" * 60)
    print("Step 1: Generate Synthetic Math Data")
    print("=" * 60)

    # Show a few examples
    print("\nSample problems:")
    print("-" * 40)

    for i in range(3):
        problem = generate_math_problem()
        print(f"\nProblem {i+1}:")
        print(f"  Q: {problem['question']}")
        print(f"  A: {problem['full_response'][:100]}...")


def demo_dataset_creation(output_dir: Path):
    """Demonstrate SFT dataset creation."""
    print("\n" + "=" * 60)
    print("Step 2: Create SFT Dataset")
    print("=" * 60)

    # Generate training data
    train_data = generate_sft_dataset(
        num_samples=100, output_path=output_dir / "sft_train.jsonl"
    )

    # Generate eval data
    eval_data = generate_sft_dataset(num_samples=20, output_path=output_dir / "sft_eval.jsonl")

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Evaluation samples: {len(eval_data)}")

    # Show format
    print("\nSample entry format:")
    print(json.dumps(train_data[0], indent=2)[:500] + "...")


def demo_training_config():
    """Show training configuration for the demo."""
    print("\n" + "=" * 60)
    print("Step 3: Training Configuration")
    print("=" * 60)

    config = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "use_lora": True,
        "lora_rank": 8,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "max_length": 256,
        "mask_prompt": True,
    }

    print("\nRecommended config for demo (quick run):")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nCLI command:")
    print('  uvx chuk-lazarus train sft \\')
    print('    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\')
    print('    --data ./data/math/sft_train.jsonl \\')
    print('    --eval-data ./data/math/sft_eval.jsonl \\')
    print('    --use-lora --lora-rank 8 \\')
    print('    --epochs 1 --batch-size 4 \\')
    print('    --output ./checkpoints/math-sft')


def demo_inference():
    """Show how to run inference after training."""
    print("\n" + "=" * 60)
    print("Step 4: Inference After Training")
    print("=" * 60)

    print("\nAfter training, test the model:")
    print("\nCLI command:")
    print('  uvx chuk-lazarus infer \\')
    print('    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\')
    print('    --adapter ./checkpoints/math-sft/final \\')
    print('    --prompt "What is 42 + 58?"')

    print("\nPython code:")
    print("""
    from chuk_lazarus.models import load_model
    from chuk_lazarus.inference import Generator

    model = load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path="./checkpoints/math-sft/final"
    )

    generator = Generator(model.model, model.tokenizer)
    response = generator.generate("What is 42 + 58?", max_tokens=100)
    print(response)
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("HERO DEMO 2: SFT TinyLlama on Synthetic Math Steps")
    print("=" * 60)

    # Create temp directory for demo data
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "math"

        # Run demos
        demo_data_generation()
        demo_dataset_creation(output_dir)
        demo_training_config()
        demo_inference()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo run actual training:")
    print("  1. Generate data: uvx chuk-lazarus generate --type math --output ./data/math")
    print("  2. Train model:   uvx chuk-lazarus train sft --model TinyLlama/... --data ./data/math/sft_train.jsonl")
    print("  3. Test model:    uvx chuk-lazarus infer --model TinyLlama/... --adapter ./checkpoints/...")
