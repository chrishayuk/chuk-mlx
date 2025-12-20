"""
Example: Train Mistral-7B with DPO for tool use preferences.

This demonstrates Phase 1 of the hybrid architecture:
- Start with SFT'd Mistral
- Fine-tune with DPO on tool use preferences
- Learn to prefer correct tool calls over incorrect ones

Usage:
    python -m rl.examples.train_mistral_dpo \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --data ./data/tool_preferences.jsonl
"""

import logging
import argparse
import json
from pathlib import Path

import mlx.core as mx

# Import from existing chuk-mlx infrastructure
from core.models.model_loader import load_model
from core.utils.tokenizer_loader import load_tokenizer

from ..losses.dpo_loss import DPOConfig
from ..data.preference_dataset import PreferenceDataset
from ..trainers.dpo_trainer import DPOTrainer, DPOTrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_preference_data(output_path: str, num_samples: int = 100):
    """
    Create sample preference data for tool use.

    In production, you'd generate this using MCP tools to verify correctness.
    """
    import random

    samples = []

    # Math tool preferences
    for i in range(num_samples // 2):
        a, b = random.randint(1, 100), random.randint(1, 100)
        correct = a + b

        samples.append({
            "prompt": f"What is {a} + {b}?",
            "chosen": f"I'll use the math tool to calculate this.\nTOOL: math_solve(expression=\"{a}+{b}\")\nResult: {correct}\n\nThe answer is {correct}.",
            "rejected": f"Let me think... {a} + {b} = {correct + random.randint(1, 10)}."  # Wrong!
        })

    # Physics tool preferences
    for i in range(num_samples // 2):
        target = random.uniform(50, 150)

        samples.append({
            "prompt": f"Find the angle to hit a target at {target:.1f} meters.",
            "chosen": f"I'll delegate this to the physics controller.\nDELEGATE: physics_controller(target={target:.1f})\nThe controller found the optimal angle.",
            "rejected": f"I'll guess... maybe 45 degrees? That's usually a good angle."
        })

    # Write to file
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Created {len(samples)} preference samples at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Mistral with DPO")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--data", type=str, required=True, help="Preference data path")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/mistral_dpo")
    args = parser.parse_args()

    # Create sample data if requested
    if args.create_sample_data:
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        create_sample_preference_data(args.data)

    # Check data exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.error("Use --create-sample-data to generate sample data")
        return

    logger.info(f"Loading model: {args.model}")

    # Load policy model
    policy_model = load_model(args.model, load_weights=True)
    policy_model.set_mode('TRAIN')

    # Load reference model (frozen copy)
    reference_model = load_model(args.model, load_weights=True)
    reference_model.set_mode('INFERENCE')

    # Load tokenizer
    tokenizer = load_tokenizer(args.model)

    logger.info("Loading preference dataset...")

    # Load dataset
    dataset = PreferenceDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
    )

    logger.info(f"Loaded {len(dataset)} preference pairs")

    # Training config
    config = DPOTrainerConfig(
        dpo=DPOConfig(
            beta=args.beta,
            label_smoothing=0.0,
        ),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=10,
        checkpoint_interval=100,
    )

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config,
    )

    logger.info("Starting DPO training...")
    trainer.train(dataset)

    logger.info("Training complete!")
    logger.info(f"Best reward margin: {trainer.best_reward_margin:.4f}")


if __name__ == "__main__":
    main()
