#!/usr/bin/env python3
"""
Phase 1: Dual-Reward V/O Training

This trains V/O projections to create vocabulary-aligned classifiers
at the intermediate layer.

Usage:
    python experiments/cli_classifier_emergence/train_phase1.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --steps 500 \
        --output experiments/cli_classifier_emergence/checkpoints/phase1
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str):
    """Load JSONL dataset."""
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Dual-Reward V/O Training")
    parser.add_argument("--model", "-m", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", "-d", default="experiments/cli_classifier_emergence/data/arithmetic_sft.jsonl")
    parser.add_argument("--output", "-o", default="experiments/cli_classifier_emergence/checkpoints/phase1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--classifier-layer", type=int, default=-1, help="-1 means 55% depth")
    parser.add_argument("--classifier-weight", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

    result = HFLoader.download(args.model)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.FLOAT32)
    tokenizer = HFLoader.load_tokenizer(model_path)

    # Load dataset
    logger.info(f"Loading dataset: {args.data}")
    dataset = load_dataset(args.data)
    logger.info(f"Loaded {len(dataset)} samples")

    # Create trainer
    from chuk_lazarus.training.trainers.dual_reward_trainer import (
        DualRewardTrainer,
        DualRewardTrainerConfig,
    )

    trainer_config = DualRewardTrainerConfig(
        max_steps=args.steps,
        classifier_layer=args.classifier_layer,
        classifier_weight=args.classifier_weight,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_targets=["v_proj", "o_proj"],
        log_interval=args.log_interval,
        checkpoint_interval=args.steps,  # Save at end
        checkpoint_dir=args.output,
    )

    trainer = DualRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        config=trainer_config,
        model_config=config,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(dataset)

    # Evaluate
    logger.info("\nEvaluating classifier...")
    test_prompts = [
        ("7 * 8 = ", "multiply"),
        ("12 * 5 = ", "multiply"),
        ("23 + 45 = ", "add"),
        ("17 + 38 = ", "add"),
        ("50 - 23 = ", "subtract"),
        ("89 - 34 = ", "subtract"),
        ("48 / 6 = ", "divide"),
        ("81 / 9 = ", "divide"),
    ]

    eval_results = trainer.evaluate_classifier(test_prompts)

    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"{'Prompt':<15} {'Expected':<12} {'Predicted':<12} {'Conf':>8} {'Status'}")
    print("-" * 60)

    for r in eval_results["results"]:
        status = "OK" if r["correct"] else "XX"
        print(f"  {r['prompt']:<13} {r['expected']:<12} {r['predicted']:<12} {r['confidence']:>7.1%} [{status}]")

    print("-" * 60)
    print(f"\nAccuracy: {eval_results['correct']}/{eval_results['total']} ({eval_results['accuracy']:.1%})")

    # Save final config with results
    final_config = {
        "model": args.model,
        "classifier_layer": trainer.classifier_layer,
        "classifier_weight": args.classifier_weight,
        "lora_rank": args.lora_rank,
        "steps": args.steps,
        "final_accuracy": eval_results["accuracy"],
        "classifier_token_ids": trainer.classifier_token_ids,
    }

    output_path = Path(args.output)
    with open(output_path / "training_config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    logger.info(f"\nCheckpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
