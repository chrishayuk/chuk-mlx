"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_sft(args):
    """Run SFT training."""
    from ..models import load_model
    from ..data import SFTDataset
    from ..training import SFTTrainer
    from ..training.losses import SFTConfig

    logger.info(f"Loading model: {args.model}")
    model = load_model(
        args.model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank
    )

    logger.info(f"Loading dataset: {args.data}")
    dataset = SFTDataset(
        args.data,
        model.tokenizer,
        max_length=args.max_length,
        mask_prompt=args.mask_prompt
    )

    eval_dataset = None
    if args.eval_data:
        eval_dataset = SFTDataset(
            args.eval_data,
            model.tokenizer,
            max_length=args.max_length,
            mask_prompt=args.mask_prompt
        )

    config = SFTConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_length,
        checkpoint_dir=args.output,
        log_interval=args.log_interval,
    )

    trainer = SFTTrainer(model.model, model.tokenizer, config)
    trainer.train(dataset, eval_dataset)

    logger.info(f"Training complete. Checkpoints saved to {args.output}")


def train_dpo(args):
    """Run DPO training."""
    from ..models import load_model
    from ..data import PreferenceDataset
    from ..training import DPOTrainer, DPOTrainerConfig
    from ..training.losses import DPOConfig

    logger.info(f"Loading policy model: {args.model}")
    policy_model = load_model(
        args.model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank
    )

    logger.info(f"Loading reference model: {args.ref_model or args.model}")
    ref_model = load_model(args.ref_model or args.model, use_lora=False)

    logger.info(f"Loading dataset: {args.data}")
    dataset = PreferenceDataset(
        args.data,
        policy_model.tokenizer,
        max_length=args.max_length,
    )

    eval_dataset = None
    if args.eval_data:
        eval_dataset = PreferenceDataset(
            args.eval_data,
            policy_model.tokenizer,
            max_length=args.max_length,
        )

    config = DPOTrainerConfig(
        dpo=DPOConfig(beta=args.beta),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.output,
    )

    trainer = DPOTrainer(
        policy_model.model,
        ref_model.model,
        policy_model.tokenizer,
        config
    )
    trainer.train(dataset, eval_dataset)

    logger.info(f"Training complete. Checkpoints saved to {args.output}")


def generate_data(args):
    """Generate synthetic training data."""
    from ..data.generators import generate_lazarus_dataset

    if args.type == "math":
        logger.info(f"Generating math dataset with {args.sft_samples} SFT samples")
        generate_lazarus_dataset(
            output_dir=args.output,
            sft_samples=args.sft_samples,
            dpo_samples=args.dpo_samples,
            seed=args.seed
        )
        logger.info(f"Dataset saved to {args.output}")
    else:
        logger.error(f"Unknown data type: {args.type}")
        sys.exit(1)


def run_inference(args):
    """Run inference on a model."""
    from ..models import load_model

    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model)

    if args.adapter:
        logger.info(f"Loading adapter: {args.adapter}")
        model.load_adapter(args.adapter)

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        prompts = []
        print("Enter prompts (Ctrl+D to finish):")
        try:
            while True:
                prompt = input("> ")
                if prompt:
                    prompts.append(prompt)
        except EOFError:
            pass

    for prompt in prompts:
        response = model.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")


def app():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="chuk-lazarus: MLX-based LLM training framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train SFT
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl

    # Train DPO
    lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

    # Generate training data
    lazarus generate --type math --output ./data/lazarus

    # Run inference
    lazarus infer --model ./checkpoints/dpo/final --prompt "What is 2+2?"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Training type")

    # SFT training
    sft_parser = train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", required=True, help="Model name or path")
    sft_parser.add_argument("--data", required=True, help="Training data path (JSONL)")
    sft_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    sft_parser.add_argument("--output", default="./checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    sft_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    sft_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    sft_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    sft_parser.add_argument("--mask-prompt", action="store_true", help="Mask prompt in loss")
    sft_parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    sft_parser.set_defaults(func=train_sft)

    # DPO training
    dpo_parser = train_subparsers.add_parser("dpo", help="Direct Preference Optimization")
    dpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    dpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    dpo_parser.add_argument("--data", required=True, help="Preference data path (JSONL)")
    dpo_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    dpo_parser.add_argument("--output", default="./checkpoints/dpo", help="Output directory")
    dpo_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    dpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    dpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    dpo_parser.set_defaults(func=train_dpo)

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--type", required=True, choices=["math"], help="Data type")
    gen_parser.add_argument("--output", default="./data/generated", help="Output directory")
    gen_parser.add_argument("--sft-samples", type=int, default=10000, help="SFT samples")
    gen_parser.add_argument("--dpo-samples", type=int, default=5000, help="DPO samples")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.set_defaults(func=generate_data)

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", required=True, help="Model name or path")
    infer_parser.add_argument("--adapter", help="LoRA adapter path")
    infer_parser.add_argument("--prompt", help="Single prompt")
    infer_parser.add_argument("--prompt-file", help="File with prompts (one per line)")
    infer_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    infer_parser.set_defaults(func=run_inference)

    return parser


def main():
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "train" and args.train_type is None:
        parser.parse_args(["train", "--help"])


if __name__ == "__main__":
    main()
