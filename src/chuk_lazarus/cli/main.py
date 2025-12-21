"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"
    lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
    lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3"
    lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test"
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_sft(args):
    """Run SFT training."""
    from ..data import SFTDataset
    from ..models import load_model
    from ..training import SFTTrainer
    from ..training.losses import SFTConfig

    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, use_lora=args.use_lora, lora_rank=args.lora_rank)

    logger.info(f"Loading dataset: {args.data}")
    dataset = SFTDataset(
        args.data, model.tokenizer, max_length=args.max_length, mask_prompt=args.mask_prompt
    )

    eval_dataset = None
    if args.eval_data:
        eval_dataset = SFTDataset(
            args.eval_data,
            model.tokenizer,
            max_length=args.max_length,
            mask_prompt=args.mask_prompt,
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
    from ..data import PreferenceDataset
    from ..models import load_model
    from ..training import DPOTrainer, DPOTrainerConfig
    from ..training.losses import DPOConfig

    logger.info(f"Loading policy model: {args.model}")
    policy_model = load_model(args.model, use_lora=args.use_lora, lora_rank=args.lora_rank)

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

    trainer = DPOTrainer(policy_model.model, ref_model.model, policy_model.tokenizer, config)
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
            seed=args.seed,
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


def tokenizer_encode(args):
    """Encode text and display tokens."""
    from ..data.tokenizers.token_display import TokenDisplayUtility
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    display = TokenDisplayUtility(tokenizer)

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            texts = [f.read()]
    else:
        # Interactive mode
        print("Enter text to tokenize (Ctrl+D to finish):")
        try:
            texts = [input("> ")]
        except EOFError:
            return

    for text in texts:
        print(f"\nText: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Length: {len(text)} chars\n")
        display.display_tokens_from_prompt(text, add_special_tokens=args.special_tokens)


def tokenizer_decode(args):
    """Decode token IDs back to text."""
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Parse token IDs from comma-separated or space-separated string
    token_ids = [int(t.strip()) for t in args.ids.replace(",", " ").split()]

    decoded = tokenizer.decode(token_ids)
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")


def tokenizer_vocab(args):
    """Display vocabulary information."""
    from ..data.tokenizers.token_display import TokenDisplayUtility
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    vocab = tokenizer.get_vocab()
    print("\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab)}")

    if hasattr(tokenizer, "pad_token_id"):
        print(f"  Pad token ID: {tokenizer.pad_token_id}")
    if hasattr(tokenizer, "eos_token_id"):
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
    if hasattr(tokenizer, "bos_token_id"):
        print(f"  BOS token ID: {tokenizer.bos_token_id}")
    if hasattr(tokenizer, "unk_token_id"):
        print(f"  UNK token ID: {tokenizer.unk_token_id}")

    if args.show_all:
        display = TokenDisplayUtility(tokenizer)
        display.display_full_vocabulary(chunk_size=args.chunk_size, pause_between_chunks=args.pause)
    elif args.search:
        # Search for tokens containing the search string
        print(f"\nTokens containing '{args.search}':")
        matches = [
            (token, id) for token, id in vocab.items() if args.search.lower() in token.lower()
        ]
        matches.sort(key=lambda x: x[1])
        for token, id in matches[: args.limit]:
            decoded = tokenizer.decode([id])
            print(f"  {id:6d}: {repr(token):30s} -> {repr(decoded)}")
        if len(matches) > args.limit:
            print(f"  ... and {len(matches) - args.limit} more matches")


def tokenizer_compare(args):
    """Compare tokenization between two tokenizers."""
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {args.tokenizer1}")
    tok1 = load_tokenizer(args.tokenizer1)
    logger.info(f"Loading tokenizer 2: {args.tokenizer2}")
    tok2 = load_tokenizer(args.tokenizer2)

    text = args.text

    ids1 = tok1.encode(text)
    ids2 = tok2.encode(text)

    print(f"\nText: {text}")
    print(f"\n{args.tokenizer1}:")
    print(f"  Token count: {len(ids1)}")
    print(f"  Token IDs: {ids1[:20]}{'...' if len(ids1) > 20 else ''}")

    print(f"\n{args.tokenizer2}:")
    print(f"  Token count: {len(ids2)}")
    print(f"  Token IDs: {ids2[:20]}{'...' if len(ids2) > 20 else ''}")

    print(f"\nDifference: {len(ids1) - len(ids2):+d} tokens")


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
        """,
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

    # Tokenizer subcommand
    tok_parser = subparsers.add_parser("tokenizer", help="Tokenizer utilities")
    tok_subparsers = tok_parser.add_subparsers(dest="tok_command", help="Tokenizer commands")

    # Encode command
    encode_parser = tok_subparsers.add_parser("encode", help="Encode text to tokens")
    encode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    encode_parser.add_argument("--text", help="Text to encode")
    encode_parser.add_argument("--file", "-f", help="File to encode")
    encode_parser.add_argument("--special-tokens", action="store_true", help="Add special tokens")
    encode_parser.set_defaults(func=tokenizer_encode)

    # Decode command
    decode_parser = tok_subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    decode_parser.add_argument("--ids", required=True, help="Token IDs (comma or space separated)")
    decode_parser.set_defaults(func=tokenizer_decode)

    # Vocab command
    vocab_parser = tok_subparsers.add_parser("vocab", help="Display vocabulary info")
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--show-all", action="store_true", help="Show full vocabulary")
    vocab_parser.add_argument("--search", "-s", help="Search for tokens containing string")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Max results for search")
    vocab_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for display")
    vocab_parser.add_argument("--pause", action="store_true", help="Pause between chunks")
    vocab_parser.set_defaults(func=tokenizer_vocab)

    # Compare command
    compare_parser = tok_subparsers.add_parser("compare", help="Compare two tokenizers")
    compare_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    compare_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    compare_parser.add_argument("--text", required=True, help="Text to compare")
    compare_parser.set_defaults(func=tokenizer_compare)

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
    elif args.command == "tokenizer" and args.tok_command is None:
        parser.parse_args(["tokenizer", "--help"])


if __name__ == "__main__":
    main()
