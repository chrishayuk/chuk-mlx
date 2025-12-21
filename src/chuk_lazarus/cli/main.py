"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"

Tokenizer Commands:
    lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
    lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3"
    lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test"
    lazarus tokenizer analyze coverage -t model --file corpus.txt
    lazarus tokenizer analyze entropy -t model --file corpus.txt
    lazarus tokenizer analyze fit-score -t model --file corpus.txt
    lazarus tokenizer curriculum length-buckets -t model --file corpus.txt
    lazarus tokenizer curriculum reasoning-density -t model --file corpus.txt
    lazarus tokenizer training throughput -t model --file corpus.txt
    lazarus tokenizer training pack -t model --file corpus.txt --max-length 512
    lazarus tokenizer regression run -t model --tests tests.yaml
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


# === Tokenizer Analyze Commands ===


def _load_texts(args) -> list[str]:
    """Load texts from file or stdin."""
    if args.file:
        with open(args.file) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        print("Enter texts (one per line, Ctrl+D to finish):")
        texts = []
        try:
            while True:
                line = input()
                if line.strip():
                    texts.append(line.strip())
        except EOFError:
            pass
        return texts


def analyze_coverage(args):
    """Analyze token coverage on a corpus."""
    from ..data.tokenizers.analyze import analyze_coverage as do_analyze
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing coverage on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer, include_fragments=args.fragments)

    print("\n=== Coverage Report ===")
    print(f"Total tokens:      {report.total_tokens:,}")
    print(f"Unique tokens:     {report.unique_tokens:,}")
    print(f"UNK rate:          {report.unk_rate:.2%}")
    print(f"Tokens per word:   {report.tokens_per_word:.2f}")
    print(f"Vocab utilization: {report.vocab_utilization:.2%}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.fragments and args.fragments:
        print("\nTop Fragmented Words:")
        for frag in report.fragments.top_fragmented[:10]:
            print(f"  {frag}")


def analyze_entropy(args):
    """Analyze token entropy distribution."""
    from ..data.tokenizers.analyze import analyze_entropy as do_analyze
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing entropy on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer, top_n=args.top_n)

    print("\n=== Entropy Report ===")
    print(f"Entropy:           {report.entropy:.4f} bits")
    print(f"Perplexity:        {report.perplexity:.2f}")
    print(f"Normalized:        {report.normalized_entropy:.4f}")
    print(f"Uniformity:        {report.uniformity_score:.2%}")
    print(f"Concentration:     {report.concentration_ratio:.2%}")

    if report.distribution:
        print(f"\nTop {len(report.distribution.top_tokens)} tokens:")
        for tok, count in list(report.distribution.top_tokens.items())[:10]:
            print(f"  {tok!r:20} {count:,}")


def analyze_fit_score(args):
    """Calculate tokenizer-dataset fit score."""
    from ..data.tokenizers.analyze import calculate_fit_score
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Calculating fit score on {len(texts)} texts...")
    score = calculate_fit_score(texts, tokenizer)

    print("\n=== Fit Score Report ===")
    print(f"Overall Score:     {score.score:.2f}/100")
    print(f"Grade:             {score.grade}")

    if score.recommendations:
        print("\nRecommendations:")
        for rec in score.recommendations:
            print(f"  - {rec}")

    if score.details:
        print("\nDetails:")
        for key, val in score.details.items():
            print(f"  {key}: {val}")


def analyze_diff(args):
    """Compare tokenization between two tokenizers on a corpus."""
    from ..data.tokenizers.analyze import diff_corpus
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {args.tokenizer1}")
    tok1 = load_tokenizer(args.tokenizer1)
    logger.info(f"Loading tokenizer 2: {args.tokenizer2}")
    tok2 = load_tokenizer(args.tokenizer2)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Comparing tokenization on {len(texts)} texts...")
    diff = diff_corpus(texts, tok1, tok2)

    print("\n=== Corpus Diff Report ===")
    print(f"Texts compared:        {diff.total_texts}")
    print(f"Avg length delta:      {diff.avg_length_delta:+.2f} tokens")
    print(f"Compression improved:  {diff.compression_improvement:.2%}")
    print(f"Tokenizer 1 total:     {diff.tokenizer1_total:,} tokens")
    print(f"Tokenizer 2 total:     {diff.tokenizer2_total:,} tokens")

    if diff.worst_regressions:
        print("\nWorst Regressions (tokenizer 2 is worse):")
        for reg in diff.worst_regressions[:5]:
            print(f"  Delta: {reg.length_delta:+d}, Text: {reg.text[:50]}...")


# === Tokenizer Curriculum Commands ===


def curriculum_length_buckets(args):
    """Create curriculum buckets based on token length."""
    from ..data.tokenizers.curriculum import create_length_buckets, get_curriculum_schedule
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Creating {args.num_buckets} length buckets...")
    buckets = create_length_buckets(texts, tokenizer, num_buckets=args.num_buckets)

    print("\n=== Length Buckets ===")
    for i, bucket in enumerate(buckets):
        print(
            f"Bucket {i + 1}: {bucket.min_tokens}-{bucket.max_tokens} tokens, "
            f"{bucket.sample_count} samples, avg={bucket.avg_length:.1f}"
        )

    if args.schedule:
        schedule = get_curriculum_schedule(texts, tokenizer, num_buckets=args.num_buckets)
        print("\n=== Curriculum Schedule ===")
        print(f"Total phases:    {len(schedule.phases)}")
        print(f"Warmup samples:  {schedule.warmup_samples}")
        print(f"Ramp samples:    {schedule.ramp_samples}")


def curriculum_reasoning_density(args):
    """Score texts by reasoning density for curriculum ordering."""
    from ..data.tokenizers.curriculum import get_difficulty_percentiles, sort_by_reasoning_density
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Scoring reasoning density on {len(texts)} texts...")
    sorted_scores = sort_by_reasoning_density(texts, tokenizer, descending=args.descending)
    percentiles = get_difficulty_percentiles(texts, tokenizer)

    print("\n=== Reasoning Density ===")
    print(f"Mean score:     {percentiles.mean:.4f}")
    print(f"P25:            {percentiles.p25:.4f}")
    print(f"P50 (median):   {percentiles.p50:.4f}")
    print(f"P75:            {percentiles.p75:.4f}")
    print(f"P90:            {percentiles.p90:.4f}")

    print(f"\nTop {min(10, len(sorted_scores))} by reasoning density:")
    for score in sorted_scores[:10]:
        text_preview = texts[score.text_index][:50]
        print(f"  [{score.text_index}] {score.score:.4f}: {text_preview}...")


# === Tokenizer Training Commands ===


def training_throughput(args):
    """Profile tokenization throughput."""
    from ..data.tokenizers.training import ThroughputProfiler
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Profiling throughput on {len(texts)} texts...")
    profiler = ThroughputProfiler(tokenizer)
    metrics = profiler.profile(texts, batch_size=args.batch_size, num_iterations=args.iterations)

    print("\n=== Throughput Profile ===")
    print(f"Tokens/second:     {metrics.tokens_per_second:,.0f}")
    print(f"Texts/second:      {metrics.texts_per_second:,.0f}")
    print(f"Avg batch time:    {metrics.avg_batch_time_ms:.2f} ms")
    print(f"Total tokens:      {metrics.total_tokens:,}")
    print(f"Total time:        {metrics.total_time_seconds:.2f} s")


def training_pack(args):
    """Pack sequences for efficient training."""
    from ..data.tokenizers.training import PackingConfig, pack_sequences
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    config = PackingConfig(
        max_seq_length=args.max_length,
        padding_token_id=tokenizer.pad_token_id or 0,
        separator_token_id=tokenizer.eos_token_id,
    )

    logger.info(f"Packing {len(texts)} sequences to max length {args.max_length}...")
    packed = pack_sequences(texts, tokenizer, config)

    total_tokens = sum(len(p.token_ids) for p in packed)
    efficiency = total_tokens / (len(packed) * args.max_length) if packed else 0

    print("\n=== Packing Results ===")
    print(f"Input sequences:   {len(texts)}")
    print(f"Packed sequences:  {len(packed)}")
    print(f"Packing ratio:     {len(texts) / len(packed):.2f}x" if packed else "N/A")
    print(f"Efficiency:        {efficiency:.2%}")

    if args.output:
        import json

        with open(args.output, "w") as f:
            for p in packed:
                f.write(
                    json.dumps({"token_ids": p.token_ids, "boundaries": p.sequence_boundaries})
                    + "\n"
                )
        print(f"\nSaved to: {args.output}")


# === Tokenizer Regression Commands ===


def regression_run(args):
    """Run token regression tests."""
    from ..data.tokenizers.regression import load_tests_from_yaml, run_token_tests
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    logger.info(f"Loading tests from: {args.tests}")
    suite = load_tests_from_yaml(args.tests)

    logger.info(f"Running {len(suite.tests)} tests...")
    result = run_token_tests(suite, tokenizer)

    print("\n=== Regression Test Results ===")
    print(f"Suite: {suite.name}")
    print(f"Tests: {result.total_tests}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")

    if result.failed > 0:
        print("\nFailed tests:")
        for test_result in result.results:
            if not test_result.passed:
                print(f"  - {test_result.test_name}: {test_result.message}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")


# === Runtime Commands ===


def runtime_registry(args):
    """Display special token registry."""
    from ..data.tokenizers.runtime import (
        SpecialTokenRegistry,
        TokenCategory,
        create_standard_registry,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    if args.standard:
        registry = create_standard_registry()
    else:
        registry = SpecialTokenRegistry()
        if args.tokenizer:
            tokenizer = load_tokenizer(args.tokenizer)
            # Try to populate from tokenizer's special tokens
            if hasattr(tokenizer, "special_tokens_map"):
                for name, token in tokenizer.special_tokens_map.items():
                    if isinstance(token, str):
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        registry.register(
                            token_str=token,
                            token_id=token_id,
                            category=TokenCategory.CONTROL,
                            description=name,
                        )

    print("\n=== Special Token Registry ===")
    print(f"Total tokens: {len(registry.tokens)}")

    for entry in registry.tokens:
        print(f"  {entry.token_id:5d}: {entry.token_str:20s} [{entry.category.value}]")
        if entry.description:
            print(f"         {entry.description}")


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

    # === Analyze subcommands ===
    analyze_parser = tok_subparsers.add_parser("analyze", help="Token analysis tools")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", help="Analysis type")

    # Coverage analysis
    cov_parser = analyze_subparsers.add_parser("coverage", help="Analyze token coverage")
    cov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    cov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    cov_parser.add_argument("--fragments", action="store_true", help="Include fragment analysis")
    cov_parser.set_defaults(func=analyze_coverage)

    # Entropy analysis
    ent_parser = analyze_subparsers.add_parser("entropy", help="Analyze token entropy")
    ent_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    ent_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    ent_parser.add_argument("--top-n", type=int, default=100, help="Top N tokens to show")
    ent_parser.set_defaults(func=analyze_entropy)

    # Fit score
    fit_parser = analyze_subparsers.add_parser("fit-score", help="Calculate tokenizer-dataset fit")
    fit_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    fit_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    fit_parser.set_defaults(func=analyze_fit_score)

    # Diff analysis
    diff_parser = analyze_subparsers.add_parser("diff", help="Compare tokenizers on corpus")
    diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    diff_parser.set_defaults(func=analyze_diff)

    # === Curriculum subcommands ===
    curr_parser = tok_subparsers.add_parser("curriculum", help="Curriculum learning tools")
    curr_subparsers = curr_parser.add_subparsers(dest="curriculum_command", help="Curriculum type")

    # Length buckets
    len_parser = curr_subparsers.add_parser("length-buckets", help="Create length-based curriculum")
    len_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    len_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    len_parser.add_argument("--num-buckets", type=int, default=5, help="Number of buckets")
    len_parser.add_argument("--schedule", action="store_true", help="Show curriculum schedule")
    len_parser.set_defaults(func=curriculum_length_buckets)

    # Reasoning density
    reason_parser = curr_subparsers.add_parser(
        "reasoning-density", help="Score by reasoning density"
    )
    reason_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    reason_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    reason_parser.add_argument(
        "--descending", action="store_true", help="Sort descending (hardest first)"
    )
    reason_parser.set_defaults(func=curriculum_reasoning_density)

    # === Training subcommands ===
    train_tok_parser = tok_subparsers.add_parser("training", help="Training utilities")
    train_tok_subparsers = train_tok_parser.add_subparsers(
        dest="training_command", help="Training tool"
    )

    # Throughput profiling
    thru_parser = train_tok_subparsers.add_parser(
        "throughput", help="Profile tokenization throughput"
    )
    thru_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    thru_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    thru_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    thru_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    thru_parser.set_defaults(func=training_throughput)

    # Sequence packing
    pack_parser = train_tok_subparsers.add_parser("pack", help="Pack sequences for training")
    pack_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    pack_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    pack_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    pack_parser.add_argument("--output", "-o", help="Output file (JSONL)")
    pack_parser.set_defaults(func=training_pack)

    # === Regression subcommands ===
    reg_parser = tok_subparsers.add_parser("regression", help="Token regression testing")
    reg_subparsers = reg_parser.add_subparsers(dest="regression_command", help="Regression tool")

    # Run tests
    run_parser = reg_subparsers.add_parser("run", help="Run regression tests")
    run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    run_parser.add_argument("--tests", required=True, help="Test suite YAML file")
    run_parser.set_defaults(func=regression_run)

    # === Runtime subcommands ===
    runtime_parser = tok_subparsers.add_parser("runtime", help="Runtime token utilities")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", help="Runtime tool")

    # Registry
    registry_parser = runtime_subparsers.add_parser(
        "registry", help="Display special token registry"
    )
    registry_parser.add_argument("--tokenizer", "-t", help="Tokenizer name or path")
    registry_parser.add_argument("--standard", action="store_true", help="Show standard registry")
    registry_parser.set_defaults(func=runtime_registry)

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
