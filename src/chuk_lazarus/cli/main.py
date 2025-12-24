"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"

Data Commands:
    lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl
    lazarus data lengths stats -c lengths.jsonl
    lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/
    lazarus data batchplan info -p batch_plan/
    lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4  # sharded view
    lazarus data batchplan verify -p batch_plan/ -l lengths.jsonl
    lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/  # distributed sharding

Batching Analysis Commands:
    lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512
    lazarus data batching histogram -c lengths.jsonl --bins 20
    lazarus data batching suggest -c lengths.jsonl --goal waste --num-buckets 4
    lazarus data batch generate -p batch_plan/ -d train.jsonl -t gpt2 -o batches/

Tokenizer Commands:
    lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
    lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3"
    lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test" --verbose
    lazarus tokenizer doctor -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t model --save fingerprint.json
    lazarus tokenizer fingerprint -t model --verify fingerprint.json --strict
    lazarus tokenizer benchmark -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer benchmark -t model --samples 5000 --compare
    lazarus tokenizer analyze coverage -t model --file corpus.txt
    lazarus tokenizer analyze entropy -t model --file corpus.txt
    lazarus tokenizer analyze fit-score -t model --file corpus.txt
    lazarus tokenizer analyze efficiency -t model --file corpus.txt
    lazarus tokenizer analyze vocab-suggest -t model --file corpus.txt
    lazarus tokenizer curriculum length-buckets -t model --file corpus.txt
    lazarus tokenizer curriculum reasoning-density -t model --file corpus.txt
    lazarus tokenizer training throughput -t model --file corpus.txt
    lazarus tokenizer training pack -t model --file corpus.txt --max-length 512
    lazarus tokenizer regression run -t model --tests tests.yaml
    lazarus tokenizer research soft-tokens -n 10 -d 768 --prefix task
    lazarus tokenizer research analyze-embeddings -f embeddings.json --cluster
    lazarus tokenizer research morph -f embeddings.json -s 0 -t 1 --method spherical
    lazarus tokenizer instrument histogram -t model --file corpus.txt
    lazarus tokenizer instrument oov -t model --file corpus.txt
    lazarus tokenizer instrument waste -t model --file corpus.txt --max-length 512
    lazarus tokenizer instrument vocab-diff -t1 model1 -t2 model2 --file corpus.txt

Gym Commands (Online Learning):
    lazarus gym run -t gpt2 --mock --num-episodes 10  # Test with mock stream
    lazarus gym run -t gpt2 --host localhost --port 8023  # Connect to puzzle arcade
    lazarus gym run -t gpt2 --mock --output buffer.json  # Save samples to buffer
    lazarus gym info  # Display gym stream configuration

Benchmark Commands:
    lazarus bench  # Run benchmark with synthetic data
    lazarus bench -d train.jsonl -t gpt2  # Benchmark with real dataset
    lazarus bench --bucket-edges 128,256,512 --token-budget 4096  # Custom config

Training with Batching:
    lazarus train sft --model model --data train.jsonl --batchplan batch_plan/
    lazarus train sft --model model --data train.jsonl --bucket-edges 128,256,512 --token-budget 4096
    lazarus train sft --model model --data train.jsonl --pack --pack-max-len 2048
    lazarus train sft --model model --data train.jsonl --online --gym-host localhost --gym-port 8023
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
    from ..data.tokenizers.token_display import TokenDisplayUtility
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {args.tokenizer1}")
    tok1 = load_tokenizer(args.tokenizer1)
    logger.info(f"Loading tokenizer 2: {args.tokenizer2}")
    tok2 = load_tokenizer(args.tokenizer2)

    text = args.text

    ids1 = tok1.encode(text)
    ids2 = tok2.encode(text)

    print(f"\nText: {text}")
    print(f"\n{'=' * 60}")
    print(f"{args.tokenizer1}:")
    print(f"{'=' * 60}")
    print(f"  Token count: {len(ids1)}")
    print(f"  Token IDs: {ids1[:20]}{'...' if len(ids1) > 20 else ''}")

    if args.verbose:
        display1 = TokenDisplayUtility(tok1)
        display1.display_tokens_from_prompt(text, add_special_tokens=False)

    print(f"\n{'=' * 60}")
    print(f"{args.tokenizer2}:")
    print(f"{'=' * 60}")
    print(f"  Token count: {len(ids2)}")
    print(f"  Token IDs: {ids2[:20]}{'...' if len(ids2) > 20 else ''}")

    if args.verbose:
        display2 = TokenDisplayUtility(tok2)
        display2.display_tokens_from_prompt(text, add_special_tokens=False)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    print(f"  Difference: {len(ids1) - len(ids2):+d} tokens")
    print(f"  Ratio: {len(ids1) / len(ids2):.2f}x" if len(ids2) > 0 else "  Ratio: N/A")


def tokenizer_doctor(args):
    """Run comprehensive tokenizer health check."""
    from ..data.tokenizers.fingerprint import compute_fingerprint
    from ..data.tokenizers.runtime.chat_templates import (
        ChatTemplateRegistry,
        patch_chat_template,
        suggest_template_for_model,
        validate_chat_template,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"\n{'=' * 60}")
    print(f"Tokenizer Doctor: {args.tokenizer}")
    print(f"{'=' * 60}")

    issues = []
    warnings = []
    fixes_applied = []

    # === Basic Info ===
    print("\n--- Basic Info ---")
    vocab = tokenizer.get_vocab()
    print(f"  Vocab size: {len(vocab):,}")

    # === Special Tokens ===
    print("\n--- Special Tokens ---")

    special_tokens = {
        "pad_token_id": ("PAD", "Padding"),
        "unk_token_id": ("UNK", "Unknown"),
        "bos_token_id": ("BOS", "Beginning of Sequence"),
        "eos_token_id": ("EOS", "End of Sequence"),
    }

    for attr, (short, desc) in special_tokens.items():
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            # Try to get the token string
            try:
                if hasattr(tokenizer, "convert_ids_to_tokens"):
                    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                else:
                    token_str = tokenizer.decode([token_id])
                print(f"  {short:4s} ({attr}): {token_id} -> {repr(token_str)}")
            except Exception:
                print(f"  {short:4s} ({attr}): {token_id}")
        else:
            msg = f"Missing {short} token ({attr})"
            if short in ("BOS", "EOS"):
                warnings.append(msg)
                print(f"  {short:4s} ({attr}): NOT SET (warning)")
            else:
                print(f"  {short:4s} ({attr}): NOT SET")

    # === Chat Template ===
    print("\n--- Chat Template ---")
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

    # Use new validation system
    validation_result = validate_chat_template(tokenizer)
    registry = ChatTemplateRegistry()

    if has_chat_template:
        template_str = str(tokenizer.chat_template)
        template_preview = template_str[:100]
        print("  Available: Yes")
        print(f"  Preview: {template_preview}...")
        print(f"  Format: {validation_result.format.value}")

        # Show capabilities
        if validation_result.capabilities:
            caps = [c.value for c in validation_result.capabilities]
            print(f"  Capabilities: {', '.join(caps)}")

        # Show any issues/warnings from validation
        for issue in validation_result.issues:
            if issue.severity == "error":
                issues.append(issue.message)
                print(f"  ERROR: {issue.message}")
            elif issue.severity == "warning":
                warnings.append(issue.message)
                print(f"  WARN: {issue.message}")
            else:
                if args.verbose:
                    print(f"  INFO: {issue.message}")

        # Test chat template with various scenarios
        test_scenarios = [
            ("single user", [{"role": "user", "content": "Hello"}]),
            (
                "multi-turn",
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
            ),
        ]

        # Check for system message support
        try:
            system_test = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            result = tokenizer.apply_chat_template(
                system_test, add_generation_prompt=True, tokenize=False
            )
            if "You are helpful" in result:
                print("  System messages: Supported")
            else:
                print("  System messages: May not be rendered")
        except Exception:
            print("  System messages: Not supported")
            warnings.append("System messages not supported by chat template")

        # Run basic tests
        all_pass = True
        for scenario_name, messages in test_scenarios:
            try:
                result = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                if args.verbose:
                    print(f"  Test ({scenario_name}): PASS")
                    print(f"    Output: {result[:80]}...")
            except Exception as e:
                all_pass = False
                issues.append(f"Chat template error ({scenario_name}): {e}")
                print(f"  Test ({scenario_name}): FAIL - {e}")

        if all_pass:
            print("  Tests: All PASS")
    else:
        warnings.append("No chat template defined")
        print("  Available: No")

        # Suggest a template based on model name
        suggested = suggest_template_for_model(args.tokenizer)
        if suggested:
            print(f"  Suggested format: {suggested.format.value}")
            print(f"  Description: {suggested.description}")

        # Handle --fix mode for missing template
        if getattr(args, "fix", False):
            template_format = getattr(args, "format", None)
            try:
                success = patch_chat_template(tokenizer, template_format=template_format)
                if success:
                    detected_format = registry.detect_format(tokenizer.chat_template)
                    fixes_applied.append(f"Added {detected_format.value} chat template")
                    print(f"  FIX APPLIED: Added {detected_format.value} chat template")
                else:
                    print("  FIX FAILED: Could not determine appropriate template")
                    print("  Hint: Use --format to specify (chatml, llama, phi, gemma, etc.)")
            except Exception as e:
                print(f"  FIX FAILED: {e}")
        else:
            print("  Recommendation: Add a chat template for conversational use")
            print("  Use: lazarus tokenizer doctor -t MODEL --fix")

            # Check if tokenizer_config.json exists and might need patching
            if hasattr(tokenizer, "name_or_path"):
                print(f"  Model path: {tokenizer.name_or_path}")

    # === Encode/Decode Roundtrip ===
    print("\n--- Encode/Decode Roundtrip ---")
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Special chars: @#$%^&*()",
        "Unicode: 你好 🎉",
        "Numbers: 12345 3.14159",
    ]

    roundtrip_issues = 0
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            # Normalize whitespace for comparison
            normalized_original = " ".join(text.split())
            normalized_decoded = " ".join(decoded.split())

            if normalized_original != normalized_decoded:
                roundtrip_issues += 1
                if args.verbose:
                    print(f"  WARN: '{text[:30]}...' -> '{decoded[:30]}...'")
        except Exception as e:
            roundtrip_issues += 1
            issues.append(f"Encode/decode error for '{text[:20]}...': {e}")

    if roundtrip_issues == 0:
        print(f"  All {len(test_texts)} tests: PASS")
    else:
        print(f"  Tests: {len(test_texts) - roundtrip_issues}/{len(test_texts)} PASS")
        warnings.append(f"{roundtrip_issues} roundtrip tests had differences")

    # === Fingerprint ===
    print("\n--- Fingerprint ---")
    try:
        fp = compute_fingerprint(tokenizer)
        print(f"  Fingerprint: {fp.fingerprint}")
        print(f"  Vocab hash:  {fp.vocab_hash}")
        if args.verbose:
            print(f"  Full hash:   {fp.full_hash}")
    except Exception as e:
        issues.append(f"Fingerprint error: {e}")
        print(f"  Error: {e}")

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("Diagnosis:")
    print(f"{'=' * 60}")

    if fixes_applied:
        print(f"  Fixes Applied: {len(fixes_applied)}")
        for fix in fixes_applied:
            print(f"    FIXED: {fix}")

    if not issues and not warnings:
        print("  Status: HEALTHY")
        print("  No issues found.")
    else:
        if issues:
            print(f"  Status: ISSUES FOUND ({len(issues)})")
            for issue in issues:
                print(f"    ERROR: {issue}")
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"    WARN: {warning}")

    # Save patched tokenizer if --fix and --output specified
    if getattr(args, "fix", False) and fixes_applied:
        output_path = getattr(args, "output", None)
        if output_path:
            try:
                import os

                os.makedirs(output_path, exist_ok=True)
                tokenizer.save_pretrained(output_path)
                print(f"\n  Saved patched tokenizer to: {output_path}")
            except Exception as e:
                print(f"\n  ERROR: Could not save tokenizer: {e}")
        else:
            print("\n  Note: Use --output PATH to save the patched tokenizer")

    if issues:
        sys.exit(1)


def tokenizer_fingerprint(args):
    """Generate or verify tokenizer fingerprint."""
    from ..data.tokenizers.fingerprint import (
        compute_fingerprint,
        load_fingerprint,
        save_fingerprint,
        verify_fingerprint,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Compute fingerprint
    fp = compute_fingerprint(tokenizer)

    if args.verify:
        # Verify against expected fingerprint
        logger.info(f"Verifying against: {args.verify}")

        if args.verify.endswith(".json"):
            expected = load_fingerprint(args.verify)
        else:
            expected = args.verify  # Treat as fingerprint string

        mismatch = verify_fingerprint(tokenizer, expected, strict=args.strict)

        print(f"\n{'=' * 60}")
        print("Fingerprint Verification")
        print(f"{'=' * 60}")
        print(f"  Tokenizer: {args.tokenizer}")
        print(f"  Actual:    {fp.fingerprint}")

        if isinstance(expected, str):
            print(f"  Expected:  {expected}")
        else:
            print(f"  Expected:  {expected.fingerprint}")

        if mismatch is None:
            print("\n  Result: MATCH")
        else:
            print("\n  Result: MISMATCH")
            print(f"  Compatible: {'Yes' if mismatch.is_compatible else 'No'}")
            if mismatch.warnings:
                print("\n  Warnings:")
                for w in mismatch.warnings:
                    print(f"    - {w}")

            if not mismatch.is_compatible:
                sys.exit(1)

    elif args.save:
        # Save fingerprint to file
        save_fingerprint(fp, args.save)
        print(f"\n{'=' * 60}")
        print("Fingerprint Saved")
        print(f"{'=' * 60}")
        print(f"  Tokenizer:   {args.tokenizer}")
        print(f"  Fingerprint: {fp.fingerprint}")
        print(f"  Saved to:    {args.save}")

    else:
        # Just display fingerprint
        print(f"\n{'=' * 60}")
        print("Tokenizer Fingerprint")
        print(f"{'=' * 60}")
        print(f"  Tokenizer:     {args.tokenizer}")
        print(f"  Fingerprint:   {fp.fingerprint}")
        print(f"  Full hash:     {fp.full_hash}")
        print(f"  Vocab size:    {fp.vocab_size:,}")
        print(f"  Vocab hash:    {fp.vocab_hash}")
        print(f"  Special hash:  {fp.special_tokens_hash}")
        print(f"  Merges hash:   {fp.merges_hash}")

        print("\n  Special tokens:")
        for name, token_id in fp.special_tokens.items():
            print(f"    {name}: {token_id}")


def tokenizer_benchmark(args):
    """Benchmark tokenizer throughput."""
    from ..data.tokenizers.backends.benchmark import (
        benchmark_tokenizer,
        compare_backends,
        generate_benchmark_corpus,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Generate or load corpus
    if args.file:
        logger.info(f"Loading corpus from: {args.file}")
        with open(args.file) as f:
            corpus = [line.strip() for line in f if line.strip()]
        if args.samples and len(corpus) > args.samples:
            corpus = corpus[: args.samples]
    else:
        logger.info(f"Generating synthetic corpus ({args.samples} samples)...")
        corpus = generate_benchmark_corpus(
            num_samples=args.samples,
            avg_length=args.avg_length,
            seed=args.seed,
        )

    print(f"\n{'=' * 60}")
    print("Tokenizer Benchmark")
    print(f"{'=' * 60}")
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  Samples:    {len(corpus):,}")
    print(f"  Avg length: ~{sum(len(t.split()) for t in corpus) // len(corpus)} words")
    print(f"  Workers:    {args.workers}")
    print()

    if args.compare:
        # Compare HuggingFace vs Fast backend
        logger.info("Running backend comparison...")
        comparison = compare_backends(
            tokenizer,
            corpus,
            num_workers=args.workers,
            add_special_tokens=args.special_tokens,
        )
        print(comparison.summary())
    else:
        # Single backend benchmark
        logger.info("Running benchmark...")
        result = benchmark_tokenizer(
            tokenizer,
            corpus,
            num_workers=args.workers,
            add_special_tokens=args.special_tokens,
            warmup_samples=min(args.warmup, len(corpus)),
        )

        print("Results:")
        print(f"  Backend:      {result.backend_type}")
        print(f"  Total tokens: {result.total_tokens:,}")
        print(f"  Time:         {result.elapsed_seconds:.2f}s")
        print(f"  Throughput:   {result.tokens_per_second:,.0f} tokens/sec")
        print(f"  Samples/sec:  {result.samples_per_second:,.1f}")
        print(f"  Avg tok/sample: {result.avg_tokens_per_sample:.1f}")
        print(f"{'=' * 60}")


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


def analyze_efficiency(args):
    """Analyze token efficiency metrics."""
    from ..data.tokenizers.analyze import analyze_efficiency as do_analyze
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing efficiency on {len(texts)} texts...")
    report = do_analyze(texts, tokenizer)

    print("\n=== Efficiency Report ===")
    print(f"Efficiency Score:  {report.efficiency_score:.1f}/100")

    print("\n--- Sample Statistics ---")
    print(f"Samples:           {report.sample_stats.count:,}")
    print(f"Total tokens:      {report.sample_stats.total_tokens:,}")
    print(f"Mean tokens:       {report.sample_stats.mean:.1f}")
    print(f"Median tokens:     {report.sample_stats.median:.1f}")
    print(f"Std dev:           {report.sample_stats.std:.1f}")
    print(f"P5/P95:            {report.sample_stats.p5:.0f} / {report.sample_stats.p95:.0f}")
    print(f"Min/Max:           {report.sample_stats.min_tokens} / {report.sample_stats.max_tokens}")

    if report.reasoning_steps:
        print("\n--- Reasoning Steps ---")
        print(f"Count:             {report.reasoning_steps.count}")
        print(f"Mean tokens:       {report.reasoning_steps.mean_tokens:.1f}")

    if report.equations:
        print("\n--- Equations ---")
        print(f"Count:             {report.equations.count}")
        print(f"Mean tokens:       {report.equations.mean_tokens:.1f}")

    if report.tool_calls:
        print("\n--- Tool Calls ---")
        print(f"Count:             {report.tool_calls.count}")
        print(f"Mean tokens:       {report.tool_calls.mean_tokens:.1f}")

    print("\n--- Fragmentation ---")
    print(f"Score:             {report.fragmentation.fragmentation_score:.1%}")
    print(f"Single-char:       {report.fragmentation.single_char_tokens:,}")
    print(f"Subword:           {report.fragmentation.subword_tokens:,}")

    if report.fragmentation.fragmented_words:
        print("\nMost fragmented words:")
        for word in report.fragmentation.fragmented_words[:5]:
            print(f"  {word['word']}: {word['tokens']} tokens")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")


def analyze_vocab_suggest(args):
    """Suggest vocabulary additions based on corpus analysis."""
    from ..data.tokenizers.analyze import InductionConfig, analyze_vocab_induction
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    config = InductionConfig(
        min_frequency=args.min_freq,
        min_fragmentation=args.min_frag,
        max_candidates=args.limit,
    )

    logger.info(f"Analyzing vocabulary on {len(texts)} texts...")
    report = analyze_vocab_induction(texts, tokenizer, config)

    print("\n=== Vocabulary Induction Report ===")
    print(f"Candidates found:     {report.total_candidates}")
    print(f"Potential savings:    {report.total_potential_savings:,} tokens")
    print(f"Savings percent:      {report.savings_percent:.1f}%")

    if report.domain_breakdown:
        print("\nBy domain:")
        for domain, count in sorted(report.domain_breakdown.items()):
            print(f"  {domain}: {count}")

    print(f"\nTop {min(args.show, len(report.candidates))} candidates:")
    print("-" * 70)
    print(f"{'Token':<30} {'Freq':>8} {'Tokens':>8} {'Savings':>10}")
    print("-" * 70)

    for c in report.candidates[: args.show]:
        token_display = repr(c.token_str)[:28]
        print(f"{token_display:<30} {c.frequency:>8} {c.current_tokens:>8} {c.total_savings:>10}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")


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


# === Research Commands ===


def research_soft_tokens(args):
    """Create and display soft token bank."""
    from ..data.tokenizers.research import (
        InitializationMethod,
        create_prompt_tuning_bank,
    )

    init_method = InitializationMethod(args.init_method)

    bank = create_prompt_tuning_bank(
        num_tokens=args.num_tokens,
        embedding_dim=args.embedding_dim,
        prefix=args.prefix,
        init_method=init_method,
        init_std=args.init_std,
    )

    print("\n=== Soft Token Bank ===")
    print(f"Name:           {bank.name}")
    print(f"Embedding dim:  {bank.embedding_dim}")
    print(f"Num tokens:     {len(bank.tokens)}")
    print(f"Init method:    {init_method.value}")
    print("\nTokens:")

    import numpy as np

    for token in bank.tokens:
        emb = token.embedding_array
        norm = np.linalg.norm(emb)
        print(f"  {token.token.name} (ID: {token.token.token_id})")
        print(f"    Norm: {norm:.4f}, Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")

    if args.output:
        import json

        output_data = {
            "name": bank.name,
            "embedding_dim": bank.embedding_dim,
            "tokens": [
                {
                    "name": t.token.name,
                    "token_id": t.token.token_id,
                    "embedding": t.embedding,
                }
                for t in bank.tokens
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


def research_analyze_embeddings(args):
    """Analyze embedding space from a file."""
    import json

    import numpy as np

    from ..data.tokenizers.research import (
        analyze_embeddings,
        cluster_tokens,
        project_embeddings,
    )

    # Load embeddings from file
    logger.info(f"Loading embeddings from: {args.file}")
    with open(args.file) as f:
        data = json.load(f)

    if "embeddings" in data:
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        token_ids = data.get("token_ids", list(range(len(embeddings))))
        token_strs = data.get("token_strs", [f"token_{i}" for i in range(len(embeddings))])
    else:
        logger.error("File must contain 'embeddings' key")
        return

    print("\n=== Embedding Analysis ===")
    analysis = analyze_embeddings(embeddings, num_clusters=args.num_clusters)

    print(f"Num tokens:      {analysis.num_tokens}")
    print(f"Embedding dim:   {analysis.embedding_dim}")
    print(f"Mean norm:       {analysis.mean_norm:.4f}")
    print(f"Norm std:        {analysis.std_norm:.4f}")
    print(f"Isotropy:        {analysis.isotropy_score:.4f}")
    print(f"Mean similarity: {analysis.mean_pairwise_similarity:.4f}")
    if analysis.silhouette_score is not None:
        print(f"Silhouette:      {analysis.silhouette_score:.4f}")

    if args.cluster:
        print(f"\n=== Clustering ({args.num_clusters} clusters) ===")
        clusters = cluster_tokens(embeddings, token_ids, token_strs, args.num_clusters)
        for c in clusters:
            sample = c.token_strs[:3]
            print(f"  Cluster {c.cluster_id}: {c.size} tokens")
            print(f"    Intra-dist: {c.intra_cluster_distance:.4f}")
            print(f"    Sample: {sample}")

    if args.project:
        print("\n=== 2D Projection ===")
        projection = project_embeddings(embeddings, token_ids, token_strs, dim=2)
        print(f"Variance explained: {sum(projection.explained_variance_ratio):.2%}")
        coords = projection.get_coordinates_array()
        print(f"X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
        print(f"Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")


def research_morph(args):
    """Morph between token embeddings."""
    import json

    import numpy as np

    from ..data.tokenizers.research import (
        MorphConfig,
        MorphMethod,
        compute_path_length,
        compute_straightness,
        morph_token,
    )

    # Load embeddings
    with open(args.file) as f:
        data = json.load(f)

    embeddings = np.array(data["embeddings"], dtype=np.float32)
    token_strs = data.get("token_strs", [f"token_{i}" for i in range(len(embeddings))])

    if args.source >= len(embeddings) or args.target >= len(embeddings):
        logger.error(f"Source/target index out of range (max: {len(embeddings) - 1})")
        return

    method = MorphMethod(args.method)
    config = MorphConfig(
        method=method,
        num_steps=args.steps,
        include_endpoints=True,
        normalize_output=args.normalize,
    )

    source_emb = embeddings[args.source]
    target_emb = embeddings[args.target]

    result = morph_token(
        source_emb,
        target_emb,
        token_strs[args.source],
        token_strs[args.target],
        config,
    )

    print("\n=== Token Morphing ===")
    print(f"Source:      {result.source_token}")
    print(f"Target:      {result.target_token}")
    print(f"Method:      {result.method.value}")
    print(f"Steps:       {result.num_steps}")
    print(f"Path length: {compute_path_length(result):.4f}")
    print(f"Straightness: {compute_straightness(result):.4f}")

    trajectory = result.get_embeddings_array()
    print("\nTrajectory norms:")
    for i, alpha in enumerate(result.alphas):
        norm = np.linalg.norm(trajectory[i])
        print(f"  alpha={alpha:.2f}: norm={norm:.4f}")

    if args.output:
        output_data = {
            "source": result.source_token,
            "target": result.target_token,
            "method": result.method.value,
            "alphas": result.alphas,
            "embeddings": result.embeddings,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved trajectory to: {args.output}")


# === Instrumentation Commands ===


def instrument_histogram(args):
    """Display token length histogram."""
    from ..data.tokenizers.instrumentation import (
        compute_length_histogram,
        format_histogram_ascii,
        get_length_stats,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Computing histogram for {len(texts)} texts...")

    if args.quick:
        stats = get_length_stats(texts, tokenizer)
        print("\n=== Quick Length Stats ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    else:
        histogram = compute_length_histogram(texts, tokenizer, num_bins=args.bins)
        print()
        print(format_histogram_ascii(histogram, width=args.width))


def instrument_oov(args):
    """Analyze OOV and rare tokens."""
    from ..data.tokenizers.instrumentation import (
        analyze_oov,
        find_rare_tokens,
        get_frequency_bands,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing OOV on {len(texts)} texts...")

    # Frequency bands
    bands = get_frequency_bands(texts, tokenizer)
    print("\n=== Token Frequency Bands ===")
    for band, count in sorted(bands.items(), key=lambda x: x[0].value):
        print(f"  {band.value:15s}: {count:,} tokens")

    # OOV report
    report = analyze_oov(texts, tokenizer, vocab_size=args.vocab_size)
    print("\n=== OOV Report ===")
    print(f"  Total tokens:      {report.total_tokens:,}")
    print(f"  Unique tokens:     {report.unique_tokens:,}")
    print(f"  UNK rate:          {report.unk_rate:.2%}")
    print(f"  Singleton rate:    {report.singleton_rate:.2%}")
    print(f"  Vocab utilization: {report.vocab_utilization:.2%}")

    if report.recommendations:
        print("\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

    # Rare tokens
    if args.show_rare:
        rare = find_rare_tokens(texts, tokenizer, max_frequency=args.max_freq, top_k=args.top_k)
        print(f"\n=== Rare Tokens (freq ≤ {args.max_freq}) ===")
        for token in rare:
            print(f"  {token.token_str!r:20s}: {token.count:4d}x ({token.band.value})")


def instrument_waste(args):
    """Analyze padding and truncation waste."""
    from ..data.tokenizers.instrumentation import analyze_waste
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Analyzing waste on {len(texts)} texts with max_length={args.max_length}...")

    report = analyze_waste(texts, tokenizer, max_length=args.max_length)

    print("\n=== Token Waste Report ===")
    print(f"  Max length:        {report.max_length}")
    print(f"  Total samples:     {report.total_samples}")
    print(f"  Overall efficiency: {report.overall_efficiency:.1%}")

    print("\n--- Padding Analysis ---")
    print(f"  Total positions:   {report.padding.total_positions:,}")
    print(f"  Content tokens:    {report.padding.total_content_tokens:,}")
    print(f"  Padding tokens:    {report.padding.total_padding_tokens:,}")
    print(f"  Padding rate:      {report.padding.padding_rate:.1%}")
    print(f"  Efficiency:        {report.padding.efficiency:.1%}")
    print(f"  Mean padding:      {report.padding.mean_padding_per_sample:.1f}")
    print(f"  Max padding:       {report.padding.max_padding}")

    print("\n--- Truncation Analysis ---")
    print(
        f"  Truncated samples: {report.truncation.truncated_samples}/{report.truncation.total_samples}"
    )
    print(f"  Truncation rate:   {report.truncation.truncation_rate:.1%}")
    print(f"  Tokens lost:       {report.truncation.total_tokens_lost:,}")
    print(f"  Content loss rate: {report.truncation.content_loss_rate:.1%}")
    print(f"  Minor truncation:  {report.truncation.minor_truncation}")
    print(f"  Major truncation:  {report.truncation.major_truncation}")
    print(f"  Severe truncation: {report.truncation.severe_truncation}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")


def instrument_vocab_diff(args):
    """Compare two tokenizers on a corpus."""
    from ..data.tokenizers.instrumentation import (
        compare_vocab_impact,
        estimate_retokenization_cost,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer 1: {args.tokenizer1}")
    tok1 = load_tokenizer(args.tokenizer1)
    logger.info(f"Loading tokenizer 2: {args.tokenizer2}")
    tok2 = load_tokenizer(args.tokenizer2)

    texts = _load_texts(args)
    if not texts:
        logger.error("No texts provided")
        return

    logger.info(f"Comparing tokenizers on {len(texts)} texts...")

    report = compare_vocab_impact(
        texts,
        tok1,
        tok2,
        tokenizer1_name=args.tokenizer1,
        tokenizer2_name=args.tokenizer2,
        max_examples=args.examples,
    )

    print("\n=== Vocabulary Comparison ===")
    print(f"  Tokenizer 1:       {report.tokenizer1_name}")
    print(f"  Tokenizer 2:       {report.tokenizer2_name}")
    print(f"  Vocab size 1:      {report.tokenizer1_vocab_size:,}")
    print(f"  Vocab size 2:      {report.tokenizer2_vocab_size:,}")

    print("\n--- Token Counts ---")
    print(f"  Tokens (tok1):     {report.tokens1_total:,}")
    print(f"  Tokens (tok2):     {report.tokens2_total:,}")
    print(f"  Difference:        {report.token_count_diff:+,}")
    print(f"  Token ratio:       {report.token_count_ratio:.2f}x")

    print("\n--- Compression ---")
    print(f"  Chars/token (1):   {report.chars_per_token1:.2f}")
    print(f"  Chars/token (2):   {report.chars_per_token2:.2f}")
    print(f"  Compression impr:  {report.compression_improvement:.2f}x")

    print("\n--- Per-Sample Analysis ---")
    print(f"  Improved:          {report.samples_improved}")
    print(f"  Same:              {report.samples_same}")
    print(f"  Worse:             {report.samples_worse}")
    print(f"  Improvement rate:  {report.improvement_rate:.1%}")

    print("\n--- Training Impact ---")
    print(f"  Training speedup:  {report.training_speedup:.2f}x")
    print(f"  Memory reduction:  {report.memory_reduction:.1%}")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Retokenization cost
    if args.cost:
        cost = estimate_retokenization_cost(texts, tok1, tok2)
        print("\n=== Retokenization Cost ===")
        print(
            f"  Vocab overlap:     {cost['vocab_overlap']:,} tokens ({cost['vocab_overlap_rate']:.1%})"
        )
        print(f"  New tokens:        {cost['new_tokens']:,}")
        print(f"  Removed tokens:    {cost['removed_tokens']:,}")
        print(f"  Embedding reuse:   {cost['embedding_reuse_rate']:.1%}")


# === Runtime Commands ===


# === Data Batching Commands ===


def data_lengths_build(args):
    """Build a length cache from a dataset."""
    import asyncio
    import json
    from pathlib import Path

    from ..data.batching import LengthCache
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Compute tokenizer hash for cache invalidation
    try:
        from ..data.tokenizers.fingerprint import compute_fingerprint

        fp = compute_fingerprint(tokenizer)
        tokenizer_hash = fp.fingerprint
    except Exception:
        tokenizer_hash = "unknown"

    logger.info(f"Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        if args.dataset.endswith(".jsonl"):
            samples = [json.loads(line) for line in f if line.strip()]
        else:
            samples = json.load(f)

    async def build_cache():
        output_path = Path(args.output)
        async with LengthCache.create(output_path, tokenizer_hash) as cache:
            for i, sample in enumerate(samples):
                # Get sample ID
                sample_id = sample.get("id") or sample.get("sample_id") or f"sample_{i:06d}"

                # Get text to tokenize
                text = sample.get("text") or sample.get("content") or sample.get("input")
                if text is None and "messages" in sample:
                    # Chat format - concatenate messages
                    text = " ".join(m.get("content", "") for m in sample["messages"])

                if text:
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    await cache.add(sample_id, len(token_ids))

                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

        return cache

    cache = asyncio.run(build_cache())

    print(f"\n{'=' * 60}")
    print("Length Cache Built")
    print(f"{'=' * 60}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Tokenizer:     {args.tokenizer}")
    print(f"  Samples:       {len(cache):,}")
    print(f"  Output:        {args.output}")
    print(f"  Tokenizer hash: {tokenizer_hash}")


def data_lengths_stats(args):
    """Show statistics for a length cache."""
    import asyncio
    from pathlib import Path

    from ..data.batching import LengthCache

    async def load_and_stats():
        cache = await LengthCache.load(Path(args.cache))
        return cache

    cache = asyncio.run(load_and_stats())
    lengths = cache.get_all()

    if not lengths:
        print("Cache is empty")
        return

    values = list(lengths.values())
    values.sort()

    print(f"\n{'=' * 60}")
    print("Length Cache Statistics")
    print(f"{'=' * 60}")
    print(f"  Cache file:    {args.cache}")
    print(f"  Tokenizer:     {cache.tokenizer_hash}")
    print(f"  Total samples: {len(lengths):,}")
    print(f"  Total tokens:  {sum(values):,}")
    print()
    print(f"  Min length:    {min(values)}")
    print(f"  Max length:    {max(values)}")
    print(f"  Mean length:   {sum(values) / len(values):.1f}")
    print(f"  Median:        {values[len(values) // 2]}")

    # Percentiles
    def percentile(p):
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    print()
    print(f"  P10:           {percentile(10)}")
    print(f"  P25:           {percentile(25)}")
    print(f"  P50:           {percentile(50)}")
    print(f"  P75:           {percentile(75)}")
    print(f"  P90:           {percentile(90)}")
    print(f"  P95:           {percentile(95)}")
    print(f"  P99:           {percentile(99)}")


def data_batchplan_build(args):
    """Build a batch plan from length cache."""
    import asyncio
    from pathlib import Path

    from ..data.batching import (
        BatchingConfig,
        BatchPlanBuilder,
        LengthCache,
        save_batch_plan,
    )

    async def build_plan():
        # Load length cache
        logger.info(f"Loading length cache: {args.lengths}")
        cache = await LengthCache.load(Path(args.lengths))
        lengths = cache.get_all()

        # Parse bucket edges
        bucket_edges = tuple(int(x.strip()) for x in args.bucket_edges.split(","))

        # Create config
        if args.predictable:
            config = BatchingConfig.predictable(
                token_budget=args.token_budget,
                bucket_edges=bucket_edges,
                overflow_max=args.overflow_max,
                seed=args.seed,
            )
        else:
            config = BatchingConfig.throughput(
                token_budget=args.token_budget,
                bucket_edges=bucket_edges,
                overflow_max=args.overflow_max,
            )

        # Build plan
        logger.info(f"Building batch plan for {args.epochs} epochs...")
        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash=args.dataset_hash or "unknown",
            tokenizer_hash=cache.tokenizer_hash,
        )

        plan = await builder.build(num_epochs=args.epochs)

        # Save plan
        output_path = Path(args.output)
        save_batch_plan(plan, output_path)

        return plan, output_path

    plan, output_path = asyncio.run(build_plan())

    print(f"\n{'=' * 60}")
    print("Batch Plan Built")
    print(f"{'=' * 60}")
    print(f"  Lengths cache: {args.lengths}")
    print(f"  Epochs:        {plan.num_epochs}")
    print(f"  Token budget:  {args.token_budget}")
    print(f"  Mode:          {'predictable' if args.predictable else 'throughput'}")
    print()
    print(f"  Total batches: {plan.total_microbatches}")
    print(f"  Fingerprint:   {plan.fingerprint}")
    print()
    print(f"  Output:        {output_path}")

    # Per-epoch summary
    print("\n  Per-epoch details:")
    for ep in range(plan.num_epochs):
        epoch_plan = plan.get_epoch(ep)
        print(
            f"    Epoch {ep}: {epoch_plan.num_microbatches} batches, "
            f"{epoch_plan.total_samples} samples, {epoch_plan.total_tokens:,} tokens"
        )


def data_batchplan_info(args):
    """Show information about a batch plan."""
    from pathlib import Path

    from ..data.batching import load_batch_plan

    plan = load_batch_plan(Path(args.plan))

    # Apply sharding if requested
    if args.rank is not None and args.world_size is not None:
        if args.rank >= args.world_size or args.rank < 0:
            print(f"Error: rank must be in range [0, {args.world_size})")
            return
        plan = plan.shard(args.rank, args.world_size)
        shard_info = f" (rank {args.rank}/{args.world_size})"
    else:
        shard_info = ""

    print(f"\n{'=' * 60}")
    print(f"Batch Plan Info{shard_info}")
    print(f"{'=' * 60}")
    print(f"  Plan path:     {args.plan}")
    print(f"  Fingerprint:   {plan.fingerprint}")
    print(f"  Created:       {plan.meta.created_at}")
    print()
    print(f"  Dataset hash:  {plan.meta.dataset_hash}")
    print(f"  Tokenizer:     {plan.meta.tokenizer_hash}")
    print(f"  Token budget:  {plan.meta.token_budget}")
    print(f"  Bucket edges:  {plan.meta.bucket_edges}")
    print()
    print(f"  Epochs:        {plan.num_epochs}")
    print(f"  Total batches: {plan.total_microbatches}")

    # Per-epoch summary
    print("\n  Per-epoch details:")
    for ep in range(plan.num_epochs):
        epoch_plan = plan.get_epoch(ep)
        print(
            f"    Epoch {ep}: {epoch_plan.num_microbatches} batches, "
            f"{epoch_plan.total_samples} samples, {epoch_plan.total_tokens:,} tokens"
        )

    # Sample batches
    if args.show_batches:
        print("\n  Sample batches from epoch 0:")
        epoch0 = plan.get_epoch(0)
        for i, mb in enumerate(epoch0.microbatches[: args.show_batches]):
            print(
                f"    Batch {i}: {mb.batch_size} samples, bucket={mb.bucket_id}, max_len={mb.max_len}"
            )


def data_batchplan_verify(args):
    """Verify a batch plan can be reproduced."""
    import asyncio
    from pathlib import Path

    from ..data.batching import (
        BatchPlanBuilder,
        LengthCache,
        load_batch_plan,
    )

    async def verify():
        # Load original plan
        logger.info(f"Loading batch plan: {args.plan}")
        original = load_batch_plan(Path(args.plan))

        # Rebuild from lengths
        logger.info(f"Rebuilding from lengths: {args.lengths}")
        cache = await LengthCache.load(Path(args.lengths))
        lengths = cache.get_all()

        # Recreate config from plan meta
        from ..data.batching import BatchingConfig, BatchingMode, PadPolicy

        config = BatchingConfig(
            mode=BatchingMode(original.meta.mode),
            pad_policy=PadPolicy(original.meta.pad_policy),
            token_budget=original.meta.token_budget,
            bucket_edges=tuple(original.meta.bucket_edges),
            overflow_max=original.meta.overflow_max,
            seed=original.meta.seed,
        )

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash=original.meta.dataset_hash,
            tokenizer_hash=original.meta.tokenizer_hash,
        )

        rebuilt = await builder.build(num_epochs=original.num_epochs)

        return original, rebuilt

    original, rebuilt = asyncio.run(verify())

    print(f"\n{'=' * 60}")
    print("Batch Plan Verification")
    print(f"{'=' * 60}")
    print(f"  Original fingerprint: {original.fingerprint}")
    print(f"  Rebuilt fingerprint:  {rebuilt.fingerprint}")

    if original.fingerprint == rebuilt.fingerprint:
        print("\n  Result: MATCH")
        print("  The batch plan is reproducible.")
    else:
        print("\n  Result: MISMATCH")
        print("  Warning: Rebuilt plan differs from original!")

        # Check epoch-by-epoch
        for ep in range(original.num_epochs):
            orig_mbs = list(original.iter_epoch(ep))
            rebuilt_mbs = list(rebuilt.iter_epoch(ep))

            if len(orig_mbs) != len(rebuilt_mbs):
                print(
                    f"    Epoch {ep}: batch count differs ({len(orig_mbs)} vs {len(rebuilt_mbs)})"
                )
            else:
                matches = sum(1 for o, r in zip(orig_mbs, rebuilt_mbs) if o.samples == r.samples)
                print(f"    Epoch {ep}: {matches}/{len(orig_mbs)} batches match")

        sys.exit(1)


def data_batchplan_shard(args):
    """Save sharded batch plans for distributed training."""
    from pathlib import Path

    from ..data.batching import load_batch_plan, save_batch_plan

    # Load original plan
    logger.info(f"Loading batch plan: {args.plan}")
    plan = load_batch_plan(Path(args.plan))

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Batch Plan Sharding")
    print(f"{'=' * 60}")
    print(f"  Source plan:   {args.plan}")
    print(f"  World size:    {args.world_size}")
    print(f"  Total batches: {plan.total_microbatches}")
    print()

    # Create sharded plans
    for rank in range(args.world_size):
        sharded = plan.shard(rank, args.world_size)
        shard_path = output_base / f"rank_{rank}"
        save_batch_plan(sharded, shard_path)

        print(f"  Rank {rank}: {sharded.total_microbatches} batches -> {shard_path}")

    print()
    print(f"  Output:        {output_base}")


def data_batching_analyze(args):
    """Analyze batching efficiency for a dataset."""
    import asyncio
    from pathlib import Path

    from ..data.batching import (
        BucketSpec,
        LengthCache,
        create_efficiency_report,
    )

    async def analyze():
        # Load length cache
        logger.info(f"Loading length cache: {args.cache}")
        cache = await LengthCache.load(Path(args.cache))
        lengths = cache.get_all()

        # Parse bucket edges
        bucket_edges = tuple(int(x.strip()) for x in args.bucket_edges.split(","))

        # Create bucket spec
        bucket_spec = BucketSpec(
            edges=bucket_edges,
            overflow_max=args.overflow_max,
        )

        # Create efficiency report
        report = create_efficiency_report(lengths, bucket_spec)
        return report

    report = asyncio.run(analyze())

    # Print report
    print(report.to_ascii())

    if args.output:
        # Save JSON report
        import json

        with open(args.output, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


def data_batching_histogram(args):
    """Display length histogram for a dataset."""
    import asyncio
    from pathlib import Path

    from ..data.batching import (
        LengthCache,
        compute_length_histogram,
    )

    async def load():
        cache = await LengthCache.load(Path(args.cache))
        return cache.get_all()

    lengths = asyncio.run(load())
    histogram = compute_length_histogram(lengths, num_bins=args.bins)

    print(histogram.to_ascii(width=args.width))

    print("\n--- Percentiles ---")
    print(f"  P25: {histogram.p25}")
    print(f"  P50: {histogram.p50}")
    print(f"  P75: {histogram.p75}")
    print(f"  P90: {histogram.p90}")
    print(f"  P95: {histogram.p95}")
    print(f"  P99: {histogram.p99}")


def data_batching_suggest(args):
    """Suggest optimal bucket edges for a dataset."""
    import asyncio
    from pathlib import Path

    from ..data.batching import (
        LengthCache,
        OptimizationGoal,
        suggest_bucket_edges,
    )

    async def load():
        cache = await LengthCache.load(Path(args.cache))
        return cache.get_all()

    lengths = asyncio.run(load())

    # Get goal
    goal_map = {
        "waste": OptimizationGoal.MINIMIZE_WASTE,
        "balance": OptimizationGoal.BALANCE_BUCKETS,
        "memory": OptimizationGoal.MINIMIZE_MEMORY,
    }
    goal = goal_map.get(args.goal, OptimizationGoal.MINIMIZE_WASTE)

    suggestion = suggest_bucket_edges(
        lengths,
        num_buckets=args.num_buckets,
        goal=goal,
        max_length=args.max_length,
    )

    print(f"\n{'=' * 60}")
    print("Bucket Edge Suggestions")
    print(f"{'=' * 60}")
    print(f"  Goal:           {suggestion.optimization_goal.value}")
    print(f"  Num buckets:    {args.num_buckets}")
    print()
    print(f"  Suggested edges:  {suggestion.edges}")
    print(f"  Overflow max:     {suggestion.overflow_max}")
    print(f"  Est. efficiency:  {suggestion.estimated_efficiency:.1%}")
    print()
    print(f"  Rationale: {suggestion.rationale}")

    # Show CLI command to use
    edges_str = ",".join(str(e) for e in suggestion.edges)
    print("\n  Use with:")
    print(
        f"    lazarus data batchplan build --bucket-edges {edges_str} --overflow-max {suggestion.overflow_max} ..."
    )


def data_batch_generate(args):
    """Generate NPZ batch files from a BatchPlan."""
    import asyncio
    import json
    from pathlib import Path

    from ..data.batching import (
        BatchReader,
        BatchWriter,
        load_batch_plan,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    async def generate():
        # Load batch plan
        logger.info(f"Loading batch plan: {args.plan}")
        plan = load_batch_plan(Path(args.plan))

        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        with open(args.dataset) as f:
            if args.dataset.endswith(".jsonl"):
                raw_samples = [json.loads(line) for line in f if line.strip()]
            else:
                raw_samples = json.load(f)

        # Tokenize samples
        logger.info("Tokenizing samples...")
        samples = {}
        for i, sample in enumerate(raw_samples):
            sample_id = sample.get("id") or sample.get("sample_id") or f"sample_{i:06d}"

            # Get text
            text = sample.get("text") or sample.get("content") or sample.get("input")
            if text is None and "messages" in sample:
                text = " ".join(m.get("content", "") for m in sample["messages"])

            if text:
                input_ids = tokenizer.encode(text, add_special_tokens=True)
                # Create simple loss mask (all 1s for now)
                loss_mask = [1] * len(input_ids)
                samples[sample_id] = {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }

            if (i + 1) % 1000 == 0:
                logger.info(f"Tokenized {i + 1}/{len(raw_samples)} samples")

        # Create writer
        output_dir = Path(args.output)
        logger.info(f"Writing batches to: {output_dir}")

        writer = BatchWriter(
            plan=plan,
            samples=samples,
            output_dir=output_dir,
            pad_id=tokenizer.pad_token_id or 0,
        )

        # Write batches
        files = writer.write_all()

        return len(files), output_dir

    num_files, output_dir = asyncio.run(generate())

    print(f"\n{'=' * 60}")
    print("Batch Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Batch plan:   {args.plan}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Output:       {output_dir}")
    print(f"  Files:        {num_files}")

    # Verify
    reader = BatchReader(output_dir)
    print(f"  Epochs:       {reader.num_epochs}")
    if reader.fingerprint:
        print(f"  Fingerprint:  {reader.fingerprint}")


def gym_run(args):
    """Run gym episode streaming and collect samples."""
    import asyncio

    from ..data.batching.streaming import (
        GymConfig,
        GymEpisodeStream,
        GymOutputMode,
        GymTransport,
        MockGymStream,
        ReplayBuffer,
        ReplayBufferConfig,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    async def run():
        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)

        # Configure replay buffer
        buffer_config = ReplayBufferConfig(
            max_size=args.buffer_size,
            seed=args.seed,
        )
        buffer = ReplayBuffer(buffer_config)

        # Configure gym stream
        if args.mock:
            logger.info("Using mock gym stream for testing")
            stream = MockGymStream(
                tokenizer=tokenizer,
                num_episodes=args.num_episodes,
                steps_per_episode=args.steps_per_episode,
                difficulty_range=(args.difficulty_min, args.difficulty_max),
                success_rate=args.success_rate,
                seed=args.seed,
            )
        else:
            # Parse transport
            transport = GymTransport(args.transport)
            output_mode = GymOutputMode(args.output_mode)

            config = GymConfig(
                host=args.host,
                port=args.port,
                transport=transport,
                output_mode=output_mode,
                connect_timeout=args.timeout,
                max_retries=args.retries,
                difficulty_range=(args.difficulty_min, args.difficulty_max),
            )

            stream = GymEpisodeStream(
                config=config,
                tokenizer=tokenizer,
            )

        # Run streaming
        logger.info(f"Starting gym stream to {args.host}:{args.port}")
        print(f"\n{'=' * 60}")
        print("Gym Episode Streaming")
        print(f"{'=' * 60}")

        sample_count = 0
        episode_ids = set()

        async with stream:
            async for sample in stream:
                buffer.add(sample)
                sample_count += 1
                if sample.episode_id:
                    episode_ids.add(sample.episode_id)

                if sample_count % 100 == 0:
                    print(
                        f"  Samples: {sample_count}, "
                        f"Episodes: {len(episode_ids)}, "
                        f"Buffer: {buffer.size}"
                    )

                if args.max_samples and sample_count >= args.max_samples:
                    logger.info(f"Reached max samples: {args.max_samples}")
                    break

        # Print summary
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  Total samples:    {sample_count}")
        print(f"  Total episodes:   {len(episode_ids)}")
        print(f"  Buffer size:      {buffer.size}")
        print(f"  Success rate:     {buffer.success_rate:.1%}")
        print(f"  Mean difficulty:  {buffer.mean_difficulty:.2f}")
        print(f"  Mean reward:      {buffer.mean_reward:.2f}")

        # Save buffer if output specified
        if args.output:
            import json
            from pathlib import Path

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            buffer_data = buffer.to_dict()
            with open(output_path, "w") as f:
                json.dump(buffer_data, f, indent=2, default=str)

            print(f"\n  Buffer saved to: {output_path}")

        return buffer

    asyncio.run(run())


def bench_pipeline(args):
    """Run comprehensive batching pipeline benchmark."""
    import asyncio
    import statistics
    import time

    from ..data.batching import (
        BatchingConfig,
        BatchPlanBuilder,
        BucketSpec,
        PackingConfig,
        PackingMode,
        SequenceToPack,
        analyze_bucket_efficiency,
        compute_length_histogram,
        compute_packing_metrics,
        create_efficiency_report,
        pack_sequences,
    )
    from ..utils.tokenizer_loader import load_tokenizer

    print(f"\n{'=' * 70}")
    print("LAZARUS PIPELINE BENCHMARK")
    print(f"{'=' * 70}")

    # Load tokenizer if provided, else use mock lengths
    if args.dataset:
        print(f"\nDataset: {args.dataset}")
        print(f"Tokenizer: {args.tokenizer}")

        tokenizer = load_tokenizer(args.tokenizer)

        # Tokenize and build lengths
        print("\n[1/7] Tokenizing dataset...")
        start = time.time()
        lengths = {}
        samples = {}
        import json

        with open(args.dataset) as f:
            for i, line in enumerate(f):
                if args.max_samples and i >= args.max_samples:
                    break
                data = json.loads(line)
                text = data.get("text", data.get("content", data.get("instruction", "")))
                if text:
                    ids = tokenizer.encode(text)
                    sample_id = data.get("id", f"sample_{i}")
                    lengths[sample_id] = len(ids)
                    samples[sample_id] = ids
        tokenize_time = time.time() - start
        tokenize_throughput = len(lengths) / tokenize_time if tokenize_time > 0 else 0
        print(f"    Tokenized {len(lengths)} samples in {tokenize_time:.2f}s")
        print(f"    Throughput: {tokenize_throughput:.0f} samples/sec")
    else:
        print("\nUsing synthetic data (no --dataset provided)")
        print(f"Samples: {args.num_samples}")

        # Generate synthetic lengths
        import random

        random.seed(args.seed)
        lengths = {f"s{i}": random.randint(32, args.max_length) for i in range(args.num_samples)}
        samples = {sid: list(range(length)) for sid, length in lengths.items()}
        tokenize_time = 0.0
        tokenize_throughput = 0.0

    # Parse bucket edges
    bucket_edges = tuple(int(x) for x in args.bucket_edges.split(","))
    total_tokens = sum(lengths.values())
    length_values = list(lengths.values())
    length_variance = statistics.variance(length_values) if len(length_values) > 1 else 0
    length_stddev = statistics.stdev(length_values) if len(length_values) > 1 else 0

    # Length histogram
    print("\n[2/7] Computing length histogram...")
    histogram = compute_length_histogram(lengths, num_bins=15)
    print(f"\n{histogram.to_ascii(width=50)}")
    print(f"    Min: {histogram.min_length}, Max: {histogram.max_length}")
    print(f"    Mean: {histogram.mean_length:.1f}, Median: {histogram.median_length}")
    print(f"    StdDev: {length_stddev:.1f}, Variance: {length_variance:.1f}")
    print(f"    P90: {histogram.p90}, P99: {histogram.p99}")

    # Bucket efficiency analysis
    print("\n[3/7] Analyzing bucket efficiency...")
    bucket_spec = BucketSpec(edges=bucket_edges, overflow_max=args.max_length)
    bucket_analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    print(f"\n{bucket_analysis.to_ascii()}")
    print(f"    Overall efficiency: {bucket_analysis.overall_efficiency:.1%}")

    # Batch plan building
    print("\n[4/7] Building batch plan...")
    config = BatchingConfig.predictable(
        token_budget=args.token_budget,
        bucket_edges=bucket_edges,
        overflow_max=args.max_length,
        seed=args.seed,
    )

    start = time.time()
    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=config,
        dataset_hash="benchmark",
        tokenizer_hash="benchmark",
    )
    plan = asyncio.run(builder.build(num_epochs=1))
    plan_time = time.time() - start

    total_batches = plan.total_microbatches
    epoch = plan.get_epoch(0)
    epoch_tokens = epoch.total_tokens

    print(f"    Built plan in {plan_time:.3f}s")
    print(f"    Total microbatches: {total_batches}")
    print(f"    Total tokens: {epoch_tokens:,}")
    print(f"    Fingerprint: {plan.fingerprint}")

    # Compute batch metrics
    avg_batch_size = epoch.total_samples / total_batches if total_batches > 0 else 0
    avg_tokens_per_batch = epoch_tokens / total_batches if total_batches > 0 else 0

    # Compute padding waste for pad-to-bucket strategy
    print("\n[5/7] Computing padding waste (pad-to-bucket)...")
    padded_tokens_bucket = 0
    for sid, length in lengths.items():
        bucket_id = bucket_spec.get_bucket_id(length)
        _, max_len = bucket_spec.get_bucket_range(bucket_id)
        padded_tokens_bucket += max_len
    padding_waste_bucket = (
        1.0 - (total_tokens / padded_tokens_bucket) if padded_tokens_bucket > 0 else 0
    )
    print(f"    Total tokens (raw): {total_tokens:,}")
    print(f"    Total tokens (padded to bucket): {padded_tokens_bucket:,}")
    print(f"    Padding waste: {padding_waste_bucket:.1%}")

    # Packing analysis
    print("\n[6/7] Packing analysis...")
    # Take a sample of sequences for packing demo
    sample_seqs = [
        SequenceToPack(
            sample_id=sid,
            input_ids=tuple(samples[sid][: lengths[sid]]),
            loss_mask=tuple([1] * lengths[sid]),
        )
        for sid in list(lengths.keys())[: min(500, len(lengths))]
    ]

    pack_config = PackingConfig(
        mode=PackingMode.GREEDY,
        max_length=args.max_length,
        pad_to_max=True,
    )

    start = time.time()
    packed = pack_sequences(sample_seqs, pack_config, pad_token_id=0)
    pack_time = time.time() - start
    pack_metrics = compute_packing_metrics(packed)

    print(f"    Packed {len(sample_seqs)} → {len(packed)} sequences in {pack_time:.3f}s")
    print(f"    Packing ratio: {pack_metrics.packing_ratio:.2f}x")
    print(f"    Efficiency: {pack_metrics.efficiency:.1%}")
    if pack_metrics.packing_ratio > 1:
        print(f"    Token reduction: {1 - 1 / pack_metrics.packing_ratio:.0%}")

    # Memory footprint estimation
    print("\n[7/7] Memory footprint estimation...")
    # Estimate memory for different strategies
    bytes_per_token = 4  # int32
    mem_raw = total_tokens * bytes_per_token
    mem_padded_bucket = padded_tokens_bucket * bytes_per_token
    mem_packed = (
        sum(len(p.input_ids) for p in packed) * bytes_per_token * (len(lengths) / len(sample_seqs))
    )

    print(f"    Raw tokens: {mem_raw / 1024 / 1024:.1f} MB")
    print(f"    Padded (bucket): {mem_padded_bucket / 1024 / 1024:.1f} MB")
    print(f"    Packed (estimated): {mem_packed / 1024 / 1024:.1f} MB")

    # Efficiency report
    print("\n[8/8] Creating efficiency report...")
    report = create_efficiency_report(lengths, bucket_spec)
    print(f"\n{report.to_ascii()}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PACK VS PAD COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PACK VS PAD COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n{'Strategy':<25} {'Tokens':>15} {'Waste %':>12} {'Memory':>12}")
    print("-" * 66)
    print(
        f"{'Raw (no padding)':<25} {total_tokens:>15,} {'0.0%':>12} {mem_raw / 1024 / 1024:>10.1f} MB"
    )
    print(
        f"{'Pad-to-bucket':<25} {padded_tokens_bucket:>15,} {padding_waste_bucket:>11.1%} {mem_padded_bucket / 1024 / 1024:>10.1f} MB"
    )

    # Estimate packed total tokens
    packed_total_tokens = (
        int(total_tokens / pack_metrics.efficiency) if pack_metrics.efficiency > 0 else total_tokens
    )
    packed_waste = 1.0 - pack_metrics.efficiency
    print(
        f"{'Packed (greedy)':<25} {packed_total_tokens:>15,} {packed_waste:>11.1%} {mem_packed / 1024 / 1024:>10.1f} MB"
    )

    if padding_waste_bucket > packed_waste:
        savings = padding_waste_bucket - packed_waste
        print(f"\n    → Packing saves {savings:.1%} waste vs pad-to-bucket")
    else:
        print("\n    → Pad-to-bucket is more efficient for this distribution")

    # ═══════════════════════════════════════════════════════════════════════════
    # THROUGHPUT METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("THROUGHPUT METRICS")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 57)
    print(f"{'Tokenization throughput':<35} {tokenize_throughput:>15.0f} samp/s")
    print(f"{'Plan build throughput':<35} {len(lengths) / plan_time:>15.0f} samp/s")
    print(f"{'Effective tokens/batch':<35} {avg_tokens_per_batch:>20.0f}")
    print(f"{'Tokens/batch (theoretical max)':<35} {args.token_budget:>20}")
    print(f"{'Token budget utilization':<35} {avg_tokens_per_batch / args.token_budget:>19.1%}")

    # Batch size variance
    batch_sizes = [len(mb.samples) for mb in epoch.microbatches]
    batch_size_variance = statistics.variance(batch_sizes) if len(batch_sizes) > 1 else 0
    batch_size_stddev = statistics.stdev(batch_sizes) if len(batch_sizes) > 1 else 0

    print(f"{'Batch size mean':<35} {statistics.mean(batch_sizes):>20.1f}")
    print(f"{'Batch size stddev':<35} {batch_size_stddev:>20.1f}")
    print(f"{'Batch size variance':<35} {batch_size_variance:>20.1f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 57)
    print(f"{'Samples':<35} {len(lengths):>20,}")
    print(f"{'Total tokens':<35} {total_tokens:>20,}")
    print(f"{'Length stddev':<35} {length_stddev:>20.1f}")
    print(f"{'Tokenization time':<35} {tokenize_time:>19.2f}s")
    print(f"{'Plan build time':<35} {plan_time:>19.3f}s")
    print(f"{'Pack time (500 samples)':<35} {pack_time:>19.3f}s")
    print(f"{'Microbatches per epoch':<35} {total_batches:>20,}")
    print(f"{'Avg batch size':<35} {avg_batch_size:>20.1f}")
    print(f"{'Avg tokens/batch':<35} {avg_tokens_per_batch:>20.0f}")
    print(f"{'Token budget utilization':<35} {avg_tokens_per_batch / args.token_budget:>19.1%}")
    print(f"{'Bucket efficiency':<35} {bucket_analysis.overall_efficiency:>19.1%}")
    print(f"{'Padding waste (bucket)':<35} {padding_waste_bucket:>19.1%}")
    print(f"{'Packing ratio':<35} {pack_metrics.packing_ratio:>19.2f}x")
    print(f"{'Packing efficiency':<35} {pack_metrics.efficiency:>19.1%}")
    print(f"{'Plan fingerprint':<35} {plan.fingerprint:>20}")

    if report.recommendations:
        print(f"\n{'Recommendations:':<35}")
        for rec in report.recommendations[:3]:
            print(f"  • {rec}")

    # Key insight
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print(f"{'=' * 70}")
    if pack_metrics.packing_ratio > 1.3:
        print(f"\n  Packing recommended: {pack_metrics.packing_ratio:.1f}x compression saves")
        print(f"  {1 - 1 / pack_metrics.packing_ratio:.0%} tokens per epoch.")
    elif bucket_analysis.overall_efficiency > 0.85:
        print(f"\n  Bucket efficiency is high ({bucket_analysis.overall_efficiency:.0%}).")
        print("  Pad-to-bucket is sufficient for this distribution.")
    else:
        print(
            f"\n  Consider adjusting bucket edges. Current efficiency: {bucket_analysis.overall_efficiency:.0%}"
        )
        print("  Suggested edges from report may improve utilization.")

    print(f"\n{'=' * 70}")
    print("Benchmark complete. Plan fingerprint can be used for CI/CD verification.")
    print(f"{'=' * 70}\n")


def gym_info(args):
    """Display gym stream configuration info."""
    from ..data.batching.streaming import (
        GymOutputMode,
        GymTransport,
    )

    print(f"\n{'=' * 60}")
    print("Gym Stream Configuration")
    print(f"{'=' * 60}")

    print("\nSupported Transports:")
    for transport in GymTransport:
        print(f"  - {transport.value}")

    print("\nSupported Output Modes:")
    for mode in GymOutputMode:
        print(f"  - {mode.value}")

    print("\nDefault Configuration:")
    print("  Host:             localhost")
    print("  Port:             8023")
    print("  Transport:        telnet")
    print("  Output Mode:      json")
    print("  Connect Timeout:  10.0s")
    print("  Max Retries:      3")

    print("\nExample Usage:")
    print("  # Run mock stream for testing")
    print("  lazarus gym run --tokenizer gpt2 --mock --num-episodes 10")
    print()
    print("  # Connect to puzzle arcade server")
    print("  lazarus gym run --tokenizer gpt2 --host localhost --port 8023")
    print()
    print("  # Save samples to buffer file")
    print("  lazarus gym run --tokenizer gpt2 --mock --output buffer.json")


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
    # Batching options
    sft_parser.add_argument("--batchplan", help="Use pre-computed batch plan directory")
    sft_parser.add_argument(
        "--bucket-edges",
        help="Bucket edges for length-based batching (e.g., 128,256,512)",
    )
    sft_parser.add_argument(
        "--token-budget",
        type=int,
        help="Token budget for dynamic batching (replaces --batch-size)",
    )
    sft_parser.add_argument("--pack", action="store_true", help="Enable sequence packing")
    sft_parser.add_argument("--pack-max-len", type=int, help="Max length for packed sequences")
    sft_parser.add_argument(
        "--pack-mode",
        choices=["first_fit", "best_fit", "greedy"],
        default="first_fit",
        help="Packing algorithm",
    )
    # Online training options
    sft_parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online training with gym stream",
    )
    sft_parser.add_argument(
        "--gym-host",
        default="localhost",
        help="Gym server host for online training",
    )
    sft_parser.add_argument(
        "--gym-port",
        type=int,
        default=8023,
        help="Gym server port for online training",
    )
    sft_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size for online training",
    )
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
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full tokenization"
    )
    compare_parser.set_defaults(func=tokenizer_compare)

    # Doctor command
    doctor_parser = tok_subparsers.add_parser("doctor", help="Run tokenizer health check")
    doctor_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    doctor_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix detected issues (patches chat template if missing)",
    )
    doctor_parser.add_argument(
        "--format",
        choices=["chatml", "llama", "phi", "gemma", "zephyr", "vicuna", "alpaca"],
        help="Specify chat template format when using --fix (auto-detects if not set)",
    )
    doctor_parser.add_argument(
        "--output",
        "-o",
        help="Save patched tokenizer to directory (requires --fix)",
    )
    doctor_parser.set_defaults(func=tokenizer_doctor)

    # Fingerprint command
    fingerprint_parser = tok_subparsers.add_parser(
        "fingerprint", help="Generate or verify tokenizer fingerprint"
    )
    fingerprint_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    fingerprint_parser.add_argument("--save", "-s", help="Save fingerprint to JSON file")
    fingerprint_parser.add_argument("--verify", help="Verify against fingerprint (file or string)")
    fingerprint_parser.add_argument(
        "--strict", action="store_true", help="Strict verification (merges must match)"
    )
    fingerprint_parser.set_defaults(func=tokenizer_fingerprint)

    # Benchmark command
    bench_parser = tok_subparsers.add_parser("benchmark", help="Benchmark tokenizer throughput")
    bench_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    bench_parser.add_argument("--file", "-f", help="Corpus file (one text per line)")
    bench_parser.add_argument(
        "--samples", "-n", type=int, default=1000, help="Number of samples (default: 1000)"
    )
    bench_parser.add_argument(
        "--avg-length", type=int, default=100, help="Avg words per sample for synthetic corpus"
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for synthetic corpus"
    )
    bench_parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of parallel workers"
    )
    bench_parser.add_argument("--warmup", type=int, default=10, help="Warmup samples before timing")
    bench_parser.add_argument(
        "--special-tokens", action="store_true", help="Add special tokens during encoding"
    )
    bench_parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare HuggingFace vs Fast (MLX) backend",
    )
    bench_parser.set_defaults(func=tokenizer_benchmark)

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

    # Efficiency analysis
    eff_parser = analyze_subparsers.add_parser(
        "efficiency", help="Analyze token efficiency metrics"
    )
    eff_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    eff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    eff_parser.set_defaults(func=analyze_efficiency)

    # Vocab suggestion
    vocab_parser = analyze_subparsers.add_parser(
        "vocab-suggest", help="Suggest vocabulary additions"
    )
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_parser.add_argument("--min-freq", type=int, default=5, help="Minimum frequency")
    vocab_parser.add_argument("--min-frag", type=int, default=3, help="Minimum fragmentation")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Maximum candidates")
    vocab_parser.add_argument("--show", type=int, default=20, help="Number to display")
    vocab_parser.set_defaults(func=analyze_vocab_suggest)

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

    # === Research subcommands ===
    research_parser = tok_subparsers.add_parser("research", help="Research playground tools")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command", help="Research tool"
    )

    # Soft tokens
    soft_parser = research_subparsers.add_parser(
        "soft-tokens", help="Create soft token bank for prompt tuning"
    )
    soft_parser.add_argument(
        "--num-tokens", "-n", type=int, default=10, help="Number of soft tokens"
    )
    soft_parser.add_argument(
        "--embedding-dim", "-d", type=int, default=768, help="Embedding dimension"
    )
    soft_parser.add_argument("--prefix", "-p", default="prompt", help="Token name prefix")
    soft_parser.add_argument(
        "--init-method",
        choices=["random_normal", "random_uniform", "zeros"],
        default="random_normal",
        help="Initialization method",
    )
    soft_parser.add_argument("--init-std", type=float, default=0.02, help="Std dev for random init")
    soft_parser.add_argument("--output", "-o", help="Save bank to JSON file")
    soft_parser.set_defaults(func=research_soft_tokens)

    # Analyze embeddings
    emb_parser = research_subparsers.add_parser(
        "analyze-embeddings", help="Analyze embedding space"
    )
    emb_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    emb_parser.add_argument("--num-clusters", "-k", type=int, default=10, help="Number of clusters")
    emb_parser.add_argument("--cluster", action="store_true", help="Show cluster analysis")
    emb_parser.add_argument("--project", action="store_true", help="Show 2D projection stats")
    emb_parser.set_defaults(func=research_analyze_embeddings)

    # Morph
    morph_parser = research_subparsers.add_parser("morph", help="Morph between token embeddings")
    morph_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    morph_parser.add_argument("--source", "-s", type=int, required=True, help="Source token index")
    morph_parser.add_argument("--target", "-t", type=int, required=True, help="Target token index")
    morph_parser.add_argument(
        "--method",
        "-m",
        choices=["linear", "spherical", "bezier", "cubic"],
        default="linear",
        help="Morph method",
    )
    morph_parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    morph_parser.add_argument("--normalize", action="store_true", help="Normalize output")
    morph_parser.add_argument("--output", "-o", help="Save trajectory to JSON")
    morph_parser.set_defaults(func=research_morph)

    # === Instrumentation subcommands ===
    instrument_parser = tok_subparsers.add_parser("instrument", help="Tokenizer instrumentation")
    instrument_subparsers = instrument_parser.add_subparsers(
        dest="instrument_command", help="Instrumentation tool"
    )

    # Histogram
    hist_parser = instrument_subparsers.add_parser(
        "histogram", help="Display token length histogram"
    )
    hist_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    hist_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    hist_parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    hist_parser.add_argument("--width", type=int, default=50, help="Chart width")
    hist_parser.add_argument("--quick", action="store_true", help="Quick stats only")
    hist_parser.set_defaults(func=instrument_histogram)

    # OOV analysis
    oov_parser = instrument_subparsers.add_parser("oov", help="Analyze OOV and rare tokens")
    oov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    oov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    oov_parser.add_argument("--vocab-size", type=int, default=50000, help="Expected vocab size")
    oov_parser.add_argument("--show-rare", action="store_true", help="Show rare tokens")
    oov_parser.add_argument("--max-freq", type=int, default=5, help="Max frequency for rare")
    oov_parser.add_argument("--top-k", type=int, default=20, help="Number of rare tokens to show")
    oov_parser.set_defaults(func=instrument_oov)

    # Waste analysis
    waste_parser = instrument_subparsers.add_parser(
        "waste", help="Analyze padding and truncation waste"
    )
    waste_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    waste_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    waste_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    waste_parser.set_defaults(func=instrument_waste)

    # Vocab diff
    vocab_diff_parser = instrument_subparsers.add_parser(
        "vocab-diff", help="Compare two tokenizers on a corpus"
    )
    vocab_diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    vocab_diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    vocab_diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_diff_parser.add_argument("--examples", type=int, default=5, help="Max examples to show")
    vocab_diff_parser.add_argument(
        "--cost", action="store_true", help="Show retokenization cost estimate"
    )
    vocab_diff_parser.set_defaults(func=instrument_vocab_diff)

    # === Data subcommand ===
    data_parser = subparsers.add_parser("data", help="Data processing utilities")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")

    # === Lengths subcommands ===
    lengths_parser = data_subparsers.add_parser("lengths", help="Length cache utilities")
    lengths_subparsers = lengths_parser.add_subparsers(
        dest="lengths_command", help="Lengths commands"
    )

    # Build length cache
    lengths_build_parser = lengths_subparsers.add_parser(
        "build", help="Build length cache from dataset"
    )
    lengths_build_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    lengths_build_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    lengths_build_parser.add_argument(
        "--output", "-o", required=True, help="Output cache file path"
    )
    lengths_build_parser.set_defaults(func=data_lengths_build)

    # Length cache stats
    lengths_stats_parser = lengths_subparsers.add_parser(
        "stats", help="Show length cache statistics"
    )
    lengths_stats_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    lengths_stats_parser.set_defaults(func=data_lengths_stats)

    # === BatchPlan subcommands ===
    batchplan_parser = data_subparsers.add_parser("batchplan", help="Batch plan utilities")
    batchplan_subparsers = batchplan_parser.add_subparsers(
        dest="batchplan_command", help="BatchPlan commands"
    )

    # Build batch plan
    batchplan_build_parser = batchplan_subparsers.add_parser(
        "build", help="Build batch plan from length cache"
    )
    batchplan_build_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_build_parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="Number of epochs (default: 1)"
    )
    batchplan_build_parser.add_argument(
        "--token-budget",
        "-b",
        type=int,
        default=4096,
        help="Token budget per batch (default: 4096)",
    )
    batchplan_build_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges (comma-separated, default: 128,256,512)",
    )
    batchplan_build_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batchplan_build_parser.add_argument(
        "--predictable",
        "-p",
        action="store_true",
        help="Use predictable mode (deterministic batching)",
    )
    batchplan_build_parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for predictable mode (default: 42)"
    )
    batchplan_build_parser.add_argument("--dataset-hash", help="Dataset hash for fingerprinting")
    batchplan_build_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for batch plan"
    )
    batchplan_build_parser.set_defaults(func=data_batchplan_build)

    # Batch plan info
    batchplan_info_parser = batchplan_subparsers.add_parser(
        "info", help="Show batch plan information"
    )
    batchplan_info_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_info_parser.add_argument(
        "--show-batches", "-n", type=int, default=0, help="Number of sample batches to show"
    )
    batchplan_info_parser.add_argument(
        "--rank", "-r", type=int, default=None, help="Worker rank for sharded view (0-indexed)"
    )
    batchplan_info_parser.add_argument(
        "--world-size", "-w", type=int, default=None, help="Total number of workers"
    )
    batchplan_info_parser.set_defaults(func=data_batchplan_info)

    # Batch plan verify
    batchplan_verify_parser = batchplan_subparsers.add_parser(
        "verify", help="Verify batch plan reproducibility"
    )
    batchplan_verify_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_verify_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_verify_parser.set_defaults(func=data_batchplan_verify)

    # Batch plan shard
    batchplan_shard_parser = batchplan_subparsers.add_parser(
        "shard", help="Create sharded batch plans for distributed training"
    )
    batchplan_shard_parser.add_argument(
        "--plan", "-p", required=True, help="Source batch plan directory"
    )
    batchplan_shard_parser.add_argument(
        "--world-size", "-w", type=int, required=True, help="Number of distributed workers"
    )
    batchplan_shard_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for sharded plans"
    )
    batchplan_shard_parser.set_defaults(func=data_batchplan_shard)

    # === Batching analysis subcommands ===
    batching_parser = data_subparsers.add_parser("batching", help="Batching analysis utilities")
    batching_subparsers = batching_parser.add_subparsers(
        dest="batching_command", help="Batching commands"
    )

    # Analyze batching efficiency
    batching_analyze_parser = batching_subparsers.add_parser(
        "analyze", help="Analyze batching efficiency"
    )
    batching_analyze_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_analyze_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges to analyze (comma-separated, default: 128,256,512)",
    )
    batching_analyze_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batching_analyze_parser.add_argument("--output", "-o", help="Save JSON report to file")
    batching_analyze_parser.set_defaults(func=data_batching_analyze)

    # Length histogram
    batching_histogram_parser = batching_subparsers.add_parser(
        "histogram", help="Display length histogram"
    )
    batching_histogram_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_histogram_parser.add_argument(
        "--bins", type=int, default=15, help="Number of histogram bins (default: 15)"
    )
    batching_histogram_parser.add_argument(
        "--width", type=int, default=50, help="Chart width (default: 50)"
    )
    batching_histogram_parser.set_defaults(func=data_batching_histogram)

    # Suggest bucket edges
    batching_suggest_parser = batching_subparsers.add_parser(
        "suggest", help="Suggest optimal bucket edges"
    )
    batching_suggest_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_suggest_parser.add_argument(
        "--num-buckets", "-n", type=int, default=4, help="Number of buckets (default: 4)"
    )
    batching_suggest_parser.add_argument(
        "--goal",
        "-g",
        choices=["waste", "balance", "memory"],
        default="waste",
        help="Optimization goal: waste (minimize padding), balance (even bucket sizes), memory (power-of-2 edges)",
    )
    batching_suggest_parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum sequence length (default: 2048)"
    )
    batching_suggest_parser.set_defaults(func=data_batching_suggest)

    # === Batch generation subcommands ===
    batch_parser = data_subparsers.add_parser("batch", help="Batch file generation")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", help="Batch commands")

    # Generate NPZ batch files
    batch_generate_parser = batch_subparsers.add_parser(
        "generate", help="Generate NPZ batch files from BatchPlan"
    )
    batch_generate_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batch_generate_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    batch_generate_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    batch_generate_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for NPZ files"
    )
    batch_generate_parser.set_defaults(func=data_batch_generate)

    # === Gym subcommand ===
    gym_parser = subparsers.add_parser("gym", help="Gym streaming utilities")
    gym_subparsers = gym_parser.add_subparsers(dest="gym_command", help="Gym commands")

    # Gym run command
    gym_run_parser = gym_subparsers.add_parser(
        "run", help="Run gym episode streaming and collect samples"
    )
    gym_run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    gym_run_parser.add_argument("--host", default="localhost", help="Gym server host")
    gym_run_parser.add_argument("--port", type=int, default=8023, help="Gym server port")
    gym_run_parser.add_argument(
        "--transport",
        choices=["telnet", "websocket", "http"],
        default="telnet",
        help="Transport protocol",
    )
    gym_run_parser.add_argument(
        "--output-mode",
        choices=["json", "text", "binary"],
        default="json",
        help="Output format from gym",
    )
    gym_run_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size",
    )
    gym_run_parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout")
    gym_run_parser.add_argument("--retries", type=int, default=3, help="Max connection retries")
    gym_run_parser.add_argument(
        "--difficulty-min",
        type=float,
        default=0.0,
        help="Minimum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--difficulty-max",
        type=float,
        default=1.0,
        help="Maximum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to collect (infinite if not set)",
    )
    gym_run_parser.add_argument("--output", "-o", help="Output file for buffer (JSON)")
    gym_run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Mock mode for testing
    gym_run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock gym stream for testing",
    )
    gym_run_parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes (mock mode)",
    )
    gym_run_parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=5,
        help="Steps per episode (mock mode)",
    )
    gym_run_parser.add_argument(
        "--success-rate",
        type=float,
        default=0.7,
        help="Success rate (mock mode)",
    )
    gym_run_parser.set_defaults(func=gym_run)

    # Gym info command
    gym_info_parser = gym_subparsers.add_parser(
        "info", help="Display gym stream configuration info"
    )
    gym_info_parser.set_defaults(func=gym_info)

    # =========================================================================
    # Bench command - comprehensive pipeline benchmark
    # =========================================================================
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark the batching pipeline",
        description="Run comprehensive benchmarks on tokenization, batching, packing, and efficiency.",
    )
    bench_parser.add_argument(
        "-d",
        "--dataset",
        help="JSONL dataset file (optional - uses synthetic data if not provided)",
    )
    bench_parser.add_argument(
        "-t",
        "--tokenizer",
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    bench_parser.add_argument(
        "--bucket-edges",
        default="128,256,512,1024",
        help="Bucket edge lengths (comma-separated, default: 128,256,512,1024)",
    )
    bench_parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
        help="Token budget per microbatch (default: 4096)",
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    bench_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process from dataset",
    )
    bench_parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (when no dataset, default: 1000)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    bench_parser.set_defaults(func=bench_pipeline)

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
    elif args.command == "gym" and getattr(args, "gym_command", None) is None:
        parser.parse_args(["gym", "--help"])


if __name__ == "__main__":
    main()
