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
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test" --verbose
    lazarus tokenizer doctor -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t model --save fingerprint.json
    lazarus tokenizer fingerprint -t model --verify fingerprint.json --strict
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
    from ..utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"\n{'=' * 60}")
    print(f"Tokenizer Doctor: {args.tokenizer}")
    print(f"{'=' * 60}")

    issues = []
    warnings = []

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
    if has_chat_template:
        template_preview = str(tokenizer.chat_template)[:100]
        print("  Available: Yes")
        print(f"  Preview: {template_preview}...")

        # Test chat template
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            result = tokenizer.apply_chat_template(
                test_messages, add_generation_prompt=True, tokenize=False
            )
            print("  Test: PASS")
            if args.verbose:
                print(f"  Output: {result[:100]}...")
        except Exception as e:
            issues.append(f"Chat template error: {e}")
            print(f"  Test: FAIL - {e}")
    else:
        warnings.append("No chat template defined")
        print("  Available: No (may need manual formatting)")

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
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full tokenization"
    )
    compare_parser.set_defaults(func=tokenizer_compare)

    # Doctor command
    doctor_parser = tok_subparsers.add_parser("doctor", help="Run tokenizer health check")
    doctor_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    doctor_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
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
