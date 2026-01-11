"""Tokenizer doctor command handler."""

import logging
import sys

from .._types import DoctorConfig, DoctorResult, TokenizerHealthStatus

logger = logging.getLogger(__name__)


def tokenizer_doctor(config: DoctorConfig) -> DoctorResult:
    """Run comprehensive tokenizer health check.

    Args:
        config: Doctor configuration.

    Returns:
        Doctor result with health status, issues, and warnings.
    """
    from .....data.tokenizers.fingerprint import compute_fingerprint
    from .....data.tokenizers.runtime.chat_templates import (
        ChatTemplateRegistry,
        patch_chat_template,
        suggest_template_for_model,
        validate_chat_template,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    print(f"\n{'=' * 60}")
    print(f"Tokenizer Doctor: {config.tokenizer}")
    print(f"{'=' * 60}")

    issues: list[str] = []
    warnings: list[str] = []
    fixes_applied: list[str] = []

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

    for attr, (short, _) in special_tokens.items():
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
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

    validation_result = validate_chat_template(tokenizer)
    registry = ChatTemplateRegistry()

    if has_chat_template:
        template_str = str(tokenizer.chat_template)
        template_preview = template_str[:100]
        print("  Available: Yes")
        print(f"  Preview: {template_preview}...")
        print(f"  Format: {validation_result.format.value}")

        if validation_result.capabilities:
            caps = [c.value for c in validation_result.capabilities]
            print(f"  Capabilities: {', '.join(caps)}")

        for issue in validation_result.issues:
            if issue.severity == "error":
                issues.append(issue.message)
                print(f"  ERROR: {issue.message}")
            elif issue.severity == "warning":
                warnings.append(issue.message)
                print(f"  WARN: {issue.message}")
            elif config.verbose:
                print(f"  INFO: {issue.message}")

        # Test chat template
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

        # Check system message support
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

        all_pass = True
        for scenario_name, messages in test_scenarios:
            try:
                result = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                if config.verbose:
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

        suggested = suggest_template_for_model(config.tokenizer)
        if suggested:
            print(f"  Suggested format: {suggested.format.value}")
            print(f"  Description: {suggested.description}")

        if config.fix:
            try:
                success = patch_chat_template(tokenizer, template_format=config.format)
                if success:
                    detected_format = registry.detect_format(tokenizer.chat_template)
                    fixes_applied.append(f"Added {detected_format.value} chat template")
                    print(f"  FIX APPLIED: Added {detected_format.value} chat template")
                else:
                    print("  FIX FAILED: Could not determine appropriate template")
            except Exception as e:
                print(f"  FIX FAILED: {e}")
        else:
            print("  Recommendation: Add a chat template for conversational use")
            print("  Use: lazarus tokenizer doctor -t MODEL --fix")

    # === Encode/Decode Roundtrip ===
    print("\n--- Encode/Decode Roundtrip ---")
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Special chars: @#$%^&*()",
        "Unicode: `}",
        "Numbers: 12345 3.14159",
    ]

    roundtrip_issues = 0
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            normalized_original = " ".join(text.split())
            normalized_decoded = " ".join(decoded.split())

            if normalized_original != normalized_decoded:
                roundtrip_issues += 1
                if config.verbose:
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
        if config.verbose:
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
        status = TokenizerHealthStatus.HEALTHY
        print("  Status: HEALTHY")
        print("  No issues found.")
    else:
        if issues:
            status = (
                TokenizerHealthStatus.CRITICAL if len(issues) > 2 else TokenizerHealthStatus.ISSUES
            )
            print(f"  Status: ISSUES FOUND ({len(issues)})")
            for issue in issues:
                print(f"    ERROR: {issue}")
        else:
            status = TokenizerHealthStatus.ISSUES

        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"    WARN: {warning}")

    # Save patched tokenizer if --fix and --output specified
    if config.fix and fixes_applied:
        if config.output:
            try:
                import os

                os.makedirs(config.output, exist_ok=True)
                tokenizer.save_pretrained(config.output)
                print(f"\n  Saved patched tokenizer to: {config.output}")
            except Exception as e:
                print(f"\n  ERROR: Could not save tokenizer: {e}")
        else:
            print("\n  Note: Use --output PATH to save the patched tokenizer")

    if issues:
        sys.exit(1)

    return DoctorResult(
        status=status,
        issues=issues,
        warnings=warnings,
        fixes_applied=fixes_applied,
    )
