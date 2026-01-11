"""Tokenizer fingerprint command handler."""

import logging
import sys

from .._types import FingerprintConfig, FingerprintResult

logger = logging.getLogger(__name__)


def tokenizer_fingerprint(config: FingerprintConfig) -> FingerprintResult:
    """Generate or verify tokenizer fingerprint.

    Args:
        config: Fingerprint configuration.

    Returns:
        Fingerprint result with hash information.
    """
    from .....data.tokenizers.fingerprint import (
        compute_fingerprint,
        load_fingerprint,
        save_fingerprint,
        verify_fingerprint,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Compute fingerprint
    fp = compute_fingerprint(tokenizer)

    verified = None
    match = None

    if config.verify:
        # Verify against expected fingerprint
        logger.info(f"Verifying against: {config.verify}")

        if config.verify.endswith(".json"):
            expected = load_fingerprint(config.verify)
        else:
            expected = config.verify  # Treat as fingerprint string

        mismatch = verify_fingerprint(tokenizer, expected, strict=config.strict)

        print(f"\n{'=' * 60}")
        print("Fingerprint Verification")
        print(f"{'=' * 60}")
        print(f"  Tokenizer: {config.tokenizer}")
        print(f"  Actual:    {fp.fingerprint}")

        if isinstance(expected, str):
            print(f"  Expected:  {expected}")
        else:
            print(f"  Expected:  {expected.fingerprint}")

        verified = True
        if mismatch is None:
            match = True
            print("\n  Result: MATCH")
        else:
            match = False
            print("\n  Result: MISMATCH")
            print(f"  Compatible: {'Yes' if mismatch.is_compatible else 'No'}")
            if mismatch.warnings:
                print("\n  Warnings:")
                for w in mismatch.warnings:
                    print(f"    - {w}")

            if not mismatch.is_compatible:
                sys.exit(1)

    elif config.save:
        # Save fingerprint to file
        save_fingerprint(fp, config.save)
        print(f"\n{'=' * 60}")
        print("Fingerprint Saved")
        print(f"{'=' * 60}")
        print(f"  Tokenizer:   {config.tokenizer}")
        print(f"  Fingerprint: {fp.fingerprint}")
        print(f"  Saved to:    {config.save}")

    else:
        # Just display fingerprint
        print(f"\n{'=' * 60}")
        print("Tokenizer Fingerprint")
        print(f"{'=' * 60}")
        print(f"  Tokenizer:     {config.tokenizer}")
        print(f"  Fingerprint:   {fp.fingerprint}")
        print(f"  Full hash:     {fp.full_hash}")
        print(f"  Vocab size:    {fp.vocab_size:,}")
        print(f"  Vocab hash:    {fp.vocab_hash}")
        print(f"  Special hash:  {fp.special_tokens_hash}")
        print(f"  Merges hash:   {fp.merges_hash}")

        print("\n  Special tokens:")
        for name, token_id in fp.special_tokens.items():
            print(f"    {name}: {token_id}")

    return FingerprintResult(
        fingerprint=fp.fingerprint,
        vocab_size=fp.vocab_size,
        vocab_hash=fp.vocab_hash,
        full_hash=fp.full_hash,
        special_tokens_hash=fp.special_tokens_hash,
        merges_hash=fp.merges_hash,
        special_tokens=fp.special_tokens,
        verified=verified,
        match=match,
    )
