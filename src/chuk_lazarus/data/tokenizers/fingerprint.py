"""
Tokenizer fingerprinting for compatibility verification.

Generates stable hashes of tokenizer configuration so datasets can
assert compatibility with their tokenizer requirements.

Use cases:
- Dataset metadata: "requires tokenizer fingerprint X"
- Model checkpoints: "trained with tokenizer fingerprint Y"
- CI/CD: detect tokenizer changes that break compatibility
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .types import SpecialTokenField, TokenizerProtocol


class TokenizerFingerprint(BaseModel):
    """
    Stable fingerprint of a tokenizer's configuration.

    Captures all aspects that affect tokenization behavior:
    - Vocabulary content and ordering
    - Special token assignments
    - Merge rules (if available)
    """

    # Primary fingerprint (short hash for quick comparison)
    fingerprint: str = Field(description="Short fingerprint hash (first 16 chars)")

    # Full hash for exact matching
    full_hash: str = Field(description="Full SHA-256 hash")

    # Component hashes for debugging
    vocab_hash: str = Field(description="Hash of vocabulary")
    special_tokens_hash: str = Field(description="Hash of special token config")
    merges_hash: str = Field(description="Hash of merge rules (if any)")

    # Metadata
    vocab_size: int = Field(description="Vocabulary size")
    algorithm: str = Field(default="sha256", description="Hash algorithm used")
    version: int = Field(default=1, description="Fingerprint format version")

    # Special tokens snapshot
    special_tokens: dict[str, int | None] = Field(
        default_factory=dict, description="Special token ID mapping"
    )

    def matches(self, other: "TokenizerFingerprint") -> bool:
        """Check if two fingerprints match."""
        return self.full_hash == other.full_hash

    def matches_vocab(self, other: "TokenizerFingerprint") -> bool:
        """Check if vocabularies match (ignoring special tokens)."""
        return self.vocab_hash == other.vocab_hash

    def diff(self, other: "TokenizerFingerprint") -> dict[str, bool]:
        """Get detailed diff between fingerprints."""
        return {
            "vocab_matches": self.vocab_hash == other.vocab_hash,
            "special_tokens_match": self.special_tokens_hash == other.special_tokens_hash,
            "merges_match": self.merges_hash == other.merges_hash,
            "size_matches": self.vocab_size == other.vocab_size,
            "full_match": self.full_hash == other.full_hash,
        }


class FingerprintMismatch(BaseModel):
    """Details about a fingerprint mismatch."""

    expected: TokenizerFingerprint = Field(description="Expected fingerprint")
    actual: TokenizerFingerprint = Field(description="Actual fingerprint")
    diff: dict[str, bool] = Field(description="Component-level diff")
    is_compatible: bool = Field(description="Whether mismatch is likely safe")
    warnings: list[str] = Field(default_factory=list, description="Compatibility warnings")


def _hash_dict(d: dict, algorithm: str = "sha256") -> str:
    """Hash a dictionary deterministically."""
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=True)
    hasher = hashlib.new(algorithm)
    hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


def _hash_list(items: list, algorithm: str = "sha256") -> str:
    """Hash a list deterministically."""
    serialized = json.dumps(items, ensure_ascii=True)
    hasher = hashlib.new(algorithm)
    hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


def compute_fingerprint(
    tokenizer: TokenizerProtocol,
    merges: list[str] | None = None,
) -> TokenizerFingerprint:
    """
    Compute a fingerprint for a tokenizer.

    Args:
        tokenizer: Tokenizer to fingerprint
        merges: Optional list of BPE merge rules

    Returns:
        TokenizerFingerprint capturing tokenizer identity
    """
    # Get vocabulary
    vocab = tokenizer.get_vocab()

    # Create sorted vocab representation for deterministic hashing
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_hash = _hash_list(sorted_vocab)

    # Special tokens - use enum for field names
    special_tokens: dict[str, int | None] = {}
    for field in (
        SpecialTokenField.PAD_TOKEN_ID,
        SpecialTokenField.UNK_TOKEN_ID,
        SpecialTokenField.BOS_TOKEN_ID,
        SpecialTokenField.EOS_TOKEN_ID,
    ):
        special_tokens[field.value] = getattr(tokenizer, field.value, None)

    # Add additional special tokens if available
    for field in (
        SpecialTokenField.SEP_TOKEN_ID,
        SpecialTokenField.CLS_TOKEN_ID,
        SpecialTokenField.MASK_TOKEN_ID,
    ):
        if hasattr(tokenizer, field.value):
            special_tokens[field.value] = getattr(tokenizer, field.value)

    special_tokens_hash = _hash_dict(special_tokens)

    # Merges (if provided)
    if merges is None:
        if hasattr(tokenizer, "bpe_ranks"):
            merges = list(tokenizer.bpe_ranks.keys())
        elif hasattr(tokenizer, "get_merges"):
            merges = tokenizer.get_merges()

    merges_hash = _hash_list(merges) if merges else "none"

    # Combine all hashes for full fingerprint
    combined = f"{vocab_hash}:{special_tokens_hash}:{merges_hash}"
    hasher = hashlib.sha256()
    hasher.update(combined.encode("utf-8"))
    full_hash = hasher.hexdigest()

    return TokenizerFingerprint(
        fingerprint=full_hash[:16],
        full_hash=full_hash,
        vocab_hash=vocab_hash[:16],
        special_tokens_hash=special_tokens_hash[:16],
        merges_hash=merges_hash[:16] if merges_hash != "none" else "none",
        vocab_size=len(vocab),
        special_tokens=special_tokens,
    )


def verify_fingerprint(
    tokenizer: TokenizerProtocol,
    expected: TokenizerFingerprint | str,
    strict: bool = False,
) -> FingerprintMismatch | None:
    """
    Verify a tokenizer matches an expected fingerprint.

    Args:
        tokenizer: Tokenizer to verify
        expected: Expected fingerprint (or fingerprint string)
        strict: If True, require exact match including merges

    Returns:
        FingerprintMismatch if mismatch detected, None if OK
    """
    actual = compute_fingerprint(tokenizer)

    # Handle string fingerprint (short form)
    if isinstance(expected, str):
        if actual.fingerprint == expected or actual.full_hash.startswith(expected):
            return None
        # Create minimal expected fingerprint for comparison
        expected = TokenizerFingerprint(
            fingerprint=expected,
            full_hash=expected,
            vocab_hash="unknown",
            special_tokens_hash="unknown",
            merges_hash="unknown",
            vocab_size=0,
        )

    # Full comparison
    diff = actual.diff(expected)

    if diff["full_match"]:
        return None

    # Check compatibility
    warnings: list[str] = []
    is_compatible = True

    if not diff["vocab_matches"]:
        is_compatible = False
        warnings.append("Vocabulary mismatch - tokenization will differ")

    if not diff["special_tokens_match"]:
        warnings.append("Special token IDs differ - may affect chat/tool templates")
        if diff["vocab_matches"]:
            is_compatible = True

    if not diff["merges_match"] and strict:
        warnings.append("Merge rules differ - BPE behavior may vary")
        is_compatible = False

    if not diff["size_matches"]:
        warnings.append(f"Vocab size differs: {expected.vocab_size} vs {actual.vocab_size}")

    return FingerprintMismatch(
        expected=expected,
        actual=actual,
        diff=diff,
        is_compatible=is_compatible,
        warnings=warnings,
    )


def assert_fingerprint(
    tokenizer: TokenizerProtocol,
    expected: TokenizerFingerprint | str,
    strict: bool = False,
) -> None:
    """
    Assert a tokenizer matches expected fingerprint, raise if not.

    Args:
        tokenizer: Tokenizer to verify
        expected: Expected fingerprint
        strict: If True, require exact match

    Raises:
        ValueError: If fingerprint doesn't match
    """
    mismatch = verify_fingerprint(tokenizer, expected, strict=strict)

    if mismatch is not None and not mismatch.is_compatible:
        warnings_str = "\n  - ".join(mismatch.warnings)
        raise ValueError(
            f"Tokenizer fingerprint mismatch!\n"
            f"Expected: {mismatch.expected.fingerprint}\n"
            f"Actual:   {mismatch.actual.fingerprint}\n"
            f"Issues:\n  - {warnings_str}"
        )


# =============================================================================
# Sync I/O
# =============================================================================


def save_fingerprint(fingerprint: TokenizerFingerprint, path: str | Path) -> None:
    """Save fingerprint to file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(fingerprint.model_dump(), f, indent=2)


def load_fingerprint(path: str | Path) -> TokenizerFingerprint:
    """Load fingerprint from file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return TokenizerFingerprint(**data)


# =============================================================================
# Async I/O
# =============================================================================


async def save_fingerprint_async(fingerprint: TokenizerFingerprint, path: str | Path) -> None:
    """Async: Save fingerprint to file."""
    import aiofiles

    path = Path(path)
    async with aiofiles.open(path, "w") as f:
        await f.write(json.dumps(fingerprint.model_dump(), indent=2))


async def load_fingerprint_async(path: str | Path) -> TokenizerFingerprint:
    """Async: Load fingerprint from file."""
    import aiofiles

    path = Path(path)
    async with aiofiles.open(path) as f:
        content = await f.read()
        data = json.loads(content)
    return TokenizerFingerprint(**data)


# =============================================================================
# Utilities
# =============================================================================


def fingerprint_from_json(json_str: str) -> TokenizerFingerprint:
    """Create fingerprint from JSON string."""
    data = json.loads(json_str)
    return TokenizerFingerprint(**data)


class FingerprintRegistry:
    """
    Registry of known tokenizer fingerprints.

    Use to validate tokenizers against known-good configurations.
    """

    def __init__(self) -> None:
        self._fingerprints: dict[str, TokenizerFingerprint] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        fingerprint: TokenizerFingerprint,
        aliases: list[str] | None = None,
    ) -> None:
        """
        Register a known fingerprint.

        Args:
            name: Canonical name (e.g., "llama-3-8b")
            fingerprint: The fingerprint
            aliases: Alternative names
        """
        self._fingerprints[name] = fingerprint
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> TokenizerFingerprint | None:
        """Get fingerprint by name or alias."""
        if name in self._fingerprints:
            return self._fingerprints[name]
        if name in self._aliases:
            return self._fingerprints[self._aliases[name]]
        return None

    def verify(
        self,
        tokenizer: TokenizerProtocol,
        expected_name: str,
        strict: bool = False,
    ) -> FingerprintMismatch | None:
        """Verify tokenizer against a registered fingerprint."""
        expected = self.get(expected_name)
        if expected is None:
            raise KeyError(f"Unknown fingerprint: {expected_name}")
        return verify_fingerprint(tokenizer, expected, strict=strict)

    def identify(
        self,
        tokenizer: TokenizerProtocol,
    ) -> list[tuple[str, TokenizerFingerprint]]:
        """
        Try to identify a tokenizer from registered fingerprints.

        Returns list of (name, fingerprint) for matches.
        """
        actual = compute_fingerprint(tokenizer)
        matches = []

        for name, expected in self._fingerprints.items():
            if actual.matches(expected):
                matches.append((name, expected))
            elif actual.matches_vocab(expected):
                matches.append((f"{name} (vocab-only)", expected))

        return matches

    def list_all(self) -> list[str]:
        """List all registered fingerprint names."""
        return list(self._fingerprints.keys())


# Global registry instance
_global_registry = FingerprintRegistry()


def get_registry() -> FingerprintRegistry:
    """Get the global fingerprint registry."""
    return _global_registry
