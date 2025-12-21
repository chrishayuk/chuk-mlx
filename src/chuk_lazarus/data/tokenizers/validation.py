"""Tokenizer validation utilities with Pydantic models."""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationCategory(str, Enum):
    """Category of validation issue."""

    ROUNDTRIP = "roundtrip"
    ENCODING = "encoding"
    DECODING = "decoding"
    SPECIAL_TOKENS = "special_tokens"
    VOCABULARY = "vocabulary"
    CONSISTENCY = "consistency"


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class ValidationIssue(BaseModel):
    """A single validation issue found during testing."""

    category: ValidationCategory = Field(description="Category of the issue")
    severity: ValidationSeverity = Field(description="Severity level")
    message: str = Field(description="Human-readable description")
    details: dict | None = Field(default=None, description="Additional context")


class RoundtripResult(BaseModel):
    """Result of a roundtrip encode-decode test."""

    original: str = Field(description="Original input text")
    encoded: list[int] = Field(description="Encoded token IDs")
    decoded: str = Field(description="Decoded text")
    is_lossless: bool = Field(description="Whether roundtrip was lossless")
    diff_chars: list[str] = Field(default_factory=list, description="Characters that changed")


class BatchRoundtripResult(BaseModel):
    """Aggregated results from multiple roundtrip tests."""

    total_tests: int = Field(ge=0, description="Number of tests run")
    passed: int = Field(ge=0, description="Number of lossless roundtrips")
    failed: int = Field(ge=0, description="Number of lossy roundtrips")
    pass_rate: float = Field(ge=0.0, le=1.0, description="Ratio of passed tests")
    failures: list[RoundtripResult] = Field(
        default_factory=list, description="Details of failed roundtrips"
    )


class EncodingConsistency(BaseModel):
    """Result of encoding consistency test."""

    text: str = Field(description="Test text")
    is_consistent: bool = Field(description="Whether all encodings matched")
    encodings: list[list[int]] = Field(description="All encoding results")
    num_trials: int = Field(ge=1, description="Number of encoding trials")


class ValidationReport(BaseModel):
    """Complete validation report for a tokenizer."""

    tokenizer_name: str = Field(description="Name/identifier of tokenizer")
    vocab_size: int = Field(ge=0, description="Vocabulary size")
    issues: list[ValidationIssue] = Field(default_factory=list, description="All issues found")
    roundtrip_result: BatchRoundtripResult | None = Field(
        default=None, description="Roundtrip test results"
    )
    is_valid: bool = Field(description="Overall validation passed")
    error_count: int = Field(ge=0, description="Number of errors")
    warning_count: int = Field(ge=0, description="Number of warnings")


def check_roundtrip(
    text: str,
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> RoundtripResult:
    """
    Test if encoding then decoding preserves the text.

    Args:
        text: Text to test
        tokenizer: Tokenizer instance
        add_special_tokens: Whether to add special tokens

    Returns:
        RoundtripResult with comparison details
    """
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    decoded = tokenizer.decode(encoded)

    # Find differing characters
    diff_chars = []
    if text != decoded:
        orig_set = set(text)
        decoded_set = set(decoded)
        diff_chars = list(orig_set.symmetric_difference(decoded_set))

    return RoundtripResult(
        original=text,
        encoded=encoded,
        decoded=decoded,
        is_lossless=(text == decoded),
        diff_chars=diff_chars,
    )


def check_batch_roundtrip(
    texts: list[str],
    tokenizer: TokenizerProtocol,
    add_special_tokens: bool = False,
) -> BatchRoundtripResult:
    """
    Test roundtrip on multiple texts.

    Args:
        texts: List of texts to test
        tokenizer: Tokenizer instance
        add_special_tokens: Whether to add special tokens

    Returns:
        BatchRoundtripResult with aggregated statistics
    """
    results = [check_roundtrip(text, tokenizer, add_special_tokens) for text in texts]

    passed = sum(1 for r in results if r.is_lossless)
    failed = len(results) - passed
    failures = [r for r in results if not r.is_lossless]

    return BatchRoundtripResult(
        total_tests=len(results),
        passed=passed,
        failed=failed,
        pass_rate=passed / len(results) if results else 1.0,
        failures=failures,
    )


def check_encoding_consistency(
    text: str,
    tokenizer: TokenizerProtocol,
    num_trials: int = 5,
    add_special_tokens: bool = False,
) -> EncodingConsistency:
    """
    Test if encoding is deterministic.

    Args:
        text: Text to encode multiple times
        tokenizer: Tokenizer instance
        num_trials: Number of times to encode
        add_special_tokens: Whether to add special tokens

    Returns:
        EncodingConsistency result
    """
    encodings = [
        tokenizer.encode(text, add_special_tokens=add_special_tokens) for _ in range(num_trials)
    ]

    is_consistent = all(enc == encodings[0] for enc in encodings)

    return EncodingConsistency(
        text=text,
        is_consistent=is_consistent,
        encodings=encodings,
        num_trials=num_trials,
    )


def validate_special_tokens(
    tokenizer: TokenizerProtocol,
) -> list[ValidationIssue]:
    """
    Validate special token configuration.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        List of validation issues found
    """
    issues: list[ValidationIssue] = []
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Check common special tokens
    special_attrs = [
        ("pad_token_id", "<pad>", "[PAD]"),
        ("unk_token_id", "<unk>", "[UNK]"),
        ("bos_token_id", "<s>", "[CLS]"),
        ("eos_token_id", "</s>", "[SEP]"),
    ]

    for attr, *token_names in special_attrs:
        token_id = getattr(tokenizer, attr, None)

        if token_id is None:
            # Check if token exists in vocab but isn't configured
            for name in token_names:
                if name in vocab:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.SPECIAL_TOKENS,
                            severity=ValidationSeverity.WARNING,
                            message=f"{attr} not set but {name} exists in vocabulary",
                            details={"token": name, "id": vocab[name]},
                        )
                    )
                    break
        elif token_id < 0 or token_id >= vocab_size:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SPECIAL_TOKENS,
                    severity=ValidationSeverity.ERROR,
                    message=f"{attr}={token_id} is out of vocabulary range [0, {vocab_size})",
                    details={"token_id": token_id, "vocab_size": vocab_size},
                )
            )

    return issues


def validate_vocabulary(
    tokenizer: TokenizerProtocol,
) -> list[ValidationIssue]:
    """
    Validate vocabulary integrity.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        List of validation issues found
    """
    issues: list[ValidationIssue] = []
    vocab = tokenizer.get_vocab()

    if not vocab:
        issues.append(
            ValidationIssue(
                category=ValidationCategory.VOCABULARY,
                severity=ValidationSeverity.ERROR,
                message="Vocabulary is empty",
            )
        )
        return issues

    # Check for duplicate IDs
    id_to_tokens: dict[int, list[str]] = {}
    for token, token_id in vocab.items():
        if token_id not in id_to_tokens:
            id_to_tokens[token_id] = []
        id_to_tokens[token_id].append(token)

    for token_id, tokens in id_to_tokens.items():
        if len(tokens) > 1:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.VOCABULARY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Token ID {token_id} is assigned to multiple tokens",
                    details={"token_id": token_id, "tokens": tokens},
                )
            )

    # Check for negative IDs
    negative_ids = [(t, i) for t, i in vocab.items() if i < 0]
    for token, token_id in negative_ids:
        issues.append(
            ValidationIssue(
                category=ValidationCategory.VOCABULARY,
                severity=ValidationSeverity.ERROR,
                message=f"Token '{token}' has negative ID {token_id}",
                details={"token": token, "id": token_id},
            )
        )

    # Check for gaps in ID sequence
    all_ids = set(vocab.values())
    if all_ids:
        min_id = min(all_ids)
        max_id = max(all_ids)
        expected = set(range(min_id, max_id + 1))
        missing = expected - all_ids

        if len(missing) > 0:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.VOCABULARY,
                    severity=ValidationSeverity.INFO,
                    message=f"Vocabulary has {len(missing)} gaps in ID sequence",
                    details={
                        "missing_count": len(missing),
                        "sample_missing": sorted(missing)[:10],
                    },
                )
            )

    return issues


def validate_encoding_decoding(
    tokenizer: TokenizerProtocol,
    test_texts: list[str] | None = None,
) -> list[ValidationIssue]:
    """
    Validate encode/decode functionality.

    Args:
        tokenizer: Tokenizer instance
        test_texts: Optional custom test texts

    Returns:
        List of validation issues found
    """
    issues: list[ValidationIssue] = []

    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "12345 67890",
            "Special chars: @#$%^&*()",
            "",
            " ",
            "\n\t",
        ]

    for text in test_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)

            if text and not encoded:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ENCODING,
                        severity=ValidationSeverity.WARNING,
                        message="Non-empty text encoded to empty sequence",
                        details={"text": text[:50]},
                    )
                )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENCODING,
                    severity=ValidationSeverity.ERROR,
                    message=f"Encoding failed: {e!s}",
                    details={"text": text[:50], "error": str(e)},
                )
            )
            continue

        try:
            tokenizer.decode(encoded)
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.DECODING,
                    severity=ValidationSeverity.ERROR,
                    message=f"Decoding failed: {e!s}",
                    details={"encoded": encoded[:20], "error": str(e)},
                )
            )

    return issues


def create_validation_report(
    tokenizer: TokenizerProtocol,
    tokenizer_name: str = "unknown",
    test_texts: list[str] | None = None,
    run_roundtrip: bool = True,
) -> ValidationReport:
    """
    Create a comprehensive validation report for a tokenizer.

    Args:
        tokenizer: Tokenizer instance to validate
        tokenizer_name: Name for identification
        test_texts: Optional custom test texts
        run_roundtrip: Whether to run roundtrip tests

    Returns:
        ValidationReport with all findings
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    all_issues: list[ValidationIssue] = []

    # Run all validations
    all_issues.extend(validate_vocabulary(tokenizer))
    all_issues.extend(validate_special_tokens(tokenizer))
    all_issues.extend(validate_encoding_decoding(tokenizer, test_texts))

    # Run roundtrip tests
    roundtrip_result = None
    if run_roundtrip and test_texts:
        roundtrip_result = check_batch_roundtrip(test_texts, tokenizer)
        if roundtrip_result.failed > 0:
            all_issues.append(
                ValidationIssue(
                    category=ValidationCategory.ROUNDTRIP,
                    severity=ValidationSeverity.WARNING,
                    message=f"{roundtrip_result.failed} roundtrip tests failed",
                    details={"pass_rate": roundtrip_result.pass_rate},
                )
            )

    # Count by severity
    error_count = sum(1 for i in all_issues if i.severity == ValidationSeverity.ERROR)
    warning_count = sum(1 for i in all_issues if i.severity == ValidationSeverity.WARNING)

    return ValidationReport(
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        issues=all_issues,
        roundtrip_result=roundtrip_result,
        is_valid=(error_count == 0),
        error_count=error_count,
        warning_count=warning_count,
    )


def assert_valid_tokenizer(
    tokenizer: TokenizerProtocol,
    tokenizer_name: str = "unknown",
    allow_warnings: bool = True,
) -> ValidationReport:
    """
    Validate a tokenizer and raise if invalid.

    Args:
        tokenizer: Tokenizer to validate
        tokenizer_name: Name for error messages
        allow_warnings: Whether to allow warnings (only fail on errors)

    Returns:
        ValidationReport if valid

    Raises:
        ValueError: If tokenizer fails validation
    """
    report = create_validation_report(tokenizer, tokenizer_name)

    if not report.is_valid:
        errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR]
        error_msgs = [f"  - {e.message}" for e in errors]
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' validation failed with {len(errors)} errors:\n"
            + "\n".join(error_msgs)
        )

    if not allow_warnings and report.warning_count > 0:
        warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
        warning_msgs = [f"  - {w.message}" for w in warnings]
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' has {len(warnings)} warnings:\n"
            + "\n".join(warning_msgs)
        )

    return report
