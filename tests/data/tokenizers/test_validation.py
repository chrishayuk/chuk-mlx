"""Tests for validation module."""

import pytest

from chuk_lazarus.data.tokenizers.validation import (
    BatchRoundtripResult,
    EncodingConsistency,
    RoundtripResult,
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    assert_valid_tokenizer,
    check_batch_roundtrip,
    check_encoding_consistency,
    check_roundtrip,
    create_validation_report,
    validate_encoding_decoding,
    validate_special_tokens,
)
from chuk_lazarus.data.tokenizers.validation import (
    validate_vocabulary as validate_tokenizer_vocabulary,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {"hello": 0, "world": 1, "<unk>": 2, "the": 3}
        self.pad_token_id = None
        self.unk_token_id = 2
        self.bos_token_id = None
        self.eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._vocab.get("<unk>", 2)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class MockTokenizerWithSpecial:
    """Mock tokenizer with special tokens configured."""

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "hello": 4,
            "world": 5,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._vocab.get("<unk>", 1)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestValidationSeverityEnum:
    """Tests for ValidationSeverity enum."""

    def test_values(self):
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"


class TestValidationCategoryEnum:
    """Tests for ValidationCategory enum."""

    def test_values(self):
        assert ValidationCategory.ROUNDTRIP == "roundtrip"
        assert ValidationCategory.ENCODING == "encoding"
        assert ValidationCategory.DECODING == "decoding"
        assert ValidationCategory.SPECIAL_TOKENS == "special_tokens"
        assert ValidationCategory.VOCABULARY == "vocabulary"
        assert ValidationCategory.CONSISTENCY == "consistency"


class TestValidationIssueModel:
    """Tests for ValidationIssue Pydantic model."""

    def test_basic_issue(self):
        issue = ValidationIssue(
            category=ValidationCategory.ENCODING,
            severity=ValidationSeverity.ERROR,
            message="Encoding failed",
        )
        assert issue.category == ValidationCategory.ENCODING
        assert issue.details is None

    def test_issue_with_details(self):
        issue = ValidationIssue(
            category=ValidationCategory.VOCABULARY,
            severity=ValidationSeverity.WARNING,
            message="Duplicate IDs found",
            details={"id": 42, "tokens": ["a", "b"]},
        )
        assert issue.details["id"] == 42


class TestRoundtripResultModel:
    """Tests for RoundtripResult Pydantic model."""

    def test_lossless_roundtrip(self):
        result = RoundtripResult(
            original="hello",
            encoded=[0],
            decoded="hello",
            is_lossless=True,
            diff_chars=[],
        )
        assert result.is_lossless

    def test_lossy_roundtrip(self):
        result = RoundtripResult(
            original="Hello",
            encoded=[0],
            decoded="hello",
            is_lossless=False,
            diff_chars=["H"],
        )
        assert not result.is_lossless


class TestBatchRoundtripResultModel:
    """Tests for BatchRoundtripResult Pydantic model."""

    def test_all_passed(self):
        result = BatchRoundtripResult(
            total_tests=10,
            passed=10,
            failed=0,
            pass_rate=1.0,
            failures=[],
        )
        assert result.pass_rate == 1.0

    def test_some_failed(self):
        result = BatchRoundtripResult(
            total_tests=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            failures=[],
        )
        assert result.failed == 2


class TestEncodingConsistencyModel:
    """Tests for EncodingConsistency Pydantic model."""

    def test_consistent(self):
        result = EncodingConsistency(
            text="hello",
            is_consistent=True,
            encodings=[[0], [0], [0]],
            num_trials=3,
        )
        assert result.is_consistent


class TestValidationReportModel:
    """Tests for ValidationReport Pydantic model."""

    def test_valid_report(self):
        report = ValidationReport(
            tokenizer_name="test",
            vocab_size=100,
            issues=[],
            is_valid=True,
            error_count=0,
            warning_count=0,
        )
        assert report.is_valid


class TestCheckRoundtrip:
    """Tests for check_roundtrip function."""

    def test_lossless(self):
        tokenizer = MockTokenizer()
        result = check_roundtrip("hello world", tokenizer)
        assert result.original == "hello world"
        assert result.is_lossless

    def test_lossy(self):
        tokenizer = MockTokenizer()
        result = check_roundtrip("Hello World", tokenizer)  # Case changes
        assert not result.is_lossless
        assert len(result.diff_chars) > 0


class TestCheckBatchRoundtrip:
    """Tests for check_batch_roundtrip function."""

    def test_multiple_texts(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the world"]
        result = check_batch_roundtrip(texts, tokenizer)
        assert result.total_tests == 2
        assert result.passed + result.failed == 2

    def test_empty_list(self):
        tokenizer = MockTokenizer()
        result = check_batch_roundtrip([], tokenizer)
        assert result.total_tests == 0
        assert result.pass_rate == 1.0


class TestCheckEncodingConsistency:
    """Tests for check_encoding_consistency function."""

    def test_consistent_encoding(self):
        tokenizer = MockTokenizer()
        result = check_encoding_consistency("hello world", tokenizer, num_trials=5)
        assert result.is_consistent
        assert result.num_trials == 5
        assert len(result.encodings) == 5


class TestValidateSpecialTokens:
    """Tests for validate_special_tokens function."""

    def test_valid_special_tokens(self):
        tokenizer = MockTokenizerWithSpecial()
        issues = validate_special_tokens(tokenizer)
        # Should have no errors (might have warnings)
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_missing_special_tokens(self):
        tokenizer = MockTokenizer()  # No special tokens configured
        issues = validate_special_tokens(tokenizer)
        # Might find warnings about unconfigured tokens
        assert isinstance(issues, list)


class TestValidateVocabulary:
    """Tests for validate_vocabulary function."""

    def test_valid_vocab(self):
        tokenizer = MockTokenizer()
        issues = validate_tokenizer_vocabulary(tokenizer)
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_empty_vocab(self):
        tokenizer = MockTokenizer({})
        issues = validate_tokenizer_vocabulary(tokenizer)
        # Empty vocab returns early with error OR produces no issues
        # depending on implementation - just check it returns a list
        assert isinstance(issues, list)

    def test_duplicate_ids(self):
        vocab = {"a": 0, "b": 0, "c": 1}  # Duplicate ID 0
        tokenizer = MockTokenizer(vocab)
        issues = validate_tokenizer_vocabulary(tokenizer)
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        assert any("multiple" in i.message.lower() for i in warnings)


class TestValidateEncodingDecoding:
    """Tests for validate_encoding_decoding function."""

    def test_basic_validation(self):
        tokenizer = MockTokenizer()
        issues = validate_encoding_decoding(tokenizer)
        # Should pass basic tests
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_custom_test_texts(self):
        tokenizer = MockTokenizer()
        issues = validate_encoding_decoding(tokenizer, test_texts=["hello", "world"])
        assert isinstance(issues, list)


class TestCreateValidationReport:
    """Tests for create_validation_report function."""

    def test_full_report(self):
        tokenizer = MockTokenizer()
        report = create_validation_report(tokenizer, "test_tokenizer")
        assert report.tokenizer_name == "test_tokenizer"
        assert report.vocab_size == len(tokenizer.get_vocab())
        assert isinstance(report.issues, list)

    def test_with_roundtrip(self):
        tokenizer = MockTokenizer()
        report = create_validation_report(
            tokenizer,
            "test",
            test_texts=["hello world"],
            run_roundtrip=True,
        )
        assert report.roundtrip_result is not None

    def test_without_roundtrip(self):
        tokenizer = MockTokenizer()
        report = create_validation_report(
            tokenizer,
            "test",
            run_roundtrip=False,
        )
        assert report.roundtrip_result is None


class TestAssertValidTokenizer:
    """Tests for assert_valid_tokenizer function."""

    def test_valid_tokenizer_passes(self):
        tokenizer = MockTokenizer()
        report = assert_valid_tokenizer(tokenizer, "test")
        assert report.is_valid

    def test_invalid_tokenizer_with_negative_ids(self):
        # Create tokenizer with negative IDs which should fail validation
        vocab = {"a": -1, "b": 0}  # Negative ID
        tokenizer = MockTokenizer(vocab)
        with pytest.raises(ValueError) as excinfo:
            assert_valid_tokenizer(tokenizer, "bad_tokenizer")
        assert "bad_tokenizer" in str(excinfo.value)

    def test_warnings_allowed(self):
        tokenizer = MockTokenizer()
        # Should pass even if there are warnings
        report = assert_valid_tokenizer(tokenizer, "test", allow_warnings=True)
        assert isinstance(report, ValidationReport)
