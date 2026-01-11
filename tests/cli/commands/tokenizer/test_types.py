"""Tests for tokenizer command types."""

from pathlib import Path
from unittest.mock import MagicMock

from chuk_lazarus.cli.commands.tokenizer._types import (
    BenchmarkConfig,
    BenchmarkResult,
    CompareConfig,
    CompareResult,
    DecodeConfig,
    DecodeResult,
    DoctorConfig,
    DoctorResult,
    EncodeConfig,
    FingerprintConfig,
    FingerprintResult,
    InitMethod,
    MorphMethod,
    TokenizerHealthStatus,
)


class TestEncodeConfig:
    """Tests for EncodeConfig."""

    def test_from_args_with_text(self):
        """Test creating config from args with text."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.text = "Hello world"
        args.file = None
        args.special_tokens = True

        config = EncodeConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.text == "Hello world"
        assert config.file is None
        assert config.special_tokens is True

    def test_from_args_with_file(self):
        """Test creating config from args with file."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.text = None
        args.file = Path("/path/to/file.txt")
        args.special_tokens = False

        config = EncodeConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/file.txt")
        assert config.special_tokens is False


class TestDecodeConfig:
    """Tests for DecodeConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.ids = "1,2,3,4,5"

        config = DecodeConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.ids == "1,2,3,4,5"


class TestDecodeResult:
    """Tests for DecodeResult."""

    def test_to_display(self):
        """Test result display formatting."""
        result = DecodeResult(token_ids=[1, 2, 3], decoded="Hello")

        display = result.to_display()

        assert "Token IDs: [1, 2, 3]" in display
        assert "Decoded: Hello" in display


class TestCompareConfig:
    """Tests for CompareConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        args = MagicMock()
        args.tokenizer1 = "gpt2"
        args.tokenizer2 = "llama"
        args.text = "Test text"
        args.verbose = True

        config = CompareConfig.from_args(args)

        assert config.tokenizer1 == "gpt2"
        assert config.tokenizer2 == "llama"
        assert config.text == "Test text"
        assert config.verbose is True


class TestCompareResult:
    """Tests for CompareResult."""

    def test_to_display(self):
        """Test result display formatting."""
        result = CompareResult(
            tokenizer1_count=10,
            tokenizer2_count=8,
            difference=2,
            ratio=1.25,
        )

        display = result.to_display()

        assert "Token count 1: 10" in display
        assert "Token count 2: 8" in display
        assert "Difference: +2 tokens" in display
        assert "Ratio: 1.25x" in display


class TestDoctorConfig:
    """Tests for DoctorConfig."""

    def test_from_args_basic(self):
        """Test creating config from basic args."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.verbose = False
        args.fix = False
        args.format = None
        args.output = None

        config = DoctorConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.verbose is False
        assert config.fix is False

    def test_from_args_with_fix(self):
        """Test creating config with fix mode."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.verbose = True
        args.fix = True
        args.format = "chatml"
        args.output = Path("/path/to/output")

        config = DoctorConfig.from_args(args)

        assert config.fix is True
        assert config.format == "chatml"
        assert config.output == Path("/path/to/output")


class TestDoctorResult:
    """Tests for DoctorResult."""

    def test_healthy_status(self):
        """Test healthy status display."""
        result = DoctorResult(status=TokenizerHealthStatus.HEALTHY)

        display = result.to_display()

        assert "Status: HEALTHY" in display

    def test_with_issues(self):
        """Test display with issues."""
        result = DoctorResult(
            status=TokenizerHealthStatus.ISSUES,
            issues=["Missing EOS token"],
            warnings=["No chat template"],
        )

        display = result.to_display()

        assert "Issues: 1" in display
        assert "Missing EOS token" in display
        assert "Warnings: 1" in display

    def test_with_fixes(self):
        """Test display with fixes applied."""
        result = DoctorResult(
            status=TokenizerHealthStatus.HEALTHY,
            fixes_applied=["Added chat template"],
        )

        display = result.to_display()

        assert "Fixes Applied: 1" in display
        assert "Added chat template" in display


class TestFingerprintConfig:
    """Tests for FingerprintConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.verify = None
        args.save = None
        args.strict = False

        config = FingerprintConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.verify is None
        assert config.save is None
        assert config.strict is False

    def test_from_args_with_verify(self):
        """Test config with verify option."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.verify = "abc123"
        args.save = None
        args.strict = True

        config = FingerprintConfig.from_args(args)

        assert config.verify == "abc123"
        assert config.strict is True


class TestFingerprintResult:
    """Tests for FingerprintResult."""

    def test_to_display_basic(self):
        """Test basic display."""
        result = FingerprintResult(
            fingerprint="abc123",
            vocab_size=32000,
            vocab_hash="hash_v",
            full_hash="hash_f",
            special_tokens_hash="hash_s",
            merges_hash="hash_m",
            special_tokens={"pad": 0},
        )

        display = result.to_display()

        assert "Fingerprint:   abc123" in display
        assert "Vocab size:    32,000" in display

    def test_to_display_with_verification(self):
        """Test display with verification result."""
        result = FingerprintResult(
            fingerprint="abc123",
            vocab_size=32000,
            vocab_hash="hash_v",
            full_hash="hash_f",
            special_tokens_hash="hash_s",
            merges_hash="hash_m",
            special_tokens={},
            verified=True,
            match=True,
        )

        display = result.to_display()

        assert "Verification: MATCH" in display


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.samples = 500
        args.avg_length = 50
        args.seed = 42
        args.workers = 4
        args.file = None
        args.compare = True
        args.special_tokens = True
        args.warmup = 5

        config = BenchmarkConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.samples == 500
        assert config.workers == 4
        assert config.compare is True


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_display(self):
        """Test result display."""
        result = BenchmarkResult(
            backend_type="fast",
            total_tokens=100000,
            elapsed_seconds=1.5,
            tokens_per_second=66666.67,
            samples_per_second=1000.0,
            avg_tokens_per_sample=100.0,
        )

        display = result.to_display()

        assert "Backend:      fast" in display
        assert "Total tokens: 100,000" in display


class TestEnums:
    """Tests for enum types."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert TokenizerHealthStatus.HEALTHY.value == "healthy"
        assert TokenizerHealthStatus.ISSUES.value == "issues"
        assert TokenizerHealthStatus.CRITICAL.value == "critical"

    def test_init_method_values(self):
        """Test init method enum values."""
        assert InitMethod.RANDOM.value == "random"
        assert InitMethod.NORMAL.value == "normal"
        assert InitMethod.UNIFORM.value == "uniform"

    def test_morph_method_values(self):
        """Test morph method enum values."""
        assert MorphMethod.LINEAR.value == "linear"
        assert MorphMethod.SLERP.value == "slerp"
        assert MorphMethod.GEODESIC.value == "geodesic"
