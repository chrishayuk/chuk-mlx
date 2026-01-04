"""Tests for tokenizer fingerprint command."""

from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import FingerprintConfig
from chuk_lazarus.cli.commands.tokenizer.health.fingerprint import tokenizer_fingerprint

LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"
FINGERPRINT_PATCH = "chuk_lazarus.data.tokenizers.fingerprint"


class TestTokenizerFingerprint:
    """Tests for tokenizer_fingerprint command."""

    def test_basic_fingerprint(self, mock_tokenizer, mock_fingerprint):
        """Test basic fingerprint generation."""
        config = FingerprintConfig(tokenizer="gpt2")

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
        ):
            result = tokenizer_fingerprint(config)

        assert result.fingerprint == "abc123"
        assert result.vocab_size == 32000
        assert result.verified is None
        assert result.match is None

    def test_fingerprint_verify_match(self, mock_tokenizer, mock_fingerprint):
        """Test fingerprint verification with match."""
        config = FingerprintConfig(
            tokenizer="gpt2",
            verify="abc123",
            strict=False,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
            patch(f"{FINGERPRINT_PATCH}.verify_fingerprint", return_value=None),
        ):
            result = tokenizer_fingerprint(config)

        assert result.verified is True
        assert result.match is True

    def test_fingerprint_verify_mismatch(self, mock_tokenizer, mock_fingerprint):
        """Test fingerprint verification with mismatch."""
        config = FingerprintConfig(
            tokenizer="gpt2",
            verify="different_hash",
            strict=False,
        )

        mismatch = MagicMock()
        mismatch.is_compatible = True
        mismatch.warnings = ["Vocab size differs"]

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
            patch(f"{FINGERPRINT_PATCH}.verify_fingerprint", return_value=mismatch),
        ):
            result = tokenizer_fingerprint(config)

        assert result.verified is True
        assert result.match is False

    def test_fingerprint_save(self, mock_tokenizer, mock_fingerprint, tmp_path):
        """Test saving fingerprint to file."""
        output_file = tmp_path / "fingerprint.json"
        config = FingerprintConfig(
            tokenizer="gpt2",
            save=output_file,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
            patch(f"{FINGERPRINT_PATCH}.save_fingerprint") as mock_save,
        ):
            result = tokenizer_fingerprint(config)

        mock_save.assert_called_once_with(mock_fingerprint, output_file)
        assert result.fingerprint == "abc123"

    def test_result_display_basic(self, mock_tokenizer, mock_fingerprint):
        """Test basic result display."""
        config = FingerprintConfig(tokenizer="gpt2")

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
        ):
            result = tokenizer_fingerprint(config)

        display = result.to_display()
        assert "Fingerprint:   abc123" in display
        assert "Vocab size:    32,000" in display

    def test_result_display_with_verification(self, mock_tokenizer, mock_fingerprint):
        """Test result display with verification."""
        config = FingerprintConfig(
            tokenizer="gpt2",
            verify="abc123",
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, return_value=mock_tokenizer),
            patch(f"{FINGERPRINT_PATCH}.compute_fingerprint", return_value=mock_fingerprint),
            patch(f"{FINGERPRINT_PATCH}.verify_fingerprint", return_value=None),
        ):
            result = tokenizer_fingerprint(config)

        display = result.to_display()
        assert "Verification: MATCH" in display
