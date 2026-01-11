"""Tests for tokenizer_fingerprint command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import FingerprintConfig
from chuk_lazarus.cli.commands.tokenizer.health.fingerprint import tokenizer_fingerprint


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
        args.tokenizer = "llama"
        args.verify = "abc123"
        args.save = None
        args.strict = True

        config = FingerprintConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.verify == "abc123"
        assert config.strict is True

    def test_from_args_with_save(self):
        """Test config with save option."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.verify = None
        args.save = Path("/path/to/fingerprint.json")
        args.strict = False

        config = FingerprintConfig.from_args(args)

        assert config.save == Path("/path/to/fingerprint.json")


class TestTokenizerFingerprint:
    """Tests for tokenizer_fingerprint function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    def test_fingerprint_display(self, mock_compute_fp, mock_load_tokenizer, capsys):
        """Test basic fingerprint display."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_fp = MagicMock()
        mock_fp.fingerprint = "gpt2-v1-abc123"
        mock_fp.full_hash = "fullhash123"
        mock_fp.vocab_size = 50257
        mock_fp.vocab_hash = "vochash"
        mock_fp.special_tokens_hash = "spechash"
        mock_fp.merges_hash = "mergehash"
        mock_fp.special_tokens = {"pad_token_id": 0, "eos_token_id": 50256}
        mock_compute_fp.return_value = mock_fp

        config = FingerprintConfig(tokenizer="gpt2")
        result = tokenizer_fingerprint(config)

        captured = capsys.readouterr()
        assert "Tokenizer Fingerprint" in captured.out
        assert "gpt2-v1-abc123" in captured.out
        assert result.fingerprint == "gpt2-v1-abc123"
        assert result.vocab_size == 50257

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.verify_fingerprint")
    def test_fingerprint_verify_match(
        self, mock_verify, mock_compute_fp, mock_load_tokenizer, capsys
    ):
        """Test fingerprint verification - match."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_fp = MagicMock()
        mock_fp.fingerprint = "gpt2-v1-abc123"
        mock_fp.full_hash = "fullhash123"
        mock_fp.vocab_size = 50257
        mock_fp.vocab_hash = "vochash"
        mock_fp.special_tokens_hash = "spechash"
        mock_fp.merges_hash = "mergehash"
        mock_fp.special_tokens = {}
        mock_compute_fp.return_value = mock_fp

        mock_verify.return_value = None  # None means match

        config = FingerprintConfig(tokenizer="gpt2", verify="gpt2-v1-abc123")
        result = tokenizer_fingerprint(config)

        captured = capsys.readouterr()
        assert "Fingerprint Verification" in captured.out
        assert "MATCH" in captured.out
        assert result.verified is True
        assert result.match is True

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.verify_fingerprint")
    def test_fingerprint_verify_mismatch(
        self, mock_verify, mock_compute_fp, mock_load_tokenizer, capsys
    ):
        """Test fingerprint verification - mismatch but compatible."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_fp = MagicMock()
        mock_fp.fingerprint = "gpt2-v1-abc123"
        mock_fp.full_hash = "fullhash123"
        mock_fp.vocab_size = 50257
        mock_fp.vocab_hash = "vochash"
        mock_fp.special_tokens_hash = "spechash"
        mock_fp.merges_hash = "mergehash"
        mock_fp.special_tokens = {}
        mock_compute_fp.return_value = mock_fp

        mock_mismatch = MagicMock()
        mock_mismatch.is_compatible = True
        mock_mismatch.warnings = ["Minor version difference"]
        mock_verify.return_value = mock_mismatch

        config = FingerprintConfig(tokenizer="gpt2", verify="gpt2-v1-xyz789")
        result = tokenizer_fingerprint(config)

        captured = capsys.readouterr()
        assert "MISMATCH" in captured.out
        assert "Compatible: Yes" in captured.out
        assert result.match is False

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.save_fingerprint")
    def test_fingerprint_save(
        self, mock_save, mock_compute_fp, mock_load_tokenizer, capsys, tmp_path
    ):
        """Test fingerprint save."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_fp = MagicMock()
        mock_fp.fingerprint = "gpt2-v1-abc123"
        mock_fp.full_hash = "fullhash123"
        mock_fp.vocab_size = 50257
        mock_fp.vocab_hash = "vochash"
        mock_fp.special_tokens_hash = "spechash"
        mock_fp.merges_hash = "mergehash"
        mock_fp.special_tokens = {}
        mock_compute_fp.return_value = mock_fp

        save_path = tmp_path / "fingerprint.json"
        config = FingerprintConfig(tokenizer="gpt2", save=save_path)
        tokenizer_fingerprint(config)

        captured = capsys.readouterr()
        assert "Fingerprint Saved" in captured.out
        mock_save.assert_called_once_with(mock_fp, save_path)
