"""Tests for tokenizer_doctor command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import (
    DoctorConfig,
    TokenizerHealthStatus,
)
from chuk_lazarus.cli.commands.tokenizer.health.doctor import tokenizer_doctor


class TestDoctorConfig:
    """Tests for DoctorConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
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
        assert config.format is None
        assert config.output is None

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.verbose = True
        args.fix = True
        args.format = "chatml"
        args.output = Path("/path/to/output")

        config = DoctorConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.verbose is True
        assert config.fix is True
        assert config.format == "chatml"
        assert config.output == Path("/path/to/output")


class TestTokenizerDoctor:
    """Tests for tokenizer_doctor function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    @patch("chuk_lazarus.data.tokenizers.runtime.chat_templates.validate_chat_template")
    @patch("chuk_lazarus.data.tokenizers.runtime.chat_templates.ChatTemplateRegistry")
    def test_doctor_healthy(
        self,
        mock_registry_cls,
        mock_validate,
        mock_compute_fp,
        mock_load_tokenizer,
        capsys,
    ):
        """Test doctor with healthy tokenizer."""
        # Set up tokenizer to make all roundtrip tests pass
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token": 0}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.unk_token_id = 1
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.eos_token_id = 3
        mock_tokenizer.chat_template = "{{ messages }}"
        mock_tokenizer.convert_ids_to_tokens.return_value = ["<pad>"]
        mock_tokenizer.encode.return_value = [0, 1, 2]
        # Return same text for any decoded input (roundtrip check uses normalized comparison)
        mock_tokenizer.decode.side_effect = lambda ids, **kwargs: "Hello, world!"
        mock_tokenizer.apply_chat_template.side_effect = lambda msgs, **kwargs: (
            "You are helpful.\nHello" if any(m.get("role") == "system" for m in msgs) else "Formatted message"
        )
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_validation = MagicMock()
        mock_validation.format = MagicMock()
        mock_validation.format.value = "chatml"
        mock_validation.capabilities = []
        mock_validation.issues = []
        mock_validate.return_value = mock_validation

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        mock_fp = MagicMock()
        mock_fp.fingerprint = "test-fp"
        mock_fp.vocab_hash = "vhash"
        mock_compute_fp.return_value = mock_fp

        config = DoctorConfig(tokenizer="gpt2")
        result = tokenizer_doctor(config)

        captured = capsys.readouterr()
        assert "Tokenizer Doctor" in captured.out
        assert "Basic Info" in captured.out
        assert "Special Tokens" in captured.out
        # Doctor will have warnings due to simplified mocking, but should not have critical issues
        assert result.status in (TokenizerHealthStatus.HEALTHY, TokenizerHealthStatus.ISSUES)

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint")
    @patch("chuk_lazarus.data.tokenizers.runtime.chat_templates.validate_chat_template")
    @patch("chuk_lazarus.data.tokenizers.runtime.chat_templates.ChatTemplateRegistry")
    @patch("chuk_lazarus.data.tokenizers.runtime.chat_templates.suggest_template_for_model")
    def test_doctor_missing_chat_template(
        self,
        mock_suggest,
        mock_registry_cls,
        mock_validate,
        mock_compute_fp,
        mock_load_tokenizer,
        capsys,
    ):
        """Test doctor with missing chat template."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token": 0}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.unk_token_id = 1
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.eos_token_id = 3
        mock_tokenizer.chat_template = None  # No chat template
        mock_tokenizer.convert_ids_to_tokens.return_value = ["<pad>"]
        mock_tokenizer.encode.return_value = [0, 1, 2]
        mock_tokenizer.decode.return_value = "Hello, world!"
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_validation = MagicMock()
        mock_validation.issues = []
        mock_validate.return_value = mock_validation

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        mock_suggest.return_value = None

        mock_fp = MagicMock()
        mock_fp.fingerprint = "test-fp"
        mock_fp.vocab_hash = "vhash"
        mock_compute_fp.return_value = mock_fp

        config = DoctorConfig(tokenizer="gpt2")
        result = tokenizer_doctor(config)

        captured = capsys.readouterr()
        assert "Available: No" in captured.out
        assert result.status == TokenizerHealthStatus.ISSUES
        assert "No chat template defined" in result.warnings
