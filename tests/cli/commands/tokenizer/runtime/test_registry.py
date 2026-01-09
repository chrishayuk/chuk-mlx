"""Tests for runtime_registry command."""

from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer.runtime.registry import (
    RuntimeRegistryWithTokenizerConfig,
    runtime_registry,
)


class TestRuntimeRegistryConfig:
    """Tests for RuntimeRegistryWithTokenizerConfig."""

    def test_from_args_basic(self):
        """Test basic config."""
        args = MagicMock()
        args.verbose = False
        args.tokenizer = None
        args.standard = False

        config = RuntimeRegistryWithTokenizerConfig.from_args(args)

        assert config.verbose is False
        assert config.tokenizer is None
        assert config.standard is False

    def test_from_args_with_options(self):
        """Test config with options."""
        args = MagicMock()
        args.verbose = True
        args.tokenizer = "gpt2"
        args.standard = True

        config = RuntimeRegistryWithTokenizerConfig.from_args(args)

        assert config.verbose is True
        assert config.tokenizer == "gpt2"
        assert config.standard is True


class TestRuntimeRegistry:
    """Tests for runtime_registry function."""

    @patch("chuk_lazarus.data.tokenizers.runtime.create_standard_registry")
    def test_registry_standard(self, mock_create_registry, capsys):
        """Test standard registry display."""
        mock_entry = MagicMock()
        mock_entry.token_id = 0
        mock_entry.token_str = "<pad>"
        mock_entry.category = MagicMock()
        mock_entry.category.value = "control"
        mock_entry.description = "Padding token"

        mock_registry = MagicMock()
        mock_registry.tokens = [mock_entry]
        mock_create_registry.return_value = mock_registry

        config = RuntimeRegistryWithTokenizerConfig(standard=True)
        runtime_registry(config)

        captured = capsys.readouterr()
        assert "Special Token Registry" in captured.out
        assert "Total tokens:" in captured.out
        assert "<pad>" in captured.out
        assert "[control]" in captured.out
        assert "Padding token" in captured.out

    @patch("chuk_lazarus.data.tokenizers.runtime.SpecialTokenRegistry")
    @patch("chuk_lazarus.data.tokenizers.runtime.TokenCategory")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_registry_from_tokenizer(
        self, mock_load_tokenizer, mock_token_category, mock_registry_cls, capsys
    ):
        """Test registry from tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.special_tokens_map = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
        }
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: 0 if x == "<pad>" else 1
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_registry = MagicMock()
        mock_registry.tokens = []
        mock_registry_cls.return_value = mock_registry

        config = RuntimeRegistryWithTokenizerConfig(tokenizer="gpt2")
        runtime_registry(config)

        captured = capsys.readouterr()
        assert "Special Token Registry" in captured.out
        # Check register was called for the special tokens
        assert mock_registry.register.call_count == 2

    @patch("chuk_lazarus.data.tokenizers.runtime.SpecialTokenRegistry")
    def test_registry_empty(self, mock_registry_cls, capsys):
        """Test empty registry."""
        mock_registry = MagicMock()
        mock_registry.tokens = []
        mock_registry_cls.return_value = mock_registry

        config = RuntimeRegistryWithTokenizerConfig()
        runtime_registry(config)

        captured = capsys.readouterr()
        assert "Total tokens: 0" in captured.out
