"""Tests for tokenizer_vocab command."""

from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import VocabConfig
from chuk_lazarus.cli.commands.tokenizer.core.vocab import tokenizer_vocab


class TestVocabConfig:
    """Tests for VocabConfig."""

    def test_from_args_basic(self):
        """Test basic config."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.show_all = False
        args.search = None
        args.limit = 20
        args.chunk_size = 100
        args.pause = False

        config = VocabConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.show_all is False
        assert config.search is None
        assert config.limit == 20
        assert config.chunk_size == 100
        assert config.pause is False

    def test_from_args_with_search(self):
        """Test config with search."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.show_all = False
        args.search = "test"
        args.limit = 50
        args.chunk_size = 100
        args.pause = False

        config = VocabConfig.from_args(args)

        assert config.search == "test"
        assert config.limit == 50


class TestTokenizerVocab:
    """Tests for tokenizer_vocab function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_vocab_basic(self, mock_load_tokenizer, capsys):
        """Test basic vocab display."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 0, "token2": 1, "token3": 2}
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.unk_token_id = 3
        mock_load_tokenizer.return_value = mock_tokenizer

        config = VocabConfig(tokenizer="gpt2")
        tokenizer_vocab(config)

        captured = capsys.readouterr()
        assert "Vocabulary Statistics" in captured.out
        assert "Total tokens: 3" in captured.out
        assert "Pad token ID: 0" in captured.out
        assert "EOS token ID: 2" in captured.out

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_vocab_search(self, mock_load_tokenizer, capsys):
        """Test vocab search."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            "hello": 0,
            "world": 1,
            "test_hello": 2,
            "hello_world": 3,
        }
        mock_tokenizer.decode.return_value = "token"
        mock_load_tokenizer.return_value = mock_tokenizer

        config = VocabConfig(tokenizer="gpt2", search="hello", limit=10)
        tokenizer_vocab(config)

        captured = capsys.readouterr()
        assert "Tokens containing 'hello'" in captured.out
        # Should find 3 tokens: hello, test_hello, hello_world

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.token_display.TokenDisplayUtility")
    def test_vocab_show_all(self, mock_display_cls, mock_load_tokenizer, capsys):
        """Test vocab show all."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"token1": 0}
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_display = MagicMock()
        mock_display_cls.return_value = mock_display

        config = VocabConfig(tokenizer="gpt2", show_all=True, chunk_size=50, pause=True)
        tokenizer_vocab(config)

        mock_display.display_full_vocabulary.assert_called_once_with(
            chunk_size=50, pause_between_chunks=True
        )
