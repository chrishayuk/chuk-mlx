"""Tests for research_soft_tokens command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from chuk_lazarus.cli.commands.tokenizer._types import InitMethod, ResearchSoftTokensConfig
from chuk_lazarus.cli.commands.tokenizer.research.soft_tokens import research_soft_tokens


class TestResearchSoftTokensConfig:
    """Tests for ResearchSoftTokensConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.num_tokens = 10
        args.embedding_dim = 768
        args.prefix = "soft"
        args.init_method = "normal"
        args.init_std = 0.02
        args.output = None

        config = ResearchSoftTokensConfig.from_args(args)

        assert config.num_tokens == 10
        assert config.embedding_dim == 768
        assert config.prefix == "soft"
        assert config.init_method == InitMethod.NORMAL
        assert config.init_std == 0.02
        assert config.output is None

    def test_from_args_with_options(self):
        """Test config with all options."""
        args = MagicMock()
        args.num_tokens = 20
        args.embedding_dim = 1024
        args.prefix = "prompt"
        args.init_method = "uniform"
        args.init_std = 0.1
        args.output = Path("/path/to/bank.json")

        config = ResearchSoftTokensConfig.from_args(args)

        assert config.num_tokens == 20
        assert config.embedding_dim == 1024
        assert config.prefix == "prompt"
        assert config.init_method == InitMethod.UNIFORM
        assert config.output == Path("/path/to/bank.json")


class TestResearchSoftTokens:
    """Tests for research_soft_tokens function."""

    @patch("chuk_lazarus.data.tokenizers.research.create_prompt_tuning_bank")
    @patch("chuk_lazarus.data.tokenizers.research.InitializationMethod")
    def test_soft_tokens_basic(self, mock_init_method_cls, mock_create_bank, capsys):
        """Test basic soft token creation."""
        # Create mock token
        mock_token = MagicMock()
        mock_token.token = MagicMock()
        mock_token.token.name = "soft_0"
        mock_token.token.token_id = 0
        mock_token.embedding_array = np.random.randn(768).astype(np.float32)

        # Create mock bank
        mock_bank = MagicMock()
        mock_bank.name = "soft_tokens"
        mock_bank.embedding_dim = 768
        mock_bank.tokens = [mock_token]
        mock_create_bank.return_value = mock_bank

        mock_init_method = MagicMock()
        mock_init_method.value = "random_normal"
        mock_init_method_cls.return_value = mock_init_method

        config = ResearchSoftTokensConfig(
            num_tokens=10,
            embedding_dim=768,
            prefix="soft",
            init_method=InitMethod.NORMAL,
        )
        research_soft_tokens(config)

        captured = capsys.readouterr()
        assert "Soft Token Bank" in captured.out
        assert "Name:" in captured.out
        assert "Embedding dim:" in captured.out
        assert "Num tokens:" in captured.out
