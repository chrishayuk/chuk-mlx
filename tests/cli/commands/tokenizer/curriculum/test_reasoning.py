"""Tests for curriculum_reasoning_density command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import CurriculumReasoningConfig
from chuk_lazarus.cli.commands.tokenizer.curriculum.reasoning import (
    curriculum_reasoning_density,
)


class TestCurriculumReasoningConfig:
    """Tests for CurriculumReasoningConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.descending = True

        config = CurriculumReasoningConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.descending is True

    def test_from_args_with_file(self):
        """Test config with file and descending=False."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.descending = False

        config = CurriculumReasoningConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.descending is False


class TestCurriculumReasoningDensity:
    """Tests for curriculum_reasoning_density function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.curriculum.reasoning.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_reasoning_density_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = CurriculumReasoningConfig(tokenizer="gpt2")
        curriculum_reasoning_density(config)

        captured = capsys.readouterr()
        assert "Reasoning Density" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.curriculum.reasoning.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.curriculum.sort_by_reasoning_density")
    @patch("chuk_lazarus.data.tokenizers.curriculum.get_difficulty_percentiles")
    def test_reasoning_density_basic(
        self,
        mock_get_percentiles,
        mock_sort_by_density,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test basic reasoning density analysis."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        texts = ["Simple text", "Complex reasoning step 1 -> step 2", "Math: 2+2=4"]
        mock_load_texts.return_value = texts

        # Create mock score
        mock_score = MagicMock()
        mock_score.text_index = 1
        mock_score.score = 0.85
        mock_sort_by_density.return_value = [mock_score]

        mock_percentiles = MagicMock()
        mock_percentiles.mean = 0.5
        mock_percentiles.p25 = 0.25
        mock_percentiles.p50 = 0.5
        mock_percentiles.p75 = 0.75
        mock_percentiles.p90 = 0.9
        mock_get_percentiles.return_value = mock_percentiles

        config = CurriculumReasoningConfig(tokenizer="gpt2")
        curriculum_reasoning_density(config)

        captured = capsys.readouterr()
        assert "Reasoning Density" in captured.out
        assert "Mean score:" in captured.out
        assert "P25:" in captured.out
        assert "P50 (median):" in captured.out
        assert "P75:" in captured.out
        assert "P90:" in captured.out
        assert "Top" in captured.out
        assert "by reasoning density" in captured.out
