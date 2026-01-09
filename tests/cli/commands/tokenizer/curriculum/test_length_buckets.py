"""Tests for curriculum_length_buckets command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import CurriculumLengthBucketsConfig
from chuk_lazarus.cli.commands.tokenizer.curriculum.length_buckets import (
    curriculum_length_buckets,
)


class TestCurriculumLengthBucketsConfig:
    """Tests for CurriculumLengthBucketsConfig."""

    def test_from_args_basic(self):
        """Test basic config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.file = None
        args.num_buckets = 5
        args.schedule = False

        config = CurriculumLengthBucketsConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.file is None
        assert config.num_buckets == 5
        assert config.schedule is False

    def test_from_args_with_file_and_schedule(self):
        """Test config with file and schedule enabled."""
        args = MagicMock()
        args.tokenizer = "llama"
        args.file = Path("/path/to/corpus.txt")
        args.num_buckets = 10
        args.schedule = True

        config = CurriculumLengthBucketsConfig.from_args(args)

        assert config.tokenizer == "llama"
        assert config.file == Path("/path/to/corpus.txt")
        assert config.num_buckets == 10
        assert config.schedule is True


class TestCurriculumLengthBuckets:
    """Tests for curriculum_length_buckets function."""

    @patch("chuk_lazarus.cli.commands.tokenizer.curriculum.length_buckets.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    def test_length_buckets_no_texts(self, mock_load_tokenizer, mock_load_texts, capsys):
        """Test with no texts provided."""
        mock_load_tokenizer.return_value = MagicMock()
        mock_load_texts.return_value = []

        config = CurriculumLengthBucketsConfig(tokenizer="gpt2")
        curriculum_length_buckets(config)

        captured = capsys.readouterr()
        assert "Length Buckets" not in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.curriculum.length_buckets.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.curriculum.create_length_buckets")
    def test_length_buckets_basic(
        self, mock_create_buckets, mock_load_tokenizer, mock_load_texts, capsys
    ):
        """Test basic length buckets."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Short text", "Medium length text here", "Long text"]

        # Create mock buckets
        bucket1 = MagicMock()
        bucket1.min_tokens = 1
        bucket1.max_tokens = 5
        bucket1.sample_count = 1
        bucket1.avg_length = 3.0

        bucket2 = MagicMock()
        bucket2.min_tokens = 5
        bucket2.max_tokens = 10
        bucket2.sample_count = 2
        bucket2.avg_length = 7.5

        mock_create_buckets.return_value = [bucket1, bucket2]

        config = CurriculumLengthBucketsConfig(tokenizer="gpt2", num_buckets=2)
        curriculum_length_buckets(config)

        captured = capsys.readouterr()
        assert "Length Buckets" in captured.out
        assert "Bucket 1:" in captured.out
        assert "Bucket 2:" in captured.out

    @patch("chuk_lazarus.cli.commands.tokenizer.curriculum.length_buckets.load_texts")
    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.curriculum.create_length_buckets")
    @patch("chuk_lazarus.data.tokenizers.curriculum.get_curriculum_schedule")
    def test_length_buckets_with_schedule(
        self,
        mock_get_schedule,
        mock_create_buckets,
        mock_load_tokenizer,
        mock_load_texts,
        capsys,
    ):
        """Test length buckets with schedule enabled."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_texts.return_value = ["Text 1", "Text 2"]

        bucket = MagicMock()
        bucket.min_tokens = 1
        bucket.max_tokens = 10
        bucket.sample_count = 2
        bucket.avg_length = 5.0
        mock_create_buckets.return_value = [bucket]

        mock_schedule = MagicMock()
        mock_schedule.phases = [MagicMock(), MagicMock(), MagicMock()]
        mock_schedule.warmup_samples = 100
        mock_schedule.ramp_samples = 500
        mock_get_schedule.return_value = mock_schedule

        config = CurriculumLengthBucketsConfig(tokenizer="gpt2", schedule=True)
        curriculum_length_buckets(config)

        captured = capsys.readouterr()
        assert "Curriculum Schedule" in captured.out
        assert "Total phases:" in captured.out
        assert "Warmup samples:" in captured.out
        assert "Ramp samples:" in captured.out
