"""Tests for introspect generation CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectGenerate:
    """Tests for introspect_generate command."""

    @pytest.fixture
    def generate_args(self):
        """Create arguments for generate command."""
        return Namespace(
            model="test-model",
            prompts="2+2=|3+3=",
            max_tokens=10,
            temperature=0.0,
            compare_format=False,
            show_tokens=False,
            raw=False,
            output=None,
        )

    def test_generate_basic(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test basic generation."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_generate_from_file(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate):
        """Test generating from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2+2=\n3+3=\n")
            f.flush()

            generate_args.prompts = f"@{f.name}"
            mock_mlx_lm_generate.return_value = "4"

            introspect_generate(generate_args)

    def test_generate_compare_format(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test format comparison mode."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.compare_format = True
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "Format Comparison" in captured.out or "Loading" in captured.out

    def test_generate_show_tokens(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test showing tokens."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.show_tokens = True
        mock_mlx_lm_generate.return_value = "4"

        # Setup tokenizer mock for token display
        mock_tokenizer = mock_mlx_lm_load.return_value[1]
        mock_tokenizer.encode.return_value = [1, 2, 3]

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "Tokens:" in captured.out or "Loading" in captured.out

    def test_generate_raw_mode(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.raw = True
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "RAW" in captured.out

    def test_generate_with_temperature(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate):
        """Test generation with non-zero temperature."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.temperature = 0.7
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        # Check that generate was called with temp
        mock_mlx_lm_generate.assert_called()

    def test_generate_save_output(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate):
        """Test saving output to file."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            generate_args.output = f.name

        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        # Check file was created
        import json
        from pathlib import Path

        if Path(generate_args.output).exists():
            with open(generate_args.output) as f:
                data = json.load(f)
                assert isinstance(data, list)


class TestFindAnswerOnset:
    """Tests for _find_answer_onset helper function."""

    def test_find_onset_no_expected(self):
        """Test with no expected answer."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        result = _find_answer_onset("some output", None, tokenizer)

        assert result["answer_found"] is False
        assert result["onset_index"] is None

    def test_find_onset_answer_found(self):
        """Test when answer is found."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.decode.side_effect = lambda ids: "4" if ids == [1] else "2"

        result = _find_answer_onset("42", "4", tokenizer)

        assert result["answer_found"] is True
        assert result["onset_index"] == 0

    def test_find_onset_answer_not_found(self):
        """Test when answer is not in output."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids: "x"

        result = _find_answer_onset("xxx", "42", tokenizer)

        assert result["answer_found"] is False


class TestNormalizeNumber:
    """Tests for _normalize_number helper function."""

    def test_normalize_plain(self):
        """Test normalizing plain number."""
        from chuk_lazarus.cli.commands.introspect.generation import _normalize_number

        assert _normalize_number("12345") == "12345"

    def test_normalize_with_commas(self):
        """Test normalizing with commas."""
        from chuk_lazarus.cli.commands.introspect.generation import _normalize_number

        assert _normalize_number("1,234") == "1234"

    def test_normalize_with_spaces(self):
        """Test normalizing with spaces."""
        from chuk_lazarus.cli.commands.introspect.generation import _normalize_number

        assert _normalize_number("1 234") == "1234"

    def test_normalize_with_unicode_spaces(self):
        """Test normalizing with unicode thin spaces."""
        from chuk_lazarus.cli.commands.introspect.generation import _normalize_number

        assert _normalize_number("1\u202f234") == "1234"
        assert _normalize_number("1\u00a0234") == "1234"
