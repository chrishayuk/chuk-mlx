"""Tests for tokenizer shared utilities."""

from unittest.mock import patch

from chuk_lazarus.cli.commands.tokenizer._utils import load_texts


class TestLoadTexts:
    """Tests for load_texts function."""

    def test_load_texts_from_file(self, tmp_path):
        """Test loading texts from a file."""
        # Create a temp file with test content
        test_file = tmp_path / "texts.txt"
        test_file.write_text("Hello world\nTest line\n\nAnother line\n")

        texts = load_texts(test_file)

        assert texts == ["Hello world", "Test line", "Another line"]

    def test_load_texts_from_file_strips_whitespace(self, tmp_path):
        """Test that texts are stripped of whitespace."""
        test_file = tmp_path / "texts.txt"
        test_file.write_text("  leading spaces\ntrailing spaces  \n  both  \n")

        texts = load_texts(test_file)

        assert texts == ["leading spaces", "trailing spaces", "both"]

    def test_load_texts_from_file_skips_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        test_file = tmp_path / "texts.txt"
        test_file.write_text("Line 1\n\n\nLine 2\n   \nLine 3\n")

        texts = load_texts(test_file)

        assert texts == ["Line 1", "Line 2", "Line 3"]

    def test_load_texts_from_empty_file(self, tmp_path):
        """Test loading from an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        texts = load_texts(test_file)

        assert texts == []

    def test_load_texts_from_stdin_single_line(self, capsys):
        """Test loading texts from stdin with a single line."""
        with patch("builtins.input", side_effect=["Hello world", EOFError]):
            texts = load_texts(None)

        assert texts == ["Hello world"]
        captured = capsys.readouterr()
        assert "Enter texts" in captured.out

    def test_load_texts_from_stdin_multiple_lines(self, capsys):
        """Test loading texts from stdin with multiple lines."""
        with patch("builtins.input", side_effect=["Line 1", "Line 2", "Line 3", EOFError]):
            texts = load_texts(None)

        assert texts == ["Line 1", "Line 2", "Line 3"]

    def test_load_texts_from_stdin_skips_empty(self, capsys):
        """Test that empty lines from stdin are skipped."""
        with patch("builtins.input", side_effect=["Line 1", "", "  ", "Line 2", EOFError]):
            texts = load_texts(None)

        assert texts == ["Line 1", "Line 2"]

    def test_load_texts_from_stdin_empty_input(self, capsys):
        """Test loading from stdin with no input."""
        with patch("builtins.input", side_effect=[EOFError]):
            texts = load_texts(None)

        assert texts == []

    def test_load_texts_from_stdin_strips_whitespace(self, capsys):
        """Test that stdin texts are stripped."""
        with patch("builtins.input", side_effect=["  spaces  ", EOFError]):
            texts = load_texts(None)

        assert texts == ["spaces"]
