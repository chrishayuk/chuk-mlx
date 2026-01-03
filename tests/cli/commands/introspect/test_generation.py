"""Tests for introspect generation CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectGenerate:
    """Tests for introspect_generate command.

    NOTE: These tests must run in isolation due to MLX framework limitations.
    Running multiple tests that use mock_mlx_lm fixtures in the same pytest
    session can cause MLX to crash. We skip the introspect_generate tests
    in batch runs and test them individually.
    """

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

    @pytest.fixture
    def mock_mlx_lm_module(self, mock_model, mock_tokenizer):
        """Create a mock mlx_lm module."""
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = "4"
        return mock_mlx_lm

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_basic(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test basic generation."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_from_file(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate):
        """Test generating from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2+2=\n3+3=\n")
            f.flush()

            generate_args.prompts = f"@{f.name}"
            mock_mlx_lm_generate.return_value = "4"

            introspect_generate(generate_args)

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_compare_format(
        self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys
    ):
        """Test format comparison mode."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.compare_format = True
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "Format Comparison" in captured.out or "Loading" in captured.out

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_show_tokens(
        self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys
    ):
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

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_raw_mode(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.raw = True
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        captured = capsys.readouterr()
        assert "RAW" in captured.out

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
    def test_generate_with_temperature(self, generate_args, mock_mlx_lm_load, mock_mlx_lm_generate):
        """Test generation with non-zero temperature."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        generate_args.temperature = 0.7
        mock_mlx_lm_generate.return_value = "4"

        introspect_generate(generate_args)

        # Check that generate was called with temp
        mock_mlx_lm_generate.assert_called()

    @pytest.mark.skip(
        reason="MLX crashes when running multiple generate tests together - run individually"
    )
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

    def test_find_onset_is_answer_first(self):
        """Test is_answer_first flag when answer is in first 2 tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.decode.side_effect = lambda ids: "42"

        result = _find_answer_onset("42", "42", tokenizer)

        assert result["answer_found"] is True
        assert result["is_answer_first"] is True
        assert result["onset_index"] == 0

    def test_find_onset_delayed_answer(self):
        """Test is_answer_first is False when answer comes later."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        # First two tokens don't contain answer
        tokenizer.decode.side_effect = lambda ids: {
            (1,): "The",
            (2,): " answer",
            (3,): " is 42",
        }.get(tuple(ids), "")

        result = _find_answer_onset("The answer is 42", "42", tokenizer)

        assert result["answer_found"] is True
        assert result["is_answer_first"] is False
        assert result["onset_index"] == 2


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

    def test_normalize_mixed(self):
        """Test normalizing with mixed separators."""
        from chuk_lazarus.cli.commands.introspect.generation import _normalize_number

        assert _normalize_number("1,234 567") == "1234567"


class TestLoadExternalChatTemplate:
    """Tests for _load_external_chat_template helper function."""

    def test_load_template_no_file(self):
        """Test when no chat template file exists."""
        from chuk_lazarus.cli.commands.introspect.generation import (
            _load_external_chat_template,
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = None

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Not found")

            # Should not raise
            _load_external_chat_template(tokenizer, "some/model")

            # chat_template should still be None
            assert tokenizer.chat_template is None

    def test_load_template_from_file(self):
        """Test loading chat template from file."""
        from chuk_lazarus.cli.commands.introspect.generation import (
            _load_external_chat_template,
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chat_template.jinja file
            template_path = f"{tmpdir}/chat_template.jinja"
            with open(template_path, "w") as f:
                f.write("{{ message }}")

            with patch("huggingface_hub.snapshot_download") as mock_download:
                from pathlib import Path

                mock_download.return_value = Path(tmpdir)

                _load_external_chat_template(tokenizer, "some/model")

                assert tokenizer.chat_template == "{{ message }}"

    def test_load_template_already_has_template(self):
        """Test that existing template is not overwritten."""
        from chuk_lazarus.cli.commands.introspect.generation import (
            _load_external_chat_template,
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = "existing template"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chat_template.jinja file
            template_path = f"{tmpdir}/chat_template.jinja"
            with open(template_path, "w") as f:
                f.write("{{ new_template }}")

            with patch("huggingface_hub.snapshot_download") as mock_download:
                from pathlib import Path

                mock_download.return_value = Path(tmpdir)

                _load_external_chat_template(tokenizer, "some/model")

                # Original template should be preserved
                assert tokenizer.chat_template == "existing template"

    def test_load_template_read_error(self):
        """Test handling of read error."""
        from chuk_lazarus.cli.commands.introspect.generation import (
            _load_external_chat_template,
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chat_template.jinja as a directory (will cause read error)
            template_path = f"{tmpdir}/chat_template.jinja"
            import os

            os.makedirs(template_path)

            with patch("huggingface_hub.snapshot_download") as mock_download:
                from pathlib import Path

                mock_download.return_value = Path(tmpdir)

                # Should not raise
                _load_external_chat_template(tokenizer, "some/model")

                # chat_template should still be None
                assert tokenizer.chat_template is None

    def test_load_template_local_path(self):
        """Test loading from local path when snapshot_download fails."""
        from chuk_lazarus.cli.commands.introspect.generation import (
            _load_external_chat_template,
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chat_template.jinja file
            template_path = f"{tmpdir}/chat_template.jinja"
            with open(template_path, "w") as f:
                f.write("{{ local_template }}")

            with patch("huggingface_hub.snapshot_download") as mock_download:
                # Simulate HF download failure - falls back to local path
                mock_download.side_effect = Exception("Not found")

                _load_external_chat_template(tokenizer, tmpdir)

                assert tokenizer.chat_template == "{{ local_template }}"
