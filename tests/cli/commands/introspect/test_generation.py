"""Tests for introspect generation CLI commands."""

import json
import tempfile
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
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

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids: "4" if len(ids) == 1 else "test output"
        tokenizer.chat_template = "{{ messages }}"
        tokenizer.eos_token_id = 2
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.config = MagicMock()
        return model

    @contextmanager
    def mock_mlx_and_introspection(
        self, mock_model, mock_tokenizer, generate_return_value="4", setup_introspection_module=None
    ):
        """Context manager to mock mlx_lm and introspection modules."""
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate.return_value = generate_return_value

        # Setup introspection module mock with apply_chat_template and extract_expected_answer
        if setup_introspection_module:
            mock_introspection = setup_introspection_module
        else:
            mock_introspection = MagicMock()

        mock_introspection.apply_chat_template = MagicMock(side_effect=lambda t, p: p)
        mock_introspection.extract_expected_answer = MagicMock(return_value=None)

        with (
            patch.dict(
                "sys.modules",
                {"mlx_lm": mock_mlx_lm, "chuk_lazarus.introspection": mock_introspection},
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.generation._load_external_chat_template"
            ) as mock_load_tpl,
        ):
            yield {
                "mlx_lm": mock_mlx_lm,
                "load_template": mock_load_tpl,
                "apply_template": mock_introspection.apply_chat_template,
                "extract_answer": mock_introspection.extract_expected_answer,
            }

    def test_generate_basic(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test basic generation with temperature=0."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["apply_template"].side_effect = lambda t, p: f"<chat>{p}</chat>"
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "Loading model: test-model" in captured.out
            assert "CHAT" in captured.out
            assert "2+2=" in captured.out
            assert "3+3=" in captured.out

    def test_generate_raw_mode(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.raw = True

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer):
            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "Mode: RAW (no chat template)" in captured.out

    def test_generate_no_chat_template(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test when model has no chat template."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        mock_tokenizer.chat_template = None

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer):
            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "Mode: RAW (model has no chat template)" in captured.out

    def test_generate_from_file(self, generate_args, mock_model, mock_tokenizer):
        """Test generating from file."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2+2=\n3+3=\n")
            f.flush()
            temp_path = f.name

        try:
            generate_args.prompts = f"@{temp_path}"

            with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
                introspect_generate(generate_args)

                # Should have called generate for each prompt
                assert mocks["mlx_lm"].generate.call_count == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_compare_format(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test format comparison mode."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.compare_format = True
        generate_args.prompts = "2+2="

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "Format Comparison Summary" in captured.out
            assert "Legend:" in captured.out
            # Should generate for both with and without space
            assert mocks["mlx_lm"].generate.call_count == 2

    def test_generate_show_tokens(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test showing tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.show_tokens = True
        generate_args.prompts = "2+2="

        # Mock tokenizer to return specific tokens for display
        mock_tokenizer.encode.side_effect = lambda text: [1, 2, 3] if "2+2=" in text else [4, 5]
        mock_tokenizer.decode.side_effect = lambda ids: {
            (1,): "2",
            (2,): "+",
            (3,): "2",
            (4,): "4",
            (5,): "",
        }.get(tuple(ids), "")

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "Tokens:" in captured.out

    def test_generate_with_temperature(self, generate_args, mock_model, mock_tokenizer):
        """Test generation with non-zero temperature."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.temperature = 0.7
        generate_args.prompts = "2+2="

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            introspect_generate(generate_args)

            # Check that generate was called with temp parameter
            assert mocks["mlx_lm"].generate.call_count == 1
            call_kwargs = mocks["mlx_lm"].generate.call_args[1]
            assert call_kwargs.get("temp") == 0.7

    def test_generate_save_output(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test saving output to file."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            generate_args.output = temp_path
            generate_args.prompts = "2+2="

            with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
                mocks["extract_answer"].return_value = "4"

                introspect_generate(generate_args)

                captured = capsys.readouterr()
                assert f"Results saved to: {temp_path}" in captured.out

                # Check file was created with valid JSON
                with open(temp_path) as f:
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == 1
                    assert data[0]["prompt"] == "2+2="
                    assert data[0]["output"] == "4"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_answer_found_first(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test when answer is found in first token."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.prompts = "2+2="

        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.side_effect = lambda ids: "4"

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "onset=0 (answer-first)" in captured.out

    def test_generate_answer_delayed(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test when answer appears later in output."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.prompts = "2+2="

        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer.decode.side_effect = lambda ids: {
            (1,): "The",
            (2,): " answer",
            (3,): " is",
            (4,): " 4",
        }.get(tuple(ids), "")

        with self.mock_mlx_and_introspection(
            mock_model, mock_tokenizer, "The answer is 4"
        ) as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "(delayed)" in captured.out

    def test_generate_answer_not_found(self, generate_args, mock_model, mock_tokenizer, capsys):
        """Test when expected answer is not in output."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.prompts = "2+2="

        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.side_effect = lambda ids: "incorrect"

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer, "incorrect") as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "NOT FOUND in output" in captured.out

    def test_generate_show_tokens_with_onset(
        self, generate_args, mock_model, mock_tokenizer, capsys
    ):
        """Test showing tokens with highlighted onset token."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.show_tokens = True
        generate_args.prompts = "2+2="

        # Mock tokenizer for token-by-token breakdown
        def encode_side_effect(text):
            if "2+2=" in text and "answer is 4" in text:
                return [1, 2, 3, 4, 5, 6]  # prompt + output
            elif "2+2=" in text:
                return [1, 2, 3]  # just prompt
            else:
                return [4, 5, 6]  # just output

        mock_tokenizer.encode.side_effect = encode_side_effect
        mock_tokenizer.decode.side_effect = lambda ids: {
            (4,): "answer",
            (5,): " is",
            (6,): " 4",
        }.get(tuple(ids), "")

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer, "answer is 4") as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            # Should highlight the token with answer
            assert "[" in captured.out and "]" in captured.out

    def test_generate_show_tokens_long_output(
        self, generate_args, mock_model, mock_tokenizer, capsys
    ):
        """Test showing tokens with output longer than 10 tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.show_tokens = True
        generate_args.prompts = "2+2="

        # Return 15 tokens for the output
        def encode_side_effect(text):
            if "2+2=" in text and "a" in text:
                return list(range(1, 18))  # 17 tokens total
            elif "2+2=" in text:
                return [1, 2]  # 2 for prompt
            else:
                return list(range(3, 18))  # 15 for output

        mock_tokenizer.encode.side_effect = encode_side_effect
        mock_tokenizer.decode.side_effect = lambda ids: "a"

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer, "a" * 20):
            introspect_generate(generate_args)

            captured = capsys.readouterr()
            # Should show ellipsis for truncated tokens
            assert "..." in captured.out

    def test_generate_compare_format_diagnoses(
        self, generate_args, mock_model, mock_tokenizer, capsys
    ):
        """Test all diagnosis types in format comparison."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.compare_format = True
        generate_args.prompts = "2+2=|3+3=|4+4=|5+5="

        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            """Return different outputs based on call count to test different diagnoses."""
            result = {
                0: "4",  # no-space, answer found at 0
                1: "4",  # with-space, answer found at 0 -> SPACE-LOCK ONLY
                2: "wrong",  # no-space, answer not found
                3: "6",  # with-space, answer found -> COMPUTE BLOCKED
                4: "8",  # no-space, answer found at 0
                5: "wrong",  # with-space, answer not found -> WEIRD
                6: "The answer is definitely 10",  # no-space, delayed (onset=4)
                7: "10",  # with-space, answer at 0 -> ONSET ROUTING
            }[call_count[0]]
            call_count[0] += 1
            return result

        def encode_for_output(text):
            # Return appropriate tokens for each output
            if "answer is definitely" in text:
                return [1, 2, 3, 4, 5]
            return [1]

        def decode_for_output(ids):
            return {
                (1,): "The",
                (2,): " answer",
                (3,): " is",
                (4,): " definitely",
                (5,): " 10",
            }.get(tuple(ids), "" if len(ids) <= 1 else "text")

        mock_tokenizer.encode.side_effect = encode_for_output
        mock_tokenizer.decode.side_effect = decode_for_output

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["mlx_lm"].generate.side_effect = generate_side_effect
            mocks["extract_answer"].side_effect = lambda p: {
                "2+2=": "4",
                "3+3=": "6",
                "4+4=": "8",
                "5+5=": "10",
            }.get(p.strip(), None)

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "SPACE-LOCK ONLY" in captured.out
            assert "COMPUTE BLOCKED" in captured.out
            assert "WEIRD" in captured.out
            assert "ONSET ROUTING" in captured.out

    def test_generate_compare_format_both_fail(
        self, generate_args, mock_model, mock_tokenizer, capsys
    ):
        """Test diagnosis when both no-space and with-space fail."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.compare_format = True
        generate_args.prompts = "2+2="

        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.side_effect = lambda ids: "wrong"

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer, "wrong") as mocks:
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "BOTH FAIL" in captured.out

    def test_generate_compare_format_minor_difference(
        self, generate_args, mock_model, mock_tokenizer, capsys
    ):
        """Test diagnosis for minor onset difference."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.compare_format = True
        generate_args.prompts = "2+2="

        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            """Return outputs with small onset difference."""
            result = ["is 4", "4"][call_count[0]]
            call_count[0] += 1
            return result

        def encode_for_output(text):
            if "is 4" in text:
                return [1, 2]
            return [1]

        def decode_for_output(ids):
            return {
                (1,): "is",
                (2,): " 4",
            }.get(tuple(ids), "4")

        mock_tokenizer.encode.side_effect = encode_for_output
        mock_tokenizer.decode.side_effect = decode_for_output

        with self.mock_mlx_and_introspection(mock_model, mock_tokenizer) as mocks:
            mocks["mlx_lm"].generate.side_effect = generate_side_effect
            mocks["extract_answer"].return_value = "4"

            introspect_generate(generate_args)

            captured = capsys.readouterr()
            assert "MINOR DIFFERENCE" in captured.out


class TestFindAnswerOnset:
    """Tests for _find_answer_onset helper function."""

    def test_find_onset_no_expected(self):
        """Test with no expected answer."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        result = _find_answer_onset("some output", None, tokenizer)

        assert result.answer_found is False
        assert result.onset_index is None

    def test_find_onset_answer_found(self):
        """Test when answer is found."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.decode.side_effect = lambda ids: "4" if ids == [1] else "2"

        result = _find_answer_onset("42", "4", tokenizer)

        assert result.answer_found is True
        assert result.onset_index == 0

    def test_find_onset_answer_not_found(self):
        """Test when answer is not in output."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids: "x"

        result = _find_answer_onset("xxx", "42", tokenizer)

        assert result.answer_found is False

    def test_find_onset_is_answer_first(self):
        """Test is_answer_first flag when answer is in first 2 tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import _find_answer_onset

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1]
        tokenizer.decode.side_effect = lambda ids: "42"

        result = _find_answer_onset("42", "42", tokenizer)

        assert result.answer_found is True
        assert result.is_answer_first is True
        assert result.onset_index == 0

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

        assert result.answer_found is True
        assert result.is_answer_first is False
        assert result.onset_index == 2


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
