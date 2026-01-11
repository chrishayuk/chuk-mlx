"""Tests for introspect CLI utility functions."""

import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect._utils import (
    apply_chat_template,
    get_embed_tokens,
    get_final_norm,
    get_lm_head,
    get_model_layers,
    load_external_chat_template,
    normalize_number,
    parse_layers,
    parse_prompts,
    parse_value_list,
    print_analysis_result,
    validate_prompt_args,
)


class TestParseLayers:
    """Tests for parse_layers function."""

    def test_parse_layers_none(self):
        """Test parsing None returns None."""
        assert parse_layers(None) is None

    def test_parse_layers_empty_string(self):
        """Test parsing empty string returns None."""
        assert parse_layers("") is None

    def test_parse_layers_single(self):
        """Test parsing single layer."""
        assert parse_layers("5") == [5]

    def test_parse_layers_multiple(self):
        """Test parsing multiple comma-separated layers."""
        assert parse_layers("1,2,3") == [1, 2, 3]

    def test_parse_layers_with_spaces(self):
        """Test parsing layers with spaces."""
        assert parse_layers("1, 2, 3") == [1, 2, 3]

    def test_parse_layers_range(self):
        """Test parsing layer range."""
        assert parse_layers("5-8") == [5, 6, 7, 8]

    def test_parse_layers_mixed(self):
        """Test parsing mixed individual and ranges."""
        assert parse_layers("1,5-7,10") == [1, 5, 6, 7, 10]

    def test_parse_layers_multiple_ranges(self):
        """Test parsing multiple ranges."""
        assert parse_layers("1-3,8-10") == [1, 2, 3, 8, 9, 10]


class TestParsePrompts:
    """Tests for parse_prompts function."""

    def test_parse_prompts_single(self):
        """Test parsing single prompt."""
        assert parse_prompts("hello world") == ["hello world"]

    def test_parse_prompts_multiple(self):
        """Test parsing pipe-separated prompts."""
        assert parse_prompts("a|b|c") == ["a", "b", "c"]

    def test_parse_prompts_with_spaces(self):
        """Test parsing prompts with leading/trailing spaces."""
        assert parse_prompts("  a  |  b  ") == ["a", "b"]

    def test_parse_prompts_from_file(self):
        """Test parsing prompts from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt 1\n")
            f.write("prompt 2\n")
            f.write("\n")  # empty line should be skipped
            f.write("prompt 3\n")
            f.flush()

            result = parse_prompts(f"@{f.name}")
            assert result == ["prompt 1", "prompt 2", "prompt 3"]

    def test_parse_prompts_file_with_whitespace(self):
        """Test parsing prompts from file with whitespace lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  prompt 1  \n")
            f.write("   \n")  # whitespace-only line should be skipped
            f.write("prompt 2\n")
            f.flush()

            result = parse_prompts(f"@{f.name}")
            assert result == ["prompt 1", "prompt 2"]


class TestParseValueList:
    """Tests for parse_value_list function."""

    def test_parse_value_list_pipe_separated(self):
        """Test parsing pipe-separated values."""
        result = parse_value_list("a|b|c")
        assert result == ["a", "b", "c"]

    def test_parse_value_list_custom_delimiter(self):
        """Test parsing with custom delimiter."""
        result = parse_value_list("1,2,3", delimiter=",")
        assert result == ["1", "2", "3"]

    def test_parse_value_list_int_type(self):
        """Test parsing as integers."""
        result = parse_value_list("1|2|3", value_type=int)
        assert result == [1, 2, 3]

    def test_parse_value_list_float_type(self):
        """Test parsing as floats."""
        result = parse_value_list("1.5|2.5", value_type=float)
        assert result == [1.5, 2.5]

    def test_parse_value_list_from_file(self):
        """Test parsing values from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("value1\n")
            f.write("value2\n")
            f.write("\n")  # empty line should be skipped
            f.write("value3\n")
            f.flush()

            result = parse_value_list(f"@{f.name}")
            assert result == ["value1", "value2", "value3"]

    def test_parse_value_list_from_file_int_type(self):
        """Test parsing integer values from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("10\n")
            f.write("20\n")
            f.write("30\n")
            f.flush()

            result = parse_value_list(f"@{f.name}", value_type=int)
            assert result == [10, 20, 30]


class TestNormalizeNumber:
    """Tests for normalize_number function."""

    def test_normalize_plain_number(self):
        """Test normalizing plain number."""
        assert normalize_number("12345") == "12345"

    def test_normalize_with_commas(self):
        """Test normalizing number with commas."""
        assert normalize_number("1,234,567") == "1234567"

    def test_normalize_with_spaces(self):
        """Test normalizing number with spaces."""
        assert normalize_number("1 234 567") == "1234567"

    def test_normalize_with_thin_spaces(self):
        """Test normalizing number with thin spaces (unicode)."""
        assert normalize_number("1\u202f234") == "1234"

    def test_normalize_with_non_breaking_spaces(self):
        """Test normalizing number with non-breaking spaces."""
        assert normalize_number("1\u00a0234") == "1234"

    def test_normalize_mixed(self):
        """Test normalizing with mixed separators."""
        assert normalize_number("1,234 567") == "1234567"


class TestApplyChatTemplate:
    """Tests for apply_chat_template function."""

    def test_apply_chat_template_no_template(self):
        """Test when tokenizer has no chat template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        result = apply_chat_template(tokenizer, "hello")
        assert result == "hello"

    def test_apply_chat_template_no_method(self):
        """Test when tokenizer has no apply_chat_template method."""
        tokenizer = MagicMock(spec=[])

        result = apply_chat_template(tokenizer, "hello")
        assert result == "hello"

    def test_apply_chat_template_success(self):
        """Test successful chat template application."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "some template"
        tokenizer.apply_chat_template.return_value = "<|user|>hello<|assistant|>"

        result = apply_chat_template(tokenizer, "hello")

        tokenizer.apply_chat_template.assert_called_once()
        call_args = tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == [{"role": "user", "content": "hello"}]
        assert call_args[1]["tokenize"] is False
        assert call_args[1]["add_generation_prompt"] is True
        assert result == "<|user|>hello<|assistant|>"

    def test_apply_chat_template_exception(self):
        """Test chat template application with exception."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "some template"
        tokenizer.apply_chat_template.side_effect = Exception("template error")

        result = apply_chat_template(tokenizer, "hello")
        assert result == "hello"


class TestLoadExternalChatTemplate:
    """Tests for load_external_chat_template function."""

    def test_load_external_template_local_path(self):
        """Test loading template from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "chat_template.jinja"
            template_path.write_text("{{ content }}")

            tokenizer = MagicMock()
            tokenizer.chat_template = None

            with patch("huggingface_hub.snapshot_download") as mock_dl:
                mock_dl.side_effect = Exception("not found")
                load_external_chat_template(tokenizer, tmpdir)

            assert tokenizer.chat_template == "{{ content }}"

    def test_load_external_template_already_has_template(self):
        """Test that existing template is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "chat_template.jinja"
            template_path.write_text("new template")

            tokenizer = MagicMock()
            tokenizer.chat_template = "existing template"

            with patch("huggingface_hub.snapshot_download") as mock_dl:
                mock_dl.side_effect = Exception("not found")
                load_external_chat_template(tokenizer, tmpdir)

            assert tokenizer.chat_template == "existing template"

    def test_load_external_template_no_file(self):
        """Test when no template file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = MagicMock()
            tokenizer.chat_template = None

            with patch("huggingface_hub.snapshot_download") as mock_dl:
                mock_dl.side_effect = Exception("not found")
                load_external_chat_template(tokenizer, tmpdir)

            assert tokenizer.chat_template is None


class TestValidatePromptArgs:
    """Tests for validate_prompt_args function."""

    def test_validate_with_prompt(self):
        """Test validation passes with prompt."""
        args = Namespace(prompt="hello", prefix=None, criterion=None)
        # Should not raise
        validate_prompt_args(args, require_criterion=False)

    def test_validate_with_prefix(self):
        """Test validation passes with prefix."""
        args = Namespace(prompt=None, prefix="hello", criterion=None)
        # Should not raise
        validate_prompt_args(args, require_criterion=False)

    def test_validate_missing_prompt(self):
        """Test validation fails without prompt or prefix."""
        args = Namespace(prompt=None, prefix=None, criterion=None)
        with pytest.raises(SystemExit):
            validate_prompt_args(args)

    def test_validate_require_criterion_missing(self):
        """Test validation fails when criterion required but missing."""
        args = Namespace(prompt="hello", prefix=None, criterion=None)
        with pytest.raises(SystemExit):
            validate_prompt_args(args, require_criterion=True)

    def test_validate_require_criterion_present(self):
        """Test validation passes when criterion required and present."""
        args = Namespace(prompt="hello", prefix=None, criterion="some criterion")
        # Should not raise
        validate_prompt_args(args, require_criterion=True)


class TestGetModelLayers:
    """Tests for get_model_layers function."""

    def test_get_layers_nested(self):
        """Test getting layers from nested model structure."""
        layers = [MagicMock(), MagicMock()]
        model = MagicMock()
        model.model.layers = layers

        result = get_model_layers(model)
        assert result == layers

    def test_get_layers_flat(self):
        """Test getting layers from flat model structure."""
        layers = [MagicMock(), MagicMock()]
        model = MagicMock(spec=["layers"])
        model.layers = layers

        result = get_model_layers(model)
        assert result == layers

    def test_get_layers_not_found(self):
        """Test when layers not found."""
        model = MagicMock(spec=[])

        result = get_model_layers(model)
        assert result is None


class TestGetEmbedTokens:
    """Tests for get_embed_tokens function."""

    def test_get_embed_tokens_nested(self):
        """Test getting embed_tokens from nested structure."""
        embed = MagicMock()
        model = MagicMock()
        model.model.embed_tokens = embed

        result = get_embed_tokens(model)
        assert result == embed

    def test_get_embed_tokens_flat(self):
        """Test getting embed_tokens from flat structure."""
        embed = MagicMock()
        model = MagicMock(spec=["embed_tokens"])
        model.embed_tokens = embed

        result = get_embed_tokens(model)
        assert result == embed

    def test_get_embed_tokens_not_found(self):
        """Test when embed_tokens not found."""
        model = MagicMock(spec=[])

        result = get_embed_tokens(model)
        assert result is None


class TestGetLmHead:
    """Tests for get_lm_head function."""

    def test_get_lm_head_exists(self):
        """Test getting lm_head when it exists."""
        head = MagicMock()
        model = MagicMock()
        model.lm_head = head

        result = get_lm_head(model)
        assert result == head

    def test_get_lm_head_not_found(self):
        """Test when lm_head not found."""
        model = MagicMock(spec=[])

        result = get_lm_head(model)
        assert result is None


class TestGetFinalNorm:
    """Tests for get_final_norm function."""

    def test_get_final_norm_nested(self):
        """Test getting norm from nested structure."""
        norm = MagicMock()
        model = MagicMock()
        model.model.norm = norm

        result = get_final_norm(model)
        assert result == norm

    def test_get_final_norm_flat(self):
        """Test getting norm from flat structure."""
        norm = MagicMock()
        model = MagicMock(spec=["norm"])
        model.norm = norm

        result = get_final_norm(model)
        assert result == norm

    def test_get_final_norm_not_found(self):
        """Test when norm not found."""
        model = MagicMock(spec=[])

        result = get_final_norm(model)
        assert result is None


class TestPrintAnalysisResult:
    """Tests for print_analysis_result function."""

    def test_print_analysis_result_basic(self, capsys):
        """Test basic analysis result printing."""
        # Create mock result
        result = MagicMock()
        result.tokens = ["hello", "world"]
        result.captured_layers = [0, 4, 8]

        pred1 = MagicMock()
        pred1.probability = 0.8
        pred1.token = "test"

        result.final_prediction = [pred1]

        layer_pred = MagicMock()
        layer_pred.layer_idx = 0
        layer_pred.predictions = [pred1]
        result.layer_predictions = [layer_pred]
        result.token_evolutions = []

        tokenizer = MagicMock()
        args = Namespace(top_k=5)

        print_analysis_result(result, tokenizer, args)

        captured = capsys.readouterr()
        assert "Tokens (2)" in captured.out
        assert "hello" in captured.out
        assert "world" in captured.out
        assert "Final Prediction" in captured.out
        assert "0.8" in captured.out

    def test_print_analysis_result_many_tokens(self, capsys):
        """Test printing with many tokens (truncated)."""
        result = MagicMock()
        result.tokens = [f"tok{i}" for i in range(15)]
        result.captured_layers = [0]
        result.final_prediction = []
        result.layer_predictions = []
        result.token_evolutions = []

        tokenizer = MagicMock()
        args = Namespace(top_k=5)

        print_analysis_result(result, tokenizer, args)

        captured = capsys.readouterr()
        assert "Tokens (15)" in captured.out
        # First 5 and last 3 should be shown
        assert "tok0" in captured.out
        assert "tok14" in captured.out
        assert "..." in captured.out

    def test_print_analysis_result_with_evolution(self, capsys):
        """Test printing with token evolution."""
        result = MagicMock()
        result.tokens = ["test"]
        result.captured_layers = [0, 4]

        pred = MagicMock()
        pred.probability = 0.5
        pred.token = "next"
        result.final_prediction = [pred]

        layer_pred = MagicMock()
        layer_pred.layer_idx = 0
        layer_pred.predictions = [pred]
        result.layer_predictions = [layer_pred]

        # Add token evolution
        evolution = MagicMock()
        evolution.token = "evolving"
        evolution.layer_probabilities = {0: 0.1, 4: 0.9}
        evolution.layer_ranks = {0: 10, 4: 1}
        evolution.emergence_layer = 4
        result.token_evolutions = [evolution]

        tokenizer = MagicMock()
        args = Namespace(top_k=5)

        print_analysis_result(result, tokenizer, args)

        captured = capsys.readouterr()
        assert "Token Evolution" in captured.out
        assert "evolving" in captured.out
        assert "Becomes top-1 at layer 4" in captured.out
