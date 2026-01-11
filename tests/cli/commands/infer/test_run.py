"""Tests for inference run command."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.infer._types import InferenceConfig
from chuk_lazarus.cli.commands.infer.run import run_inference_cmd


class TestRunInferenceCmd:
    """Tests for run_inference_cmd handler."""

    @pytest.fixture
    def basic_args(self):
        """Create basic inference arguments."""
        return Namespace(
            model="test-model",
            adapter=None,
            prompt="What is 2+2?",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )

    def test_run_inference_cmd_basic(self, basic_args, capsys):
        """Test basic inference command execution."""
        mock_result = MagicMock()
        mock_result.to_display.return_value = "The answer is 4."

        mock_service = MagicMock()
        mock_service.run = AsyncMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {"chuk_lazarus.inference": MagicMock(InferenceService=mock_service)},
        ):
            asyncio.run(run_inference_cmd(basic_args))

            # Verify service was called
            mock_service.run.assert_called_once()

            # Verify the config was passed correctly
            call_args = mock_service.run.call_args[0]
            config = call_args[0]
            assert config.model == "test-model"
            assert config.prompt == "What is 2+2?"

        captured = capsys.readouterr()
        assert "The answer is 4." in captured.out

    def test_run_inference_cmd_with_adapter(self, capsys):
        """Test inference with adapter path."""
        args = Namespace(
            model="test-model",
            adapter="/path/to/adapter",
            prompt="Test prompt",
            prompt_file=None,
            max_tokens=128,
            temperature=0.5,
        )

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Response with adapter"

        mock_service = MagicMock()
        mock_service.run = AsyncMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {"chuk_lazarus.inference": MagicMock(InferenceService=mock_service)},
        ):
            asyncio.run(run_inference_cmd(args))

            call_args = mock_service.run.call_args[0]
            config = call_args[0]
            assert config.adapter == "/path/to/adapter"

        captured = capsys.readouterr()
        assert "Response with adapter" in captured.out

    def test_run_inference_cmd_multiline_output(self, basic_args, capsys):
        """Test inference with multiline output."""
        mock_result = MagicMock()
        mock_result.to_display.return_value = "Line 1\nLine 2\nLine 3"

        mock_service = MagicMock()
        mock_service.run = AsyncMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {"chuk_lazarus.inference": MagicMock(InferenceService=mock_service)},
        ):
            asyncio.run(run_inference_cmd(basic_args))

        captured = capsys.readouterr()
        assert "Line 1" in captured.out
        assert "Line 2" in captured.out
        assert "Line 3" in captured.out


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            model="test-model",
            adapter=None,
            prompt="Test prompt",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        assert config.model == "test-model"
        assert config.prompt == "Test prompt"
        assert config.max_tokens == 256
        assert config.temperature == 0.7

    def test_from_args_with_adapter(self):
        """Test creating config with adapter."""
        args = Namespace(
            model="test-model",
            adapter="/path/to/adapter",
            prompt="Test",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        assert config.adapter == "/path/to/adapter"

    def test_input_mode_single(self):
        """Test input mode detection for single prompt."""
        args = Namespace(
            model="test-model",
            adapter=None,
            prompt="Test",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        from chuk_lazarus.cli.commands._constants import InputMode

        assert config.input_mode == InputMode.SINGLE

    def test_input_mode_file(self, tmp_path):
        """Test input mode detection for file input."""
        # Create temp file
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Test prompt")

        args = Namespace(
            model="test-model",
            adapter=None,
            prompt=None,
            prompt_file=prompts_file,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        from chuk_lazarus.cli.commands._constants import InputMode

        assert config.input_mode == InputMode.FILE

    def test_input_mode_interactive(self):
        """Test input mode detection for interactive mode."""
        args = Namespace(
            model="test-model",
            adapter=None,
            prompt=None,
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        from chuk_lazarus.cli.commands._constants import InputMode

        assert config.input_mode == InputMode.INTERACTIVE

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from pydantic import ValidationError

        args = Namespace(
            model="test-model",
            adapter=None,
            prompt="Test",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )
        config = InferenceConfig.from_args(args)

        with pytest.raises(ValidationError):
            config.model = "other-model"
