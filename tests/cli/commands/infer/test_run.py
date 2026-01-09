"""Tests for inference run command."""

from argparse import Namespace

import pytest

from chuk_lazarus.cli.commands.infer._types import InferenceConfig


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
