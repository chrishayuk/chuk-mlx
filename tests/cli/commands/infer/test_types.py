"""Tests for infer CLI type definitions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.cli.commands.infer._types import (
    GenerationResult,
    InferenceConfig,
    InferenceResult,
    InputMode,
)


class TestInputMode:
    """Tests for InputMode enum."""

    def test_input_mode_values(self):
        """Test InputMode enum values."""
        assert InputMode.SINGLE == "single"
        assert InputMode.FILE == "file"
        assert InputMode.INTERACTIVE == "interactive"

    def test_input_mode_is_string_enum(self):
        """Test InputMode is a string enum."""
        assert isinstance(InputMode.SINGLE, str)
        assert InputMode.SINGLE.value == "single"


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_from_args_basic(self, basic_infer_args):
        """Test creating config from args."""
        config = InferenceConfig.from_args(basic_infer_args)

        assert config.model == "test-model"
        assert config.adapter is None
        assert config.prompt == "What is 2+2?"
        assert config.prompt_file is None
        assert config.max_tokens == 256
        assert config.temperature == 0.7

    def test_from_args_with_adapter(self, basic_infer_args):
        """Test config with adapter."""
        basic_infer_args.adapter = "/path/to/adapter"
        config = InferenceConfig.from_args(basic_infer_args)

        assert config.adapter == "/path/to/adapter"

    def test_from_args_with_prompt_file(self, basic_infer_args):
        """Test config with prompt file."""
        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = "/path/to/prompts.txt"
        config = InferenceConfig.from_args(basic_infer_args)

        assert config.prompt is None
        assert config.prompt_file == Path("/path/to/prompts.txt")

    def test_input_mode_single(self, basic_infer_args):
        """Test input mode detection for single prompt."""
        config = InferenceConfig.from_args(basic_infer_args)
        assert config.input_mode == InputMode.SINGLE

    def test_input_mode_file(self, basic_infer_args):
        """Test input mode detection for file."""
        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = "/path/to/prompts.txt"
        config = InferenceConfig.from_args(basic_infer_args)
        assert config.input_mode == InputMode.FILE

    def test_input_mode_interactive(self, basic_infer_args):
        """Test input mode detection for interactive."""
        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = None
        config = InferenceConfig.from_args(basic_infer_args)
        assert config.input_mode == InputMode.INTERACTIVE

    def test_max_tokens_validation(self, basic_infer_args):
        """Test max_tokens validation."""
        basic_infer_args.max_tokens = 0
        with pytest.raises(ValidationError):
            InferenceConfig.from_args(basic_infer_args)

    def test_max_tokens_upper_bound(self, basic_infer_args):
        """Test max_tokens upper bound."""
        basic_infer_args.max_tokens = 10000
        with pytest.raises(ValidationError):
            InferenceConfig.from_args(basic_infer_args)

    def test_temperature_validation(self, basic_infer_args):
        """Test temperature validation."""
        basic_infer_args.temperature = -0.1
        with pytest.raises(ValidationError):
            InferenceConfig.from_args(basic_infer_args)

    def test_temperature_upper_bound(self, basic_infer_args):
        """Test temperature upper bound."""
        basic_infer_args.temperature = 2.5
        with pytest.raises(ValidationError):
            InferenceConfig.from_args(basic_infer_args)

    def test_config_is_frozen(self, basic_infer_args):
        """Test config is immutable."""
        config = InferenceConfig.from_args(basic_infer_args)
        with pytest.raises(ValidationError):
            config.model = "new-model"

    def test_config_forbids_extra_fields(self):
        """Test config forbids extra fields."""
        with pytest.raises(ValidationError):
            InferenceConfig(
                model="test",
                extra_field="value",
            )


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = GenerationResult(
            prompt="Hello",
            response="Hi there!",
            tokens_generated=5,
        )
        assert result.prompt == "Hello"
        assert result.response == "Hi there!"
        assert result.tokens_generated == 5

    def test_to_display(self):
        """Test display formatting."""
        result = GenerationResult(
            prompt="What is 2+2?",
            response="4",
            tokens_generated=1,
        )
        display = result.to_display()
        assert "Prompt: What is 2+2?" in display
        assert "Response: 4" in display

    def test_default_tokens_generated(self):
        """Test default tokens_generated."""
        result = GenerationResult(
            prompt="Hello",
            response="Hi",
        )
        assert result.tokens_generated == 0

    def test_tokens_generated_validation(self):
        """Test tokens_generated must be non-negative."""
        with pytest.raises(ValidationError):
            GenerationResult(
                prompt="Hello",
                response="Hi",
                tokens_generated=-1,
            )


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        gen = GenerationResult(
            prompt="Hello",
            response="Hi",
            tokens_generated=1,
        )
        result = InferenceResult(
            generations=[gen],
            model="test-model",
        )
        assert len(result.generations) == 1
        assert result.model == "test-model"
        assert result.adapter is None

    def test_with_adapter(self):
        """Test result with adapter."""
        result = InferenceResult(
            generations=[],
            model="test-model",
            adapter="/path/to/adapter",
        )
        assert result.adapter == "/path/to/adapter"

    def test_to_display(self):
        """Test display formatting."""
        gen = GenerationResult(
            prompt="What is 2+2?",
            response="4",
            tokens_generated=1,
        )
        result = InferenceResult(
            generations=[gen],
            model="test-model",
        )
        display = result.to_display()
        assert "Inference Results" in display
        assert "Model" in display
        assert "test-model" in display
        assert "Generations" in display
        assert "What is 2+2?" in display

    def test_to_display_with_adapter(self):
        """Test display with adapter shows adapter info."""
        result = InferenceResult(
            generations=[],
            model="test-model",
            adapter="/path/to/adapter",
        )
        display = result.to_display()
        assert "Adapter" in display
        assert "/path/to/adapter" in display

    def test_empty_generations(self):
        """Test result with empty generations."""
        result = InferenceResult(
            generations=[],
            model="test-model",
        )
        assert len(result.generations) == 0
        display = result.to_display()
        assert "Generations" in display

    def test_multiple_generations(self):
        """Test result with multiple generations."""
        gens = [GenerationResult(prompt=f"Prompt {i}", response=f"Response {i}") for i in range(3)]
        result = InferenceResult(
            generations=gens,
            model="test-model",
        )
        display = result.to_display()
        assert "Prompt 0" in display
        assert "Prompt 1" in display
        assert "Prompt 2" in display
