"""Tests for inference run command."""

import logging
from unittest.mock import patch

import pytest

from chuk_lazarus.cli.commands.infer._types import InferenceConfig
from chuk_lazarus.cli.commands.infer.run import run_inference, run_inference_cmd

# The patch target for load_model since models module may not exist
LOAD_MODEL_PATCH = "chuk_lazarus.models.load_model"


class TestRunInference:
    """Tests for run_inference async command."""

    @pytest.fixture
    def basic_config(self, basic_infer_args):
        """Create basic inference config."""
        return InferenceConfig.from_args(basic_infer_args)

    @pytest.mark.asyncio
    async def test_run_inference_with_prompt(self, basic_config, mock_model):
        """Test inference with a single prompt."""
        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "4"

            result = await run_inference(basic_config)

            # Verify model was loaded
            mock_load.assert_called_once_with("test-model")

            # Verify generate was called
            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=256,
                temperature=0.7,
            )

            # Check result
            assert len(result.generations) == 1
            assert result.generations[0].prompt == "What is 2+2?"
            assert result.generations[0].response == "4"
            assert result.model == "test-model"
            assert result.adapter is None

    @pytest.mark.asyncio
    async def test_run_inference_with_adapter(self, basic_infer_args, mock_model):
        """Test inference with a LoRA adapter."""
        basic_infer_args.adapter = "/path/to/adapter"
        config = InferenceConfig.from_args(basic_infer_args)

        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            result = await run_inference(config)

            # Verify adapter was loaded
            mock_model.load_adapter.assert_called_once_with("/path/to/adapter")
            assert result.adapter == "/path/to/adapter"

    @pytest.mark.asyncio
    async def test_run_inference_with_prompt_file(self, basic_infer_args, mock_model, tmp_path):
        """Test inference with prompts from a file."""
        # Create a temporary prompts file
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("What is 2+2?\nWhat is 3+3?\n\nWhat is 4+4?\n")

        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = str(prompts_file)
        config = InferenceConfig.from_args(basic_infer_args)

        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["4", "6", "8"]

            result = await run_inference(config)

            # Should process 3 prompts (skipping empty lines)
            assert len(result.generations) == 3
            assert result.generations[0].response == "4"
            assert result.generations[1].response == "6"
            assert result.generations[2].response == "8"

    @pytest.mark.asyncio
    async def test_run_inference_interactive_mode(self, basic_infer_args, mock_model):
        """Test interactive mode (no prompt or prompt_file)."""
        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = None
        config = InferenceConfig.from_args(basic_infer_args)

        # Simulate user input: two prompts then Ctrl+D
        user_inputs = ["Hello, world!", "How are you?"]

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch("builtins.input", side_effect=user_inputs + [EOFError()]),
            patch("builtins.print"),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["Hi!", "I'm good!"]

            result = await run_inference(config)

            # Should process 2 prompts
            assert len(result.generations) == 2
            assert result.generations[0].prompt == "Hello, world!"
            assert result.generations[0].response == "Hi!"

    @pytest.mark.asyncio
    async def test_run_inference_interactive_with_empty_prompts(self, basic_infer_args, mock_model):
        """Test interactive mode with some empty inputs."""
        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = None
        config = InferenceConfig.from_args(basic_infer_args)

        # Empty, valid, empty, valid, then Ctrl+D
        user_inputs = ["", "Hello", "", "World", EOFError()]

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch("builtins.input", side_effect=user_inputs),
            patch("builtins.print"),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["Response1", "Response2"]

            result = await run_inference(config)

            # Should only process non-empty prompts
            assert len(result.generations) == 2

    @pytest.mark.asyncio
    async def test_run_inference_logging(self, basic_config, mock_model, caplog):
        """Test that logging messages are generated."""
        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            caplog.at_level(logging.INFO),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            await run_inference(basic_config)

            assert "Loading model: test-model" in caplog.text

    @pytest.mark.asyncio
    async def test_run_inference_with_adapter_logging(self, basic_infer_args, mock_model, caplog):
        """Test logging when adapter is loaded."""
        basic_infer_args.adapter = "/path/to/adapter"
        config = InferenceConfig.from_args(basic_infer_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            caplog.at_level(logging.INFO),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            await run_inference(config)

            assert "Loading adapter: /path/to/adapter" in caplog.text

    @pytest.mark.asyncio
    async def test_run_inference_with_zero_temperature(self, basic_infer_args, mock_model):
        """Test inference with temperature=0 for deterministic output."""
        basic_infer_args.temperature = 0.0
        config = InferenceConfig.from_args(basic_infer_args)

        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            await run_inference(config)

            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=256,
                temperature=0.0,
            )

    @pytest.mark.asyncio
    async def test_run_inference_with_custom_max_tokens(self, basic_infer_args, mock_model):
        """Test inference with custom max_tokens."""
        basic_infer_args.max_tokens = 100
        config = InferenceConfig.from_args(basic_infer_args)

        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            await run_inference(config)

            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=100,
                temperature=0.7,
            )

    @pytest.mark.asyncio
    async def test_run_inference_tokens_count(self, basic_config, mock_model):
        """Test that tokens_generated is approximated."""
        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "This is a five word response"

            result = await run_inference(basic_config)

            # Tokens count is approximated by word count
            assert result.generations[0].tokens_generated == 6


class TestRunInferenceCmd:
    """Tests for run_inference_cmd CLI entry point."""

    @pytest.mark.asyncio
    async def test_run_inference_cmd(self, basic_infer_args, mock_model, capsys):
        """Test CLI entry point."""
        with patch(LOAD_MODEL_PATCH, create=True) as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "4"

            await run_inference_cmd(basic_infer_args)

            captured = capsys.readouterr()
            assert "Inference Results" in captured.out
            assert "What is 2+2?" in captured.out
            assert "4" in captured.out
