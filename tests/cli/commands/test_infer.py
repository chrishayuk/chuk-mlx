"""Tests for inference command handlers."""

import sys
from argparse import Namespace
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestRunInference:
    """Tests for run_inference command."""

    @pytest.fixture(autouse=True)
    def setup_models_module(self):
        """Set up mock models module."""
        # Create a mock models module if it doesn't exist
        if "chuk_lazarus.models" not in sys.modules:
            mock_models_module = MagicMock()
            sys.modules["chuk_lazarus.models"] = mock_models_module
            yield
            # Clean up
            if "chuk_lazarus.models" in sys.modules:
                del sys.modules["chuk_lazarus.models"]
        else:
            yield

    @pytest.fixture
    def basic_infer_args(self):
        """Create basic inference arguments."""
        return Namespace(
            model="test-model",
            adapter=None,
            prompt="What is 2+2?",
            prompt_file=None,
            max_tokens=256,
            temperature=0.7,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.load_adapter = MagicMock()
        model.generate = MagicMock(return_value="4")
        return model

    def test_run_inference_with_prompt(self, basic_infer_args, mock_model, capsys):
        """Test inference with a single prompt."""
        from chuk_lazarus.cli.commands.infer import run_inference

        with patch("chuk_lazarus.models.load_model") as mock_load:
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Verify model was loaded
            mock_load.assert_called_once_with("test-model")

            # Verify generate was called with correct parameters
            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=256,
                temperature=0.7,
            )

            # Check output
            captured = capsys.readouterr()
            assert "Prompt: What is 2+2?" in captured.out
            assert "Response: 4" in captured.out

    def test_run_inference_with_adapter(self, basic_infer_args, mock_model, capsys):
        """Test inference with a LoRA adapter."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.adapter = "/path/to/adapter"

        with patch("chuk_lazarus.models.load_model") as mock_load:
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Verify adapter was loaded
            mock_model.load_adapter.assert_called_once_with("/path/to/adapter")

    def test_run_inference_with_prompt_file(self, basic_infer_args, mock_model, capsys):
        """Test inference with prompts from a file."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = "/tmp/prompts.txt"

        prompts_content = "What is 2+2?\nWhat is 3+3?\n\nWhat is 4+4?\n"

        with (
            patch("chuk_lazarus.models.load_model") as mock_load,
            patch("builtins.open", mock_open(read_data=prompts_content)),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["4", "6", "8"]

            run_inference(basic_infer_args)

            # Should process 3 prompts (skipping empty lines)
            assert mock_model.generate.call_count == 3

            # Verify prompts were processed correctly
            captured = capsys.readouterr()
            assert "What is 2+2?" in captured.out
            assert "What is 3+3?" in captured.out
            assert "What is 4+4?" in captured.out

    def test_run_inference_interactive_mode(self, basic_infer_args, mock_model, capsys):
        """Test interactive mode (no prompt or prompt_file)."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = None

        # Simulate user input: two prompts then Ctrl+D
        user_inputs = ["Hello, world!", "How are you?"]

        with (
            patch("chuk_lazarus.models.load_model") as mock_load,
            patch("builtins.input", side_effect=user_inputs + [EOFError()]),
            patch("builtins.print") as mock_print,
        ):
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["Hi!", "I'm good!"]

            run_inference(basic_infer_args)

            # Should have prompted for input
            mock_print.assert_any_call("Enter prompts (Ctrl+D to finish):")

            # Should process 2 prompts
            assert mock_model.generate.call_count == 2

    def test_run_inference_interactive_mode_with_empty_prompts(
        self, basic_infer_args, mock_model, capsys
    ):
        """Test interactive mode with some empty inputs."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.prompt = None
        basic_infer_args.prompt_file = None

        # Simulate user input: empty, valid, empty, valid, then Ctrl+D
        user_inputs = ["", "Hello", "", "World", EOFError()]

        with (
            patch("chuk_lazarus.models.load_model") as mock_load,
            patch("builtins.input", side_effect=user_inputs),
        ):
            mock_load.return_value = mock_model
            mock_model.generate.side_effect = ["Response1", "Response2"]

            run_inference(basic_infer_args)

            # Should only process non-empty prompts
            assert mock_model.generate.call_count == 2

    def test_run_inference_multiple_prompts(self, basic_infer_args, mock_model, capsys):
        """Test inference with multiple prompts passed as prompt argument."""
        from chuk_lazarus.cli.commands.infer import run_inference

        # Test that we handle a list of prompts properly
        basic_infer_args.prompt = "First prompt"

        with patch("chuk_lazarus.models.load_model") as mock_load:
            mock_load.return_value = mock_model
            mock_model.generate.return_value = "Response"

            run_inference(basic_infer_args)

            # Should process the single prompt
            mock_model.generate.assert_called_once()

            captured = capsys.readouterr()
            assert "Prompt: First prompt" in captured.out
            assert "Response: Response" in captured.out

    def test_run_inference_logging(self, basic_infer_args, mock_model, caplog):
        """Test that logging messages are generated."""
        import logging

        from chuk_lazarus.cli.commands.infer import run_inference

        with (
            patch("chuk_lazarus.models.load_model") as mock_load,
            caplog.at_level(logging.INFO),
        ):
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Check log messages
            assert "Loading model: test-model" in caplog.text

    def test_run_inference_with_adapter_logging(self, basic_infer_args, mock_model, caplog):
        """Test logging when adapter is loaded."""
        import logging

        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.adapter = "/path/to/adapter"

        with (
            patch("chuk_lazarus.models.load_model") as mock_load,
            caplog.at_level(logging.INFO),
        ):
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Check log messages
            assert "Loading adapter: /path/to/adapter" in caplog.text

    def test_run_inference_with_zero_temperature(self, basic_infer_args, mock_model):
        """Test inference with temperature=0 for deterministic output."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.temperature = 0.0

        with patch("chuk_lazarus.models.load_model") as mock_load:
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Verify temperature was passed correctly
            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=256,
                temperature=0.0,
            )

    def test_run_inference_with_custom_max_tokens(self, basic_infer_args, mock_model):
        """Test inference with custom max_tokens."""
        from chuk_lazarus.cli.commands.infer import run_inference

        basic_infer_args.max_tokens = 100

        with patch("chuk_lazarus.models.load_model") as mock_load:
            mock_load.return_value = mock_model

            run_inference(basic_infer_args)

            # Verify max_tokens was passed correctly
            mock_model.generate.assert_called_once_with(
                "What is 2+2?",
                max_tokens=100,
                temperature=0.7,
            )
