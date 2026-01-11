"""Tests for steering utils module."""

import json
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.introspection.steering.utils import (
    compare_steering_effects,
    format_functiongemma_prompt,
    steer_model,
)


class TestFormatFunctiongemmaPrompt:
    """Tests for format_functiongemma_prompt function."""

    def test_basic_prompt(self):
        """Test basic prompt formatting."""
        prompt = format_functiongemma_prompt("What is the weather?")

        assert "<start_of_turn>developer" in prompt
        assert "<end_of_turn>" in prompt
        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt
        assert "What is the weather?" in prompt

    def test_default_tools(self):
        """Test that default tools are included."""
        prompt = format_functiongemma_prompt("test query")

        assert "get_weather" in prompt
        assert "send_email" in prompt
        assert "set_timer" in prompt

    def test_custom_tools(self):
        """Test with custom tools."""
        custom_tools = [
            {
                "name": "custom_function",
                "description": "A custom function",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            }
        ]

        prompt = format_functiongemma_prompt("test query", tools=custom_tools)

        assert "custom_function" in prompt
        assert "get_weather" not in prompt  # Default tools should not be included

    def test_prompt_structure(self):
        """Test the overall structure of the prompt."""
        prompt = format_functiongemma_prompt("Hello")

        # Check order of sections
        developer_start = prompt.find("<start_of_turn>developer")
        user_start = prompt.find("<start_of_turn>user")
        model_start = prompt.find("<start_of_turn>model")

        assert developer_start < user_start < model_start

    def test_tools_are_valid_json(self):
        """Test that tools section contains valid JSON."""
        prompt = format_functiongemma_prompt("test")

        # Extract the tools JSON from between the developer markers
        start_marker = (
            "You are a model that can do function calling with the following functions:\n"
        )
        end_marker = "\n<end_of_turn>"

        start_idx = prompt.find(start_marker) + len(start_marker)
        end_idx = prompt.find(end_marker, start_idx)
        tools_json = prompt[start_idx:end_idx]

        # Should be parseable JSON
        tools = json.loads(tools_json)
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_empty_tools_list(self):
        """Test with empty tools list."""
        prompt = format_functiongemma_prompt("test", tools=[])

        # Should still have the structure but with empty array
        assert "[]" in prompt

    def test_complex_tools(self):
        """Test with complex tool definitions."""
        tools = [
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10,
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "date_from": {"type": "string"},
                                "date_to": {"type": "string"},
                            },
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

        prompt = format_functiongemma_prompt("search for python", tools=tools)

        assert "search" in prompt
        assert "query" in prompt
        assert "limit" in prompt
        assert "filters" in prompt

    def test_special_characters_in_query(self):
        """Test with special characters in query."""
        prompt = format_functiongemma_prompt('What is "hello" & <world>?')

        assert '"hello"' in prompt
        assert "&" in prompt
        assert "<world>" in prompt

    def test_multiline_query(self):
        """Test with multiline query."""
        query = """First line
Second line
Third line"""
        prompt = format_functiongemma_prompt(query)

        assert "First line" in prompt
        assert "Second line" in prompt
        assert "Third line" in prompt

    def test_unicode_in_query(self):
        """Test with unicode characters."""
        prompt = format_functiongemma_prompt("天気はどうですか？")

        assert "天気はどうですか？" in prompt

    def test_returns_string(self):
        """Test that function returns a string."""
        result = format_functiongemma_prompt("test")
        assert isinstance(result, str)

    def test_ends_with_model_turn(self):
        """Test that prompt ends with model turn marker."""
        prompt = format_functiongemma_prompt("test")

        # Should end with the model turn start
        assert prompt.strip().endswith("<start_of_turn>model")


# Mock classes for testing steer_model and compare_steering_effects
class MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.self_attn = nn.Linear(hidden_size, hidden_size)

    def __call__(self, h, **kwargs):
        return h


class MockBackbone(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]


class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        return MockModelOutput(logits)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2

    def encode(self, text, **kwargs):
        return [[1, 2, 3]]

    def decode(self, ids, **kwargs):
        return "generated text"


class MockDirectionBundle:
    """Mock DirectionBundle for testing."""

    def __init__(self, layers=None):
        if layers is None:
            layers = [0, 1]
        self.directions = {}
        for layer in layers:
            direction_mock = Mock()
            direction_mock.direction = mx.random.normal((64,))
            direction_mock.name = f"test_direction_L{layer}"
            direction_mock.positive_label = "positive"
            direction_mock.negative_label = "negative"
            self.directions[layer] = direction_mock


class MockAblationStudy:
    """Mock AblationStudy for from_pretrained."""

    def __init__(self, model_id):
        self.adapter = Mock()
        self.adapter.model = MockModel()
        self.adapter.tokenizer = MockTokenizer()


class TestSteerModel:
    """Tests for steer_model convenience function."""

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_steer_model_default_layers(self, mock_steering_class):
        """Test steer_model with default layers."""
        # Setup mock
        mock_steerer = Mock()
        mock_steerer.generate.return_value = "steered output"
        mock_steering_class.from_pretrained.return_value = mock_steerer

        # Create mock direction bundle
        bundle = MockDirectionBundle(layers=[0, 1])

        # Call function
        result = steer_model("test-model", "test prompt", bundle)

        # Verify
        mock_steering_class.from_pretrained.assert_called_once_with("test-model")
        mock_steerer.add_directions.assert_called_once_with(bundle)
        mock_steerer.generate.assert_called_once()
        assert result == "steered output"

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_steer_model_custom_layers(self, mock_steering_class):
        """Test steer_model with custom layers."""
        mock_steerer = Mock()
        mock_steerer.generate.return_value = "output"
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle(layers=[0, 1, 2])

        result = steer_model("test-model", "test prompt", bundle, layers=[1, 2])

        # Check config was created with correct layers
        call_args = mock_steerer.generate.call_args
        config = call_args[0][1]  # Second positional argument is config
        assert config.layers == [1, 2]
        assert result == "output"

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_steer_model_custom_coefficient(self, mock_steering_class):
        """Test steer_model with custom coefficient."""
        mock_steerer = Mock()
        mock_steerer.generate.return_value = "output"
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle()

        result = steer_model("test-model", "test prompt", bundle, coefficient=2.5)

        # Check config has correct coefficient
        call_args = mock_steerer.generate.call_args
        config = call_args[0][1]
        assert config.coefficient == 2.5
        assert result == "output"

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_steer_model_all_parameters(self, mock_steering_class):
        """Test steer_model with all parameters specified."""
        mock_steerer = Mock()
        mock_steerer.generate.return_value = "output"
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle(layers=[0, 1, 2])

        result = steer_model("test-model", "test prompt", bundle, layers=[1], coefficient=3.0)

        mock_steering_class.from_pretrained.assert_called_once_with("test-model")
        mock_steerer.add_directions.assert_called_once_with(bundle)
        call_args = mock_steerer.generate.call_args
        config = call_args[0][1]
        assert config.layers == [1]
        assert config.coefficient == 3.0
        assert result == "output"


class TestCompareSteringEffects:
    """Tests for compare_steering_effects convenience function."""

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_compare_steering_effects_default_coefficients(self, mock_steering_class):
        """Test compare_steering_effects with default coefficients."""
        mock_steerer = Mock()
        mock_steerer.compare_steering.return_value = {
            -2.0: "negative",
            -1.0: "slightly negative",
            0.0: "neutral",
            1.0: "slightly positive",
            2.0: "positive",
        }
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle()

        result = compare_steering_effects("test-model", "test prompt", bundle, layer=0)

        # Verify default coefficients
        call_args = mock_steerer.compare_steering.call_args
        assert call_args[0][0] == "test prompt"
        assert call_args[0][1] == [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert len(result) == 5
        assert result[-2.0] == "negative"
        assert result[0.0] == "neutral"
        assert result[2.0] == "positive"

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_compare_steering_effects_custom_coefficients(self, mock_steering_class):
        """Test compare_steering_effects with custom coefficients."""
        mock_steerer = Mock()
        mock_steerer.compare_steering.return_value = {
            -1.5: "output1",
            0.5: "output2",
            3.0: "output3",
        }
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle()

        result = compare_steering_effects(
            "test-model", "test prompt", bundle, layer=2, coefficients=[-1.5, 0.5, 3.0]
        )

        call_args = mock_steerer.compare_steering.call_args
        assert call_args[0][1] == [-1.5, 0.5, 3.0]
        assert len(result) == 3

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_compare_steering_effects_single_layer(self, mock_steering_class):
        """Test that compare_steering_effects uses correct layer config."""
        mock_steerer = Mock()
        mock_steerer.compare_steering.return_value = {0.0: "output"}
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle(layers=[0, 1, 2])

        compare_steering_effects("test-model", "test prompt", bundle, layer=1)

        # Check that config was created with correct layer
        call_args = mock_steerer.compare_steering.call_args
        config = call_args[0][2]  # Third positional argument is config
        assert config.layers == [1]

    @patch("chuk_lazarus.introspection.steering.core.ActivationSteering")
    def test_compare_steering_effects_returns_dict(self, mock_steering_class):
        """Test that compare_steering_effects returns a dict."""
        mock_steerer = Mock()
        mock_steerer.compare_steering.return_value = {
            0.0: "neutral",
            1.0: "positive",
        }
        mock_steering_class.from_pretrained.return_value = mock_steerer

        bundle = MockDirectionBundle()

        result = compare_steering_effects("test-model", "test prompt", bundle, layer=0)

        assert isinstance(result, dict)
        assert all(isinstance(k, float) for k in result.keys())
        assert all(isinstance(v, str) for v in result.values())
