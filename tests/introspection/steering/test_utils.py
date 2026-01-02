"""Tests for steering utils module."""

import json

from chuk_lazarus.introspection.steering.utils import format_functiongemma_prompt


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
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
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
