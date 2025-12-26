"""Tests for inference/chat.py module."""

from chuk_lazarus.inference.chat import (
    ASSISTANT_SUFFIX,
    NEWLINE_DOUBLE,
    ChatHistory,
    ChatMessage,
    FallbackTemplate,
    Role,
    _format_simple,
    format_chat_prompt,
    format_history,
)


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.MODEL.value == "model"

    def test_role_display_name(self):
        """Test display_name method."""
        assert Role.SYSTEM.display_name() == "System"
        assert Role.USER.display_name() == "User"
        assert Role.ASSISTANT.display_name() == "Assistant"
        assert Role.MODEL.display_name() == "Model"

    def test_role_is_str_enum(self):
        """Test that Role is a string enum."""
        assert isinstance(Role.USER, str)
        assert Role.USER == "user"


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role=Role.USER, content="Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_to_tokenizer_format(self):
        """Test converting to tokenizer format."""
        msg = ChatMessage(role=Role.USER, content="Hello!")
        fmt = msg.to_tokenizer_format()
        assert fmt == {"role": "user", "content": "Hello!"}

    def test_to_tokenizer_format_all_roles(self):
        """Test tokenizer format for all roles."""
        for role in Role:
            msg = ChatMessage(role=role, content="test")
            fmt = msg.to_tokenizer_format()
            assert fmt["role"] == role.value
            assert fmt["content"] == "test"


class TestChatHistory:
    """Tests for ChatHistory model."""

    def test_empty_history(self):
        """Test empty chat history."""
        history = ChatHistory()
        assert len(history.messages) == 0
        assert history.system_message is None

    def test_add_user(self):
        """Test adding user message."""
        history = ChatHistory()
        result = history.add_user("Hello!")
        assert result is history  # Returns self for chaining
        assert len(history.messages) == 1
        assert history.messages[0].role == Role.USER
        assert history.messages[0].content == "Hello!"

    def test_add_assistant(self):
        """Test adding assistant message."""
        history = ChatHistory()
        result = history.add_assistant("Hi there!")
        assert result is history
        assert len(history.messages) == 1
        assert history.messages[0].role == Role.ASSISTANT
        assert history.messages[0].content == "Hi there!"

    def test_add_system(self):
        """Test setting system message."""
        history = ChatHistory()
        result = history.add_system("You are helpful.")
        assert result is history
        assert history.system_message == "You are helpful."

    def test_clear(self):
        """Test clearing history."""
        history = ChatHistory()
        history.add_system("System prompt")
        history.add_user("Message 1")
        history.add_assistant("Response 1")

        result = history.clear()
        assert result is history
        assert len(history.messages) == 0
        assert history.system_message == "System prompt"  # Preserved

    def test_chaining(self):
        """Test method chaining."""
        history = (
            ChatHistory()
            .add_system("Be helpful")
            .add_user("Hello")
            .add_assistant("Hi!")
            .add_user("How are you?")
        )
        assert history.system_message == "Be helpful"
        assert len(history.messages) == 3

    def test_to_tokenizer_format_empty(self):
        """Test tokenizer format with empty history."""
        history = ChatHistory()
        fmt = history.to_tokenizer_format()
        assert fmt == []

    def test_to_tokenizer_format_with_system(self):
        """Test tokenizer format with system message."""
        history = ChatHistory()
        history.add_system("Be helpful")
        history.add_user("Hello")

        fmt = history.to_tokenizer_format()
        assert len(fmt) == 2
        assert fmt[0] == {"role": "system", "content": "Be helpful"}
        assert fmt[1] == {"role": "user", "content": "Hello"}

    def test_to_tokenizer_format_without_system(self):
        """Test tokenizer format without system message."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")

        fmt = history.to_tokenizer_format()
        assert len(fmt) == 2
        assert fmt[0] == {"role": "user", "content": "Hello"}
        assert fmt[1] == {"role": "assistant", "content": "Hi!"}


class TestFallbackTemplate:
    """Tests for FallbackTemplate enum."""

    def test_template_values(self):
        """Test template enum values."""
        assert FallbackTemplate.SIMPLE.value == "simple"
        assert FallbackTemplate.CHATML.value == "chatml"


class TestConstants:
    """Tests for module constants."""

    def test_assistant_suffix(self):
        """Test assistant suffix constant."""
        assert ASSISTANT_SUFFIX == "Assistant:"

    def test_newline_double(self):
        """Test newline constant."""
        assert NEWLINE_DOUBLE == "\n\n"


class TestFormatSimple:
    """Tests for _format_simple function."""

    def test_format_empty(self):
        """Test formatting empty history."""
        history = ChatHistory()
        result = _format_simple(history)
        assert result == "\n\nAssistant:"

    def test_format_with_system(self):
        """Test formatting with system message."""
        history = ChatHistory()
        history.add_system("Be helpful")
        history.add_user("Hello")

        result = _format_simple(history)
        assert "System: Be helpful" in result
        assert "User: Hello" in result
        assert result.endswith("Assistant:")

    def test_format_conversation(self):
        """Test formatting multi-turn conversation."""
        history = ChatHistory()
        history.add_user("Hello")
        history.add_assistant("Hi!")
        history.add_user("How are you?")

        result = _format_simple(history)
        assert "User: Hello" in result
        assert "Assistant: Hi!" in result
        assert "User: How are you?" in result
        assert result.endswith("Assistant:")


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, has_template: bool = True, template_raises: bool = False):
        self.has_template = has_template
        self.template_raises = template_raises
        self.chat_template = "template" if has_template else None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.template_raises:
            raise ValueError("Template error")
        return f"<formatted>{len(messages)} messages</formatted>"


class TestFormatChatPrompt:
    """Tests for format_chat_prompt function."""

    def test_format_with_tokenizer_template(self):
        """Test formatting with tokenizer template."""
        tokenizer = MockTokenizer(has_template=True)
        result = format_chat_prompt(tokenizer, "Hello")
        assert "<formatted>" in result
        assert "1 messages" in result

    def test_format_with_system_message(self):
        """Test formatting with system message."""
        tokenizer = MockTokenizer(has_template=True)
        result = format_chat_prompt(tokenizer, "Hello", system_message="Be helpful")
        assert "2 messages" in result

    def test_format_fallback_no_template(self):
        """Test fallback when no template available."""
        tokenizer = MockTokenizer(has_template=False)
        result = format_chat_prompt(tokenizer, "Hello")
        assert "User: Hello" in result
        assert "Assistant:" in result

    def test_format_fallback_template_error(self):
        """Test fallback when template raises error."""
        tokenizer = MockTokenizer(has_template=True, template_raises=True)
        result = format_chat_prompt(tokenizer, "Hello")
        assert "User: Hello" in result
        assert "Assistant:" in result

    def test_format_with_generation_prompt_false(self):
        """Test with add_generation_prompt=False."""
        tokenizer = MockTokenizer(has_template=True)
        # The mock doesn't actually use this parameter, just verify it's passed
        result = format_chat_prompt(tokenizer, "Hello", add_generation_prompt=False)
        assert "<formatted>" in result


class TestFormatHistory:
    """Tests for format_history function."""

    def test_format_history_with_template(self):
        """Test formatting history with tokenizer template."""
        tokenizer = MockTokenizer(has_template=True)
        history = ChatHistory().add_user("Hello").add_assistant("Hi!")

        result = format_history(tokenizer, history)
        assert "<formatted>" in result
        assert "2 messages" in result

    def test_format_history_with_system(self):
        """Test formatting history with system message."""
        tokenizer = MockTokenizer(has_template=True)
        history = ChatHistory()
        history.add_system("Be helpful")
        history.add_user("Hello")

        result = format_history(tokenizer, history)
        assert "2 messages" in result  # system + user

    def test_format_history_fallback(self):
        """Test history formatting fallback."""
        tokenizer = MockTokenizer(has_template=False)
        history = ChatHistory().add_user("Hello")

        result = format_history(tokenizer, history)
        assert "User: Hello" in result
        assert "Assistant:" in result

    def test_format_history_empty(self):
        """Test formatting empty history."""
        tokenizer = MockTokenizer(has_template=True)
        history = ChatHistory()

        result = format_history(tokenizer, history)
        assert "<formatted>" in result
        assert "0 messages" in result
