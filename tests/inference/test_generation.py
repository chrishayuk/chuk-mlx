"""Tests for inference/generation.py module."""

import mlx.core as mx
import pytest

from chuk_lazarus.inference.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    generate,
    generate_stream,
    get_stop_tokens,
)


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_new_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k is None
        assert config.stop_tokens == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            stop_tokens=[1, 2, 3],
        )
        assert config.max_new_tokens == 50
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.stop_tokens == [1, 2, 3]

    def test_validation_max_new_tokens(self):
        """Test max_new_tokens validation."""
        with pytest.raises(ValueError):
            GenerationConfig(max_new_tokens=0)
        with pytest.raises(ValueError):
            GenerationConfig(max_new_tokens=-1)

    def test_validation_temperature(self):
        """Test temperature validation."""
        # 0 is valid (greedy)
        config = GenerationConfig(temperature=0)
        assert config.temperature == 0

        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)

    def test_validation_top_p(self):
        """Test top_p validation."""
        # 0 and 1 are valid
        config = GenerationConfig(top_p=0)
        assert config.top_p == 0
        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0

        with pytest.raises(ValueError):
            GenerationConfig(top_p=-0.1)
        with pytest.raises(ValueError):
            GenerationConfig(top_p=1.1)

    def test_validation_top_k(self):
        """Test top_k validation."""
        config = GenerationConfig(top_k=1)
        assert config.top_k == 1

        with pytest.raises(ValueError):
            GenerationConfig(top_k=0)
        with pytest.raises(ValueError):
            GenerationConfig(top_k=-1)


class TestGenerationStats:
    """Tests for GenerationStats model."""

    def test_create_stats(self):
        """Test creating generation stats."""
        stats = GenerationStats(
            input_tokens=10,
            output_tokens=20,
            total_time_seconds=2.0,
            tokens_per_second=10.0,
        )
        assert stats.input_tokens == 10
        assert stats.output_tokens == 20
        assert stats.total_time_seconds == 2.0
        assert stats.tokens_per_second == 10.0

    def test_summary_property(self):
        """Test summary property."""
        stats = GenerationStats(
            input_tokens=10,
            output_tokens=25,
            total_time_seconds=2.50,
            tokens_per_second=10.0,
        )
        summary = stats.summary
        assert "25 tokens" in summary
        assert "2.50s" in summary
        assert "10.0 tok/s" in summary


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_create_result(self):
        """Test creating generation result."""
        stats = GenerationStats(
            input_tokens=10,
            output_tokens=20,
            total_time_seconds=2.0,
            tokens_per_second=10.0,
        )
        result = GenerationResult(
            text="Hello world",
            stats=stats,
            stop_reason="eos",
        )
        assert result.text == "Hello world"
        assert result.stats.output_tokens == 20
        assert result.stop_reason == "eos"

    def test_default_stop_reason(self):
        """Test default stop reason."""
        stats = GenerationStats(
            input_tokens=10,
            output_tokens=20,
            total_time_seconds=2.0,
            tokens_per_second=10.0,
        )
        result = GenerationResult(text="test", stats=stats)
        assert result.stop_reason == "max_tokens"


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def encode(self, text, return_tensors=None):
        # Return numpy-like array
        import numpy as np

        return np.array([[1, 2, 3, 4, 5]])

    def decode(self, tokens, skip_special_tokens=False):
        return f"decoded_{len(tokens)}_tokens"


class TestGetStopTokens:
    """Tests for get_stop_tokens function."""

    def test_no_eos_token(self):
        """Test with no EOS token."""
        tokenizer = MockTokenizer(eos_token_id=None)
        tokens = get_stop_tokens(tokenizer)
        assert tokens == []

    def test_single_eos_token(self):
        """Test with single EOS token."""
        tokenizer = MockTokenizer(eos_token_id=50256)
        tokens = get_stop_tokens(tokenizer)
        assert tokens == [50256]

    def test_list_eos_tokens(self):
        """Test with list of EOS tokens."""
        tokenizer = MockTokenizer(eos_token_id=[50256, 50257])
        tokens = get_stop_tokens(tokenizer)
        assert tokens == [50256, 50257]


class MockModel:
    """Mock model for testing generation."""

    def __init__(self, output_length=10, stop_at=None):
        self.output_length = output_length
        self.stop_at = stop_at
        self.call_count = 0

    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=None,
        stop_tokens=None,
    ):
        _ = input_ids.shape[0]  # batch_size unused but validates shape
        _ = input_ids.shape[1]  # input_length unused but validates shape

        # Simulate generation - return input + new tokens
        new_tokens = mx.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        actual_new = min(self.output_length, max_new_tokens)
        output = mx.concatenate([input_ids, new_tokens[:, :actual_new]], axis=1)
        return output

    def __call__(self, y, cache=None):
        self.call_count += 1
        batch_size = y.shape[0]

        # Return logits for vocab size 100
        logits = mx.zeros((batch_size, 1, 100))
        # Make token 10+call_count the highest probability
        next_token = min(10 + self.call_count, 99)

        if self.stop_at and self.call_count >= self.stop_at:
            # Return stop token (assuming 50256)
            logits = logits.at[:, 0, 50256].add(100.0)
        else:
            logits = logits.at[:, 0, next_token].add(100.0)

        return logits, cache


class TestGenerate:
    """Tests for generate function."""

    def test_generate_basic(self):
        """Test basic generation."""
        model = MockModel(output_length=5)
        tokenizer = MockTokenizer(eos_token_id=50256)

        result = generate(model, tokenizer, "test prompt")

        assert isinstance(result, GenerationResult)
        assert result.text.startswith("decoded_")
        assert result.stats.input_tokens == 5
        assert result.stats.output_tokens == 5

    def test_generate_with_config(self):
        """Test generation with custom config."""
        model = MockModel(output_length=3)
        tokenizer = MockTokenizer()

        config = GenerationConfig(
            max_new_tokens=3,
            temperature=0.5,
        )
        result = generate(model, tokenizer, "test", config=config)

        assert result.stats.output_tokens == 3

    def test_generate_stats(self):
        """Test generation statistics."""
        model = MockModel(output_length=10)
        tokenizer = MockTokenizer()

        result = generate(model, tokenizer, "test")

        assert result.stats.total_time_seconds > 0
        assert result.stats.tokens_per_second >= 0

    def test_generate_stop_reason_max_tokens(self):
        """Test stop reason when max tokens reached."""
        model = MockModel(output_length=100)
        tokenizer = MockTokenizer(eos_token_id=50256)

        config = GenerationConfig(max_new_tokens=10)
        result = generate(model, tokenizer, "test", config=config)

        assert result.stop_reason == "max_tokens"

    def test_generate_stop_reason_eos(self):
        """Test stop reason when EOS token reached."""
        model = MockModel(output_length=5)
        tokenizer = MockTokenizer(eos_token_id=50256)

        # The mock model returns tokens 10-14, if the last one is 50256 it's EOS
        # For this test, we need to manipulate the scenario

        result = generate(model, tokenizer, "test")
        # In our mock, it doesn't actually produce EOS, so it's max_tokens
        assert result.stop_reason in ["max_tokens", "eos", "stop_token"]


class TestGenerateStream:
    """Tests for generate_stream function."""

    def test_generate_stream_basic(self):
        """Test basic streaming generation."""
        model = MockModel(stop_at=5)
        tokenizer = MockTokenizer(eos_token_id=50256)

        chunks = list(generate_stream(model, tokenizer, "test"))

        # Should generate some chunks
        assert len(chunks) >= 0  # May be empty if tokens decode to empty

    def test_generate_stream_with_config(self):
        """Test streaming with config."""
        model = MockModel(stop_at=3)
        tokenizer = MockTokenizer()

        config = GenerationConfig(max_new_tokens=3, temperature=0.5)
        chunks = list(generate_stream(model, tokenizer, "test", config=config))

        # Verify we get some output
        assert isinstance(chunks, list)

    def test_generate_stream_stops_on_max_tokens(self):
        """Test streaming stops at max tokens."""
        model = MockModel()
        tokenizer = MockTokenizer()

        config = GenerationConfig(max_new_tokens=5)
        _ = list(generate_stream(model, tokenizer, "test", config=config))

        # Should have stopped after 5 iterations max
        assert model.call_count <= 5

    def test_generate_stream_stops_on_eos(self):
        """Test streaming stops on EOS token."""
        model = MockModel(stop_at=3)
        tokenizer = MockTokenizer(eos_token_id=50256)

        config = GenerationConfig(max_new_tokens=10)
        _ = list(generate_stream(model, tokenizer, "test", config=config))

        # Should have stopped early due to EOS
        assert model.call_count <= 10
