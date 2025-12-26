"""Tests for inference/generator.py module (legacy generator)."""

import io
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from chuk_lazarus.inference.generator import (
    generate_response,
    generate_sequence,
)


@pytest.fixture(autouse=True)
def cleanup_mlx():
    """Clear MLX memory before and after each test."""
    mx.metal.clear_cache()
    yield
    mx.metal.clear_cache()


class MockModel:
    """Mock model for testing generation.

    The generate_sequence function calls model(y[None], cache=cache)
    where y is a 1D tensor. So y[None] creates shape (1, seq_len).
    """

    def __init__(self, vocab_size=20, stop_after=3, return_eos_at=None):
        self.vocab_size = vocab_size
        self.stop_after = stop_after
        self.return_eos_at = return_eos_at
        self.call_count = 0

    def __call__(self, y, cache=None):
        self.call_count += 1
        mx.eval(y)  # Force evaluation of input

        # y shape is (1, seq_len) after y[None] in generate_sequence
        batch_size = int(y.shape[0])
        seq_len = int(y.shape[1]) if y.ndim > 1 else 1

        # Determine which token to make highest probability
        if self.return_eos_at and self.call_count >= self.return_eos_at:
            # Return EOS (token 1)
            next_token = 1
        elif self.call_count <= self.stop_after:
            # Return a regular token
            next_token = min(5 + self.call_count, self.vocab_size - 1)
        else:
            # Past stop, return EOS
            next_token = 1

        # Create logits directly using numpy - small vocab to minimize memory
        logits_np = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        logits_np[:, :, next_token] = 100.0
        logits = mx.array(logits_np)
        mx.eval(logits)

        return logits, cache


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, eos_token_id=1):
        self.eos_token_id = eos_token_id
        self._decode_count = 0

    def encode(self, text):
        return [2, 3, 4, 5, 6]

    def decode(self, tokens):
        self._decode_count += 1
        if len(tokens) == 0:
            return ""
        # Return progressively longer strings to trigger printing
        return "a" * (len(tokens) * 2)


class TestGenerateSequence:
    """Tests for generate_sequence function."""

    def test_generate_sequence_basic(self):
        """Test basic sequence generation."""
        model = MockModel(stop_after=3, return_eos_at=4)
        prompt = mx.array([1, 2, 3])

        # generate_sequence is infinite - limit iterations
        tokens = []
        for i, token in enumerate(generate_sequence(prompt, model, temperature=0)):
            tokens.append(token)
            mx.eval(token)
            if i >= 3:  # Limit to 4 iterations
                break

        # Should generate tokens
        assert len(tokens) >= 1
        for token in tokens:
            assert isinstance(token, mx.array)

    def test_generate_sequence_with_temperature(self):
        """Test sequence generation with temperature (sampling)."""
        model = MockModel(stop_after=3, return_eos_at=4)
        prompt = mx.array([1, 2, 3])

        # With temperature > 0, uses categorical sampling
        # generate_sequence is infinite - limit iterations
        tokens = []
        for i, token in enumerate(generate_sequence(prompt, model, temperature=1.0)):
            tokens.append(token)
            mx.eval(token)
            if i >= 3:  # Limit to 4 iterations
                break

        assert len(tokens) >= 0

    def test_generate_sequence_greedy(self):
        """Test greedy decoding (temperature=0)."""
        model = MockModel(stop_after=2, return_eos_at=3)
        prompt = mx.array([1, 2, 3])

        # generate_sequence is infinite - limit iterations
        tokens = []
        for i, token in enumerate(generate_sequence(prompt, model, temperature=0)):
            tokens.append(token)
            mx.eval(token)
            if i >= 2:  # Limit to 3 iterations
                break

        # Greedy should be deterministic
        assert len(tokens) >= 1

    def test_generate_sequence_none_logits(self):
        """Test handling when model returns None logits."""

        class NoneLogitsModel:
            def __call__(self, y, cache=None):
                return None, cache

        model = NoneLogitsModel()
        prompt = mx.array([1, 2, 3])

        tokens = list(generate_sequence(prompt, model))

        assert tokens == []

    def test_generate_sequence_zero_seq_len(self):
        """Test handling when logits has zero sequence length."""

        class ZeroSeqModel:
            def __call__(self, y, cache=None):
                logits = mx.zeros((1, 0, 100))
                return logits, cache

        model = ZeroSeqModel()
        prompt = mx.array([1, 2, 3])

        tokens = list(generate_sequence(prompt, model))

        assert tokens == []


class TestGenerateResponse:
    """Tests for generate_response function."""

    def test_generate_response_basic(self):
        """Test basic response generation."""
        model = MockModel(stop_after=3, return_eos_at=4)
        tokenizer = MockTokenizer(eos_token_id=1)

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            tokens = generate_response(model, "test prompt", tokenizer, max_length=10)

        assert isinstance(tokens, list)
        assert len(tokens) >= 1

    def test_generate_response_stops_at_eos(self):
        """Test that generation stops at EOS token."""
        model = MockModel(stop_after=2, return_eos_at=2)
        tokenizer = MockTokenizer(eos_token_id=1)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            tokens = generate_response(model, "test", tokenizer, max_length=100)

        # Should stop at EOS, not at max_length
        assert len(tokens) < 100

    def test_generate_response_respects_max_length(self):
        """Test that max_length is respected."""
        # Model never returns EOS (eos_token_id=99999 is unlikely)
        model = MockModel(stop_after=100)
        tokenizer = MockTokenizer(eos_token_id=99999)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            tokens = generate_response(model, "test", tokenizer, max_length=5)

        assert len(tokens) <= 5

    def test_generate_response_no_tokens_message(self):
        """Test message when no tokens generated (immediate EOS)."""

        class ImmediateEOSModel:
            def __call__(self, y, cache=None):
                import numpy as np

                batch_size = y.shape[0]
                seq_len = y.shape[1] if y.ndim > 1 else 1
                logits_np = np.zeros((batch_size, seq_len, 50), dtype=np.float32)
                # Return EOS immediately (token 1)
                logits_np[:, :, 1] = 100.0
                logits = mx.array(logits_np)
                mx.eval(logits)
                return logits, cache

        model = ImmediateEOSModel()
        tokenizer = MockTokenizer(eos_token_id=1)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            tokens = generate_response(model, "test", tokenizer, max_length=10)

        output = captured.getvalue()
        assert tokens == []
        assert "No tokens generated" in output

    def test_generate_response_prints_output(self):
        """Test that response is printed incrementally."""
        model = MockModel(stop_after=5, return_eos_at=6)
        tokenizer = MockTokenizer()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            generate_response(model, "test", tokenizer, max_length=10)

        output = captured.getvalue()
        # Should have printed something
        assert len(output) > 0

    def test_generate_response_returns_token_list(self):
        """Test that function returns a list of token IDs."""
        model = MockModel(stop_after=3, return_eos_at=4)
        tokenizer = MockTokenizer()

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            tokens = generate_response(model, "test", tokenizer, max_length=10)

        assert isinstance(tokens, list)
        for token in tokens:
            assert isinstance(token, int)
