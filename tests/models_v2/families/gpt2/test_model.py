"""Tests for GPT2ForCausalLM."""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.families.gpt2 import GPT2Config, GPT2ForCausalLM


@pytest.fixture
def tiny_config() -> GPT2Config:
    """Create tiny config for testing."""
    return GPT2Config.tiny()


@pytest.fixture
def tiny_model(tiny_config: GPT2Config) -> GPT2ForCausalLM:
    """Create tiny model for testing."""
    return GPT2ForCausalLM(tiny_config)


class TestGPT2Model:
    """Tests for GPT2ForCausalLM basic functionality."""

    def test_model_creation(self, tiny_config: GPT2Config):
        """Test model can be created."""
        model = GPT2ForCausalLM(tiny_config)
        assert model is not None
        assert model.config == tiny_config

    def test_forward_pass(self, tiny_model: GPT2ForCausalLM):
        """Test forward pass produces expected output shape."""
        batch_size = 2
        seq_len = 10
        input_ids = mx.ones((batch_size, seq_len), dtype=mx.int32)

        output = tiny_model(input_ids)

        # Check logits shape
        assert output.logits.shape == (batch_size, seq_len, tiny_model.config.vocab_size)

        # Check cache is returned
        assert output.cache is not None
        assert len(output.cache) == tiny_model.config.num_hidden_layers

    def test_forward_with_cache(self, tiny_model: GPT2ForCausalLM):
        """Test forward pass with KV cache."""
        batch_size = 1
        prompt_len = 5
        gen_len = 3

        # Process prompt
        prompt = mx.ones((batch_size, prompt_len), dtype=mx.int32)
        output = tiny_model(prompt)
        cache = output.cache

        # Generate with cache
        next_token = mx.ones((batch_size, 1), dtype=mx.int32)
        for _ in range(gen_len):
            output = tiny_model(next_token, cache=cache)
            cache = output.cache
            # Single token input with cache should produce single token output
            assert output.logits.shape == (batch_size, 1, tiny_model.config.vocab_size)

    def test_output_hidden_states(self, tiny_model: GPT2ForCausalLM):
        """Test output_hidden_states returns all layer outputs."""
        input_ids = mx.ones((1, 5), dtype=mx.int32)

        output = tiny_model(input_ids, output_hidden_states=True)

        # Should have num_layers + 1 hidden states (embedding + each layer)
        assert output.hidden_states is not None
        assert len(output.hidden_states) == tiny_model.config.num_hidden_layers + 1

    def test_backbone_property(self, tiny_model: GPT2ForCausalLM):
        """Test backbone property returns transformer."""
        backbone = tiny_model.backbone
        assert backbone == tiny_model.transformer


class TestGPT2Generation:
    """Tests for GPT-2 text generation."""

    def test_generate_basic(self, tiny_model: GPT2ForCausalLM):
        """Test basic generation."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
        max_new_tokens = 5

        generated = tiny_model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )

        # Should have prompt + new tokens
        assert generated.shape[0] == 1
        assert generated.shape[1] <= prompt.shape[1] + max_new_tokens

    def test_generate_greedy(self, tiny_model: GPT2ForCausalLM):
        """Test greedy generation (temperature=0)."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

        generated = tiny_model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.0,
        )

        assert generated.shape[1] > prompt.shape[1]

    def test_generate_with_top_k(self, tiny_model: GPT2ForCausalLM):
        """Test generation with top-k sampling."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

        generated = tiny_model.generate(
            prompt,
            max_new_tokens=5,
            temperature=1.0,
            top_k=50,
        )

        assert generated.shape[1] > prompt.shape[1]

    def test_generate_with_stop_tokens(self, tiny_model: GPT2ForCausalLM):
        """Test generation stops at stop tokens."""
        prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

        # Use a token that might appear in output
        stop_tokens = [5]

        generated = tiny_model.generate(
            prompt,
            max_new_tokens=100,
            temperature=1.0,
            stop_tokens=stop_tokens,
        )

        # Generation should have stopped (may or may not hit stop token)
        assert generated.shape[1] <= prompt.shape[1] + 100


class TestGPT2EmbeddingTying:
    """Tests for GPT-2 embedding tying."""

    def test_tied_embeddings(self):
        """Test model with tied embeddings."""
        # tiny() already has tie_word_embeddings=True by default
        config = GPT2Config.tiny()

        model = GPT2ForCausalLM(config)

        # LM head should reference transformer embeddings
        assert model.lm_head.tied_embeddings is not None

    def test_untied_embeddings(self):
        """Test model with untied embeddings."""
        # Create config dict from tiny, but override tie_word_embeddings
        config_dict = GPT2Config.tiny().model_dump()
        config_dict["tie_word_embeddings"] = False
        config = GPT2Config(**config_dict)

        model = GPT2ForCausalLM(config)

        # LM head should have its own weights
        assert model.lm_head.tied_embeddings is None


class TestGPT2Gradients:
    """Tests for GPT-2 gradient computation."""

    def test_forward_backward(self, tiny_model: GPT2ForCausalLM):
        """Test that loss can be computed."""
        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        labels = mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32)

        output = tiny_model(input_ids, labels=labels)

        assert output.loss is not None
        # Loss should be a scalar
        assert output.loss.shape == ()


class TestGPT2BatchHandling:
    """Tests for GPT-2 batch handling."""

    def test_different_batch_sizes(self, tiny_model: GPT2ForCausalLM):
        """Test model handles different batch sizes."""
        for batch_size in [1, 2, 4]:
            input_ids = mx.ones((batch_size, 5), dtype=mx.int32)
            output = tiny_model(input_ids)
            assert output.logits.shape[0] == batch_size

    def test_different_sequence_lengths(self, tiny_model: GPT2ForCausalLM):
        """Test model handles different sequence lengths."""
        batch_size = 1
        for seq_len in [1, 5, 10, 50]:
            input_ids = mx.ones((batch_size, seq_len), dtype=mx.int32)
            output = tiny_model(input_ids)
            assert output.logits.shape[1] == seq_len
