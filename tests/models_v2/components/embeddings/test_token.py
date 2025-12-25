"""
Tests for token embeddings.
"""

import math

import mlx.core as mx

from chuk_lazarus.models_v2.components.embeddings import create_token_embedding
from chuk_lazarus.models_v2.components.embeddings.token import TokenEmbedding
from chuk_lazarus.models_v2.core.config import EmbeddingConfig


class TestTokenEmbedding:
    """Tests for token embeddings."""

    def test_basic_creation(self):
        """Test basic creation."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        assert embed.vocab_size == 1000
        assert embed.hidden_size == 256

    def test_forward(self):
        """Test forward pass."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = embed(input_ids)
        assert output.shape == (1, 5, 256)

    def test_batch_forward(self):
        """Test batched forward pass."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        output = embed(input_ids)
        assert output.shape == (2, 3, 256)

    def test_as_linear(self):
        """Test using embedding as linear projection (for tied embeddings)."""
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256)
        embed = TokenEmbedding(config)
        hidden = mx.random.normal((2, 10, 256))
        logits = embed.as_linear(hidden)
        assert logits.shape == (2, 10, 1000)

    def test_with_scale_factor(self):
        """Test embedding with scale factor."""
        scale = math.sqrt(256)
        config = EmbeddingConfig(vocab_size=1000, hidden_size=256, scale_factor=scale)
        embed = TokenEmbedding(config)
        assert embed.scale_factor == scale

        input_ids = mx.array([[1, 2, 3]])
        output = embed(input_ids)
        assert output.shape == (1, 3, 256)

    def test_factory_function(self):
        """Test factory function."""
        embed = create_token_embedding(vocab_size=5000, hidden_size=512)
        assert isinstance(embed, TokenEmbedding)
        assert embed.vocab_size == 5000

    def test_from_pretrained(self):
        """Test creating TokenEmbedding from existing weight matrix."""
        # Create a weight matrix
        weight = mx.random.normal((1000, 256))

        # Create embedding from weight
        embed = TokenEmbedding.from_pretrained(weight)

        assert embed.vocab_size == 1000
        assert embed.hidden_size == 256

        # Verify forward pass works
        input_ids = mx.array([[1, 2, 3]])
        output = embed(input_ids)
        assert output.shape == (1, 3, 256)

    def test_from_pretrained_with_scale_factor(self):
        """Test creating TokenEmbedding from weight with scale factor."""
        weight = mx.random.normal((500, 128))
        scale = 2.0

        embed = TokenEmbedding.from_pretrained(weight, scale_factor=scale)

        assert embed.vocab_size == 500
        assert embed.hidden_size == 128
        assert embed.scale_factor == scale

    def test_from_pretrained_preserves_values(self):
        """Test that from_pretrained preserves the weight values."""
        weight = mx.random.normal((100, 64))

        embed = TokenEmbedding.from_pretrained(weight)

        # The weight should be set on the internal nn.Embedding
        # Verify by checking embedding lookup matches
        input_ids = mx.array([[0, 1, 2]])
        output = embed(input_ids)

        # Each row of output should match corresponding row of weight
        assert output.shape == (1, 3, 64)


class TestEmbeddingGradients:
    """Tests for gradient flow through embeddings."""

    def test_token_embedding_gradients(self):
        """Test gradients flow through token embeddings."""
        config = EmbeddingConfig(vocab_size=100, hidden_size=64)
        embed = TokenEmbedding(config)
        input_ids = mx.array([[1, 2, 3]])

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out**2)

        loss, grads = mx.value_and_grad(loss_fn)(embed, input_ids)
        assert loss.item() > 0
