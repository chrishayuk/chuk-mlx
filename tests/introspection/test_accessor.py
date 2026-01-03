"""Tests for introspection accessor module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.accessor import (
    AsyncModelAccessor,
    ModelAccessor,
)


class MockConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        vocab_size: int = 1000,
        embedding_scale: float | None = None,
        d_model: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_scale = embedding_scale
        if d_model is not None:
            self.d_model = d_model


class MockEmbedding(nn.Module):
    """Mock embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.weight[input_ids]


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        return self.linear(x)


class MockModel(nn.Module):
    """Mock model with direct layers."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)


class MockNestedModel(nn.Module):
    """Mock model with nested structure (model.model.layers)."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()

        class InnerModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers):
                super().__init__()
                self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
                self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
                self.norm = nn.RMSNorm(hidden_size)

        self.model = InnerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class TestProtocols:
    """Tests for protocol definitions."""

    def test_has_layers_protocol(self):
        """Test model with layers attribute satisfies HasLayers structurally."""
        model = MockModel()
        # Check structural conformance (protocols are structural, not nominal)
        assert hasattr(model, "layers")
        assert isinstance(model.layers, list)

    def test_has_model_protocol(self):
        """Test nested model satisfies HasModel structurally."""
        model = MockNestedModel()
        assert hasattr(model, "model")

    def test_has_embed_tokens_protocol(self):
        """Test model with embed_tokens satisfies HasEmbedTokens structurally."""
        model = MockModel()
        assert hasattr(model, "embed_tokens")

    def test_has_norm_protocol(self):
        """Test model with norm satisfies HasNorm structurally."""
        model = MockModel()
        assert hasattr(model, "norm")

    def test_has_lm_head_protocol(self):
        """Test model with lm_head satisfies HasLMHead structurally."""
        model = MockNestedModel()
        assert hasattr(model, "lm_head")


class TestModelAccessor:
    """Tests for ModelAccessor class."""

    def test_init_with_direct_model(self):
        model = MockModel()
        config = MockConfig()
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.model is model
        assert accessor.config is config

    def test_init_without_config(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        assert accessor.model is model
        assert accessor.config is None

    def test_layers_property_direct(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        layers = accessor.layers
        assert isinstance(layers, list)
        assert len(layers) == 4

    def test_layers_property_nested(self):
        model = MockNestedModel(num_layers=6)
        accessor = ModelAccessor(model=model)
        layers = accessor.layers
        assert isinstance(layers, list)
        assert len(layers) == 6

    def test_layers_property_missing(self):
        # Create a model without layers
        model = nn.Module()
        accessor = ModelAccessor(model=model)
        with pytest.raises(AttributeError, match="Cannot find layers"):
            _ = accessor.layers

    def test_num_layers_property(self):
        model = MockModel(num_layers=8)
        accessor = ModelAccessor(model=model)
        assert accessor.num_layers == 8

    def test_embed_tokens_property_direct(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        embed = accessor.embed_tokens
        assert embed is model.embed_tokens

    def test_embed_tokens_property_nested(self):
        model = MockNestedModel()
        accessor = ModelAccessor(model=model)
        embed = accessor.embed_tokens
        assert embed is model.model.embed_tokens

    def test_embed_tokens_property_missing(self):
        model = nn.Module()
        accessor = ModelAccessor(model=model)
        with pytest.raises(AttributeError, match="Cannot find embed_tokens"):
            _ = accessor.embed_tokens

    def test_norm_property_direct(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        norm = accessor.norm
        assert norm is model.norm

    def test_norm_property_nested(self):
        model = MockNestedModel()
        accessor = ModelAccessor(model=model)
        norm = accessor.norm
        assert norm is model.model.norm

    def test_norm_property_missing(self):
        model = nn.Module()
        accessor = ModelAccessor(model=model)
        norm = accessor.norm
        assert norm is None

    def test_lm_head_property_present(self):
        model = MockNestedModel()
        accessor = ModelAccessor(model=model)
        lm_head = accessor.lm_head
        assert lm_head is model.lm_head

    def test_lm_head_property_missing(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        lm_head = accessor.lm_head
        assert lm_head is None

    def test_embedding_scale_from_config(self):
        config = MockConfig(embedding_scale=2.0)
        model = MockModel()
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.embedding_scale == 2.0

    def test_embedding_scale_none(self):
        config = MockConfig()
        model = MockModel()
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.embedding_scale is None

    def test_embedding_scale_no_config(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        assert accessor.embedding_scale is None

    def test_hidden_size_from_config(self):
        config = MockConfig(hidden_size=128)
        model = MockModel(hidden_size=128)
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.hidden_size == 128

    def test_hidden_size_from_d_model(self):
        config = MockConfig(d_model=256)
        delattr(config, "hidden_size")
        model = MockModel(hidden_size=256)
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.hidden_size == 256

    def test_hidden_size_from_embeddings(self):
        model = MockModel(hidden_size=64)
        accessor = ModelAccessor(model=model)
        assert accessor.hidden_size == 64

    def test_vocab_size_from_config(self):
        config = MockConfig(vocab_size=5000)
        model = MockModel(vocab_size=5000)
        accessor = ModelAccessor(model=model, config=config)
        assert accessor.vocab_size == 5000

    def test_vocab_size_from_embeddings(self):
        model = MockModel(vocab_size=3000)
        accessor = ModelAccessor(model=model)
        assert accessor.vocab_size == 3000

    def test_has_tied_embeddings_no_lm_head(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        # Should return True when no explicit head
        assert accessor.has_tied_embeddings is True

    def test_has_tied_embeddings_with_lm_head(self):
        model = MockNestedModel()
        accessor = ModelAccessor(model=model)
        # Different weight tensors - the result may be an mx.array or bool
        result = accessor.has_tied_embeddings
        if hasattr(result, "item"):
            result = result.item()
        assert result is False

    def test_get_layer_positive_index(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        layer = accessor.get_layer(2)
        assert layer is model.layers[2]

    def test_get_layer_negative_index(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        layer = accessor.get_layer(-1)
        assert layer is model.layers[-1]

    def test_get_layer_out_of_range(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        with pytest.raises(IndexError, match="Layer index .* out of range"):
            accessor.get_layer(10)

    def test_get_layer_negative_out_of_range(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        with pytest.raises(IndexError, match="Layer index .* out of range"):
            accessor.get_layer(-10)

    def test_set_layer_direct(self):
        model = MockModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        new_layer = MockLayer(64)
        accessor.set_layer(2, new_layer)
        assert model.layers[2] is new_layer

    def test_set_layer_nested(self):
        model = MockNestedModel(num_layers=4)
        accessor = ModelAccessor(model=model)
        new_layer = MockLayer(64)
        accessor.set_layer(1, new_layer)
        assert model.model.layers[1] is new_layer

    def test_set_layer_missing(self):
        model = nn.Module()
        accessor = ModelAccessor(model=model)
        with pytest.raises(AttributeError, match="Cannot set layer"):
            accessor.set_layer(0, MockLayer(64))

    def test_embed(self):
        model = MockModel(vocab_size=100, hidden_size=64)
        accessor = ModelAccessor(model=model)
        input_ids = mx.array([[1, 2, 3]])
        h = accessor.embed(input_ids)
        assert h.shape == (1, 3, 64)

    def test_embed_with_scale(self):
        config = MockConfig(embedding_scale=2.0)
        model = MockModel(vocab_size=100, hidden_size=64)
        accessor = ModelAccessor(model=model, config=config)
        input_ids = mx.array([[1, 2, 3]])
        h_scaled = accessor.embed(input_ids)

        # Compare with unscaled
        accessor_unscaled = ModelAccessor(model=model)
        h_unscaled = accessor_unscaled.embed(input_ids)

        # Scaled should be 2x unscaled
        assert mx.allclose(h_scaled, h_unscaled * 2.0).item()

    def test_apply_norm_and_head_with_lm_head(self):
        model = MockNestedModel(vocab_size=100, hidden_size=64)
        accessor = ModelAccessor(model=model)
        hidden_states = mx.random.normal((1, 5, 64))
        logits = accessor.apply_norm_and_head(hidden_states)
        assert logits.shape == (1, 5, 100)

    def test_apply_norm_and_head_tied_embeddings(self):
        model = MockModel(vocab_size=100, hidden_size=64)
        accessor = ModelAccessor(model=model)
        hidden_states = mx.random.normal((1, 5, 64))
        logits = accessor.apply_norm_and_head(hidden_states)
        assert logits.shape == (1, 5, 100)

    def test_apply_norm_and_head_no_norm(self):
        # Model without norm
        model = MockModel(vocab_size=100, hidden_size=64)
        delattr(model, "norm")
        accessor = ModelAccessor(model=model)
        hidden_states = mx.random.normal((1, 5, 64))
        logits = accessor.apply_norm_and_head(hidden_states)
        assert logits.shape == (1, 5, 100)

    def test_create_causal_mask(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        mask = accessor.create_causal_mask(5)
        assert mask.shape == (5, 5)
        # Check causality: future positions should be masked
        # Upper triangle (excluding diagonal) should be negative infinity
        assert mask[0, 1].item() < 0

    def test_create_causal_mask_with_dtype(self):
        model = MockModel()
        accessor = ModelAccessor(model=model)
        mask = accessor.create_causal_mask(3, dtype=mx.float32)
        assert mask.dtype == mx.float32


class TestAsyncModelAccessor:
    """Tests for AsyncModelAccessor class."""

    def test_inherits_from_model_accessor(self):
        model = MockModel()
        accessor = AsyncModelAccessor(model=model)
        assert isinstance(accessor, ModelAccessor)
        assert accessor.num_layers == 4

    @pytest.mark.asyncio
    async def test_forward_through_layers_all(self):
        model = MockModel(num_layers=4)
        accessor = AsyncModelAccessor(model=model)
        input_ids = mx.array([[1, 2, 3, 4]])

        captured = await accessor.forward_through_layers(input_ids)

        # Should capture all 4 layers by default
        assert len(captured) == 4
        assert all(i in captured for i in range(4))

        # Check shape
        for layer_idx, hidden in captured.items():
            assert hidden.shape[0] == 1  # batch
            assert hidden.shape[1] == 4  # seq_len

    @pytest.mark.asyncio
    async def test_forward_through_layers_subset(self):
        model = MockModel(num_layers=6)
        accessor = AsyncModelAccessor(model=model)
        input_ids = mx.array([[1, 2, 3]])

        captured = await accessor.forward_through_layers(
            input_ids,
            layers=[0, 2, 5],
        )

        assert len(captured) == 3
        assert 0 in captured
        assert 2 in captured
        assert 5 in captured

    @pytest.mark.asyncio
    async def test_forward_through_layers_no_capture(self):
        model = MockModel(num_layers=4)
        accessor = AsyncModelAccessor(model=model)
        input_ids = mx.array([[1, 2, 3]])

        captured = await accessor.forward_through_layers(
            input_ids,
            capture_hidden_states=False,
        )

        assert len(captured) == 0

    @pytest.mark.asyncio
    async def test_forward_through_layers_with_scale(self):
        config = MockConfig(embedding_scale=1.5)
        model = MockModel()
        accessor = AsyncModelAccessor(model=model, config=config)
        input_ids = mx.array([[1, 2]])

        captured = await accessor.forward_through_layers(input_ids, layers=[0])

        assert 0 in captured
        # Embeddings should be scaled
        assert captured[0] is not None
