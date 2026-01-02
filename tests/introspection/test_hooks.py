"""Tests for introspection hooks."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection import CaptureConfig, CapturedState, ModelHooks


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SimpleMLP(hidden_size)

    def __call__(self, x: mx.array, cache: mx.array | None = None) -> tuple[mx.array, None]:
        # Self-attention with residual
        h = self.norm1(x)
        h = self.attn(h, h, h)
        x = x + h

        # MLP with residual
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x, None


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing hooks."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [SimpleTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class SimpleForCausalLM(nn.Module):
    """Simple causal LM wrapper for testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = SimpleTransformerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


class TestCaptureConfig:
    """Tests for CaptureConfig dataclass."""

    def test_default_config(self):
        config = CaptureConfig()
        assert config.layers == "all"
        assert config.capture_hidden_states is True
        assert config.capture_attention_weights is False
        assert config.positions == "last"

    def test_custom_config(self):
        config = CaptureConfig(
            layers=[0, 2, 4],
            capture_attention_weights=True,
            positions="all",
        )
        assert config.layers == [0, 2, 4]
        assert config.capture_attention_weights is True
        assert config.positions == "all"


class TestCapturedState:
    """Tests for CapturedState dataclass."""

    def test_empty_state(self):
        state = CapturedState()
        assert state.num_layers_captured == 0
        assert state.captured_layers == []

    def test_state_with_data(self):
        state = CapturedState()
        state.hidden_states[0] = mx.zeros((1, 10, 64))
        state.hidden_states[4] = mx.zeros((1, 10, 64))
        state.hidden_states[8] = mx.zeros((1, 10, 64))

        assert state.num_layers_captured == 3
        assert state.captured_layers == [0, 4, 8]

    def test_clear(self):
        state = CapturedState()
        state.hidden_states[0] = mx.zeros((1, 10, 64))
        state.input_ids = mx.array([1, 2, 3])

        state.clear()

        assert state.num_layers_captured == 0
        assert state.input_ids is None

    def test_get_hidden_at_position(self):
        state = CapturedState()
        # [batch, seq, hidden]
        state.hidden_states[0] = mx.ones((1, 5, 64))

        # Get last position
        hidden = state.get_hidden_at_position(0, -1)
        assert hidden is not None
        assert hidden.shape == (64,)

        # Get first position
        hidden = state.get_hidden_at_position(0, 0)
        assert hidden is not None
        assert hidden.shape == (64,)

        # Non-existent layer
        hidden = state.get_hidden_at_position(99, 0)
        assert hidden is None


class TestModelHooks:
    """Tests for ModelHooks class."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleForCausalLM(vocab_size=1000, hidden_size=64, num_layers=4)

    @pytest.fixture
    def input_ids(self):
        """Create sample input IDs."""
        return mx.array([[1, 2, 3, 4, 5]])

    def test_init(self, model):
        hooks = ModelHooks(model)
        assert hooks.model is model
        assert hooks.config.layers == "all"
        assert hooks.state.num_layers_captured == 0

    def test_configure(self, model):
        hooks = ModelHooks(model)
        config = CaptureConfig(layers=[0, 2], capture_attention_weights=True)

        result = hooks.configure(config)

        assert result is hooks  # Chaining
        assert hooks.config is config

    def test_capture_layers_convenience(self, model):
        hooks = ModelHooks(model)
        hooks.capture_layers([0, 2], capture_attention=True)

        assert hooks.config.layers == [0, 2]
        assert hooks.config.capture_attention_weights is True

    def test_should_capture_layer_all(self, model):
        hooks = ModelHooks(model)
        hooks.config.layers = "all"

        assert hooks._should_capture_layer(0) is True
        assert hooks._should_capture_layer(99) is True

    def test_should_capture_layer_specific(self, model):
        hooks = ModelHooks(model)
        hooks.config.layers = [0, 2, 4]

        assert hooks._should_capture_layer(0) is True
        assert hooks._should_capture_layer(2) is True
        assert hooks._should_capture_layer(1) is False
        assert hooks._should_capture_layer(3) is False

    def test_get_layers(self, model):
        hooks = ModelHooks(model)
        layers = hooks._get_layers()

        assert len(layers) == 4
        assert all(isinstance(layer, SimpleTransformerLayer) for layer in layers)

    def test_get_embed_tokens(self, model):
        hooks = ModelHooks(model)
        embed = hooks._get_embed_tokens()

        assert embed is not None
        assert isinstance(embed, nn.Embedding)

    def test_get_final_norm(self, model):
        hooks = ModelHooks(model)
        norm = hooks._get_final_norm()

        assert norm is not None
        assert isinstance(norm, nn.RMSNorm)

    def test_get_lm_head(self, model):
        hooks = ModelHooks(model)
        lm_head = hooks._get_lm_head()

        assert lm_head is not None
        assert isinstance(lm_head, nn.Linear)

    def test_forward_captures_hidden_states(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                positions="all",
            )
        )

        logits = hooks.forward(input_ids)

        # Check logits shape
        assert logits is not None
        assert logits.shape == (1, 5, 1000)  # [batch, seq, vocab]

        # Check captured states
        assert hooks.state.num_layers_captured == 4
        assert 0 in hooks.state.hidden_states
        assert 3 in hooks.state.hidden_states

        # Check hidden state shapes
        for layer_idx, hidden in hooks.state.hidden_states.items():
            assert hidden.shape == (1, 5, 64)  # [batch, seq, hidden]

    def test_forward_captures_specific_layers(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers=[0, 2],
                capture_hidden_states=True,
                positions="all",
            )
        )

        hooks.forward(input_ids)

        assert hooks.state.num_layers_captured == 2
        assert 0 in hooks.state.hidden_states
        assert 2 in hooks.state.hidden_states
        assert 1 not in hooks.state.hidden_states
        assert 3 not in hooks.state.hidden_states

    def test_forward_captures_last_position(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                positions="last",
            )
        )

        hooks.forward(input_ids)

        # Should only have last position
        for hidden in hooks.state.hidden_states.values():
            assert hidden.shape == (1, 1, 64)  # [batch, 1, hidden]

    def test_forward_stores_embeddings(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.forward(input_ids)

        assert hooks.state.embeddings is not None
        assert hooks.state.embeddings.shape == (1, 5, 64)

    def test_forward_stores_final_hidden(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.forward(input_ids)

        assert hooks.state.final_hidden is not None
        assert hooks.state.final_hidden.shape == (1, 5, 64)

    def test_forward_stores_input_ids(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.forward(input_ids)

        assert hooks.state.input_ids is not None
        mx.array_equal(hooks.state.input_ids, input_ids)

    def test_forward_1d_input(self, model):
        hooks = ModelHooks(model)
        input_ids = mx.array([1, 2, 3, 4, 5])  # 1D

        logits = hooks.forward(input_ids)

        # Should have added batch dimension
        assert logits.shape == (1, 5, 1000)

    def test_forward_clears_previous_state(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.forward(input_ids)

        # Forward again with different input
        new_input = mx.array([[10, 20, 30]])
        hooks.forward(new_input)

        # State should be for new input
        mx.array_equal(hooks.state.input_ids, new_input)

    def test_forward_to_layer(self, model, input_ids):
        hooks = ModelHooks(model)

        # Get hidden state at layer 2
        hidden = hooks.forward_to_layer(input_ids, target_layer=2)

        assert hidden.shape == (1, 5, 64)

    def test_get_layer_logits(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers="all", positions="all"))
        hooks.forward(input_ids)

        # Get logits from layer 2
        logits = hooks.get_layer_logits(layer_idx=2, normalize=True)

        assert logits is not None
        assert logits.shape == (1, 5, 1000)  # [batch, seq, vocab]

    def test_get_layer_logits_uncaptured(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers=[0], positions="all"))
        hooks.forward(input_ids)

        # Layer 2 was not captured
        logits = hooks.get_layer_logits(layer_idx=2)

        assert logits is None

    def test_repr(self, model):
        hooks = ModelHooks(model)
        hooks.config.layers = [0, 2]

        repr_str = repr(hooks)

        assert "ModelHooks" in repr_str
        assert "[0, 2]" in repr_str

    def test_forward_with_detach(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                positions="all",
                detach=True,
            )
        )

        hooks.forward(input_ids)

        # States should be captured (detached from graph)
        assert hooks.state.num_layers_captured == 4

    def test_forward_with_explicit_positions(self, model, input_ids):
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                capture_hidden_states=True,
                positions=[0, 2, 4],  # Specific positions
            )
        )

        hooks.forward(input_ids)

        # Should capture specified positions
        for hidden in hooks.state.hidden_states.values():
            assert hidden.shape == (1, 3, 64)  # [batch, 3 positions, hidden]

    def test_forward_returns_none_when_no_logits(self, model, input_ids):
        # Create model without lm_head
        model_no_head = SimpleTransformerModel(vocab_size=1000, hidden_size=64, num_layers=4)
        hooks = ModelHooks(model_no_head)

        result = hooks.forward(input_ids, return_logits=False)

        assert result is None

    def test_maybe_slice_positions_2d(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(positions="last"))

        tensor = mx.ones((5, 64))  # [seq, hidden]
        sliced = hooks._maybe_slice_positions(tensor)

        assert sliced.shape == (1, 64)  # Only last position

    def test_maybe_slice_positions_4d(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(positions="last"))

        tensor = mx.ones((1, 4, 5, 5))  # [batch, heads, seq, seq]
        sliced = hooks._maybe_slice_positions(tensor)

        assert sliced.shape == (1, 4, 1, 5)  # Only last query position

    def test_maybe_slice_explicit_positions_2d(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(positions=[0, 2]))

        tensor = mx.ones((5, 64))  # [seq, hidden]
        sliced = hooks._maybe_slice_positions(tensor)

        assert sliced.shape == (2, 64)

    def test_maybe_slice_explicit_positions_4d(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(positions=[0, 2]))

        tensor = mx.ones((1, 4, 5, 5))  # [batch, heads, seq, seq]
        sliced = hooks._maybe_slice_positions(tensor)

        assert sliced.shape == (1, 4, 2, 5)

    def test_get_layers_from_transformer_h(self):
        """Test getting layers from model.transformer.h pattern."""

        class GPT2StyleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.wte = nn.Embedding(100, 64)
                self.transformer.h = [SimpleMLP(64) for _ in range(4)]

        model = GPT2StyleModel()
        hooks = ModelHooks(model)
        layers = hooks._get_layers()

        assert len(layers) == 4

    def test_get_embed_from_transformer_wte(self):
        """Test getting embedding from model.transformer.wte pattern."""

        class GPT2StyleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.wte = nn.Embedding(100, 64)
                self.transformer.h = [SimpleMLP(64) for _ in range(4)]

        model = GPT2StyleModel()
        hooks = ModelHooks(model)
        embed = hooks._get_embed_tokens()

        assert embed is not None
        assert isinstance(embed, nn.Embedding)

    def test_get_layers_direct(self):
        """Test getting layers from model.layers pattern."""

        class DirectLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.layers = [SimpleMLP(64) for _ in range(4)]
                self.norm = nn.RMSNorm(64)
                self.lm_head = nn.Linear(64, 100)

        model = DirectLayerModel()
        hooks = ModelHooks(model)
        layers = hooks._get_layers()

        assert len(layers) == 4

    def test_get_layers_raises_on_unsupported(self):
        """Test that getting layers from unsupported model raises error."""

        class WeirdModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [SimpleMLP(64) for _ in range(4)]  # Wrong name

        model = WeirdModel()
        hooks = ModelHooks(model)

        with pytest.raises(ValueError, match="Cannot find layers"):
            hooks._get_layers()

    def test_tied_embeddings(self):
        """Test getting lm_head from tied embeddings."""

        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.layers = [SimpleMLP(64) for _ in range(2)]
                self.norm = nn.RMSNorm(64)
                self.tie_word_embeddings = True

            def __call__(self, x):
                h = self.embed_tokens(x)
                for layer in self.layers:
                    h = layer(h)
                h = self.norm(h)
                return self.embed_tokens.as_linear(h)

        model = TiedModel()
        hooks = ModelHooks(model)
        lm_head = hooks._get_lm_head()

        # Should get the as_linear method from embeddings
        assert lm_head is not None

    def test_get_embedding_scale_from_model(self):
        """Test getting embedding scale from model backbone."""

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.layers = [SimpleMLP(64) for _ in range(2)]
                self.norm = nn.RMSNorm(64)

            @property
            def embedding_scale(self):
                return 8.0  # sqrt(64)

        class ScaledModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()
                self.lm_head = nn.Linear(64, 100)

        model = ScaledModel()
        hooks = ModelHooks(model)
        scale = hooks._get_embedding_scale()

        assert scale == 8.0

    def test_get_embedding_scale_override(self):
        """Test that explicit embedding scale overrides model's."""

        class ScaledModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(100, 64)
                self.model.layers = [SimpleMLP(64) for _ in range(2)]
                self.model.norm = nn.RMSNorm(64)
                self.lm_head = nn.Linear(64, 100)

            @property
            def embedding_scale(self):
                return 8.0

        model = ScaledModel()
        hooks = ModelHooks(model, embedding_scale=33.94)
        scale = hooks._get_embedding_scale()

        assert scale == 33.94  # Override, not model's 8.0
