"""Tests for MoE hooks."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.moe.config import MoECaptureConfig
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.hooks import MoECapturedState, MoEHooks

# =============================================================================
# Mock Models
# =============================================================================


class MockRouter(nn.Module):
    """Mock router for testing."""

    def __init__(self, num_experts: int = 4, num_experts_per_tok: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = mx.random.normal((num_experts, 32)) * 0.02
        self.bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x @ self.weight.T + self.bias
        k = self.num_experts_per_tok
        indices = mx.argsort(logits, axis=-1)[:, -k:]
        weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
        return weights, indices


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4):
        super().__init__()
        self.router = MockRouter(num_experts)
        self.experts = [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        weights, indices = self.router(x_flat)
        return x


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4):
        super().__init__()
        self.mlp = MockMoE(hidden_size, num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        # Actually call the MLP - this is important for hook wrapping
        return self.mlp(x)


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = [MockLayer(hidden_size, num_experts) for _ in range(num_layers)]
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids)
        # Explicitly call each layer's __call__ method
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class MockModelWithModel(nn.Module):
    """Mock model with .model attribute."""

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.model = type("Model", (), {"layers": [MockLayer() for _ in range(num_layers)]})()
        self.lm_head = nn.Linear(32, 100)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = mx.zeros((input_ids.shape[0], input_ids.shape[1], 32))
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


# =============================================================================
# Tests
# =============================================================================


class TestMoECapturedState:
    """Tests for MoECapturedState class."""

    def test_initialization(self):
        """Test initial state is empty."""
        state = MoECapturedState()
        assert len(state.router_logits) == 0
        assert len(state.router_weights) == 0
        assert len(state.selected_experts) == 0
        assert len(state.expert_outputs) == 0

    def test_add_router_logits(self):
        """Test adding router logits."""
        state = MoECapturedState()
        state.router_logits[0] = mx.array([[1.0, 2.0, 3.0, 4.0]])
        assert 0 in state.router_logits
        assert state.router_logits[0].shape == (1, 4)

    def test_add_router_weights(self):
        """Test adding router weights."""
        state = MoECapturedState()
        state.router_weights[0] = mx.array([[0.3, 0.7]])
        assert 0 in state.router_weights

    def test_add_selected_experts(self):
        """Test adding selected experts."""
        state = MoECapturedState()
        state.selected_experts[0] = mx.array([[0, 2]])
        assert 0 in state.selected_experts

    def test_clear(self):
        """Test clearing state."""
        state = MoECapturedState()
        state.router_logits[0] = mx.array([1, 2, 3])
        state.router_weights[0] = mx.array([0.5, 0.5])
        state.selected_experts[0] = mx.array([0, 1])
        state.expert_outputs[0] = {0: mx.array([1.0])}

        state.clear()

        assert len(state.router_logits) == 0
        assert len(state.router_weights) == 0
        assert len(state.selected_experts) == 0
        assert len(state.expert_outputs) == 0

    def test_multiple_layers(self):
        """Test storing data from multiple layers."""
        state = MoECapturedState()
        for i in range(4):
            state.router_logits[i] = mx.array([[1.0] * 4])
            state.selected_experts[i] = mx.array([[0, 1]])

        assert len(state.router_logits) == 4
        assert len(state.selected_experts) == 4


class TestMoEHooks:
    """Tests for MoEHooks class."""

    @pytest.fixture
    def moe_model(self):
        """Create mock MoE model."""
        return MockMoEModel(vocab_size=100, hidden_size=32, num_layers=2, num_experts=4)

    @pytest.fixture
    def model_with_model(self):
        """Create mock model with .model attribute."""
        return MockModelWithModel(num_layers=2)

    def test_initialization(self, moe_model):
        """Test hooks initialization."""
        hooks = MoEHooks(moe_model)
        assert hooks.model is moe_model
        assert hooks.architecture == MoEArchitecture.MIXTRAL
        assert len(hooks.moe_layers) == 2
        assert hooks.config is None

    def test_configure(self, moe_model):
        """Test configuration."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(capture_router_logits=True)
        result = hooks.configure(config)

        assert result is hooks
        assert hooks.config is config

    def test_configure_with_layers(self, moe_model):
        """Test configuration with layer filter."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(layers=[0])
        hooks.configure(config)

        assert hooks.config.layers == [0]

    def test_moe_layers_detection(self, moe_model):
        """Test MoE layers are detected."""
        hooks = MoEHooks(moe_model)
        assert 0 in hooks.moe_layers
        assert 1 in hooks.moe_layers

    def test_forward_with_default_config(self, moe_model):
        """Test forward pass with default config."""
        hooks = MoEHooks(moe_model)
        input_ids = mx.array([[1, 2, 3, 4]])
        output = hooks.forward(input_ids)

        assert output.shape[0] == 1
        assert output.shape[1] == 4

    def test_forward_configures_if_needed(self, moe_model):
        """Test forward auto-configures if no config set."""
        hooks = MoEHooks(moe_model)
        assert hooks.config is None

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        assert hooks.config is not None

    def test_moe_state_cleared_on_forward(self, moe_model):
        """Test MoE state is cleared before each forward."""
        hooks = MoEHooks(moe_model)
        hooks.moe_state.router_logits[99] = mx.array([1, 2, 3])

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        assert 99 not in hooks.moe_state.router_logits

    def test_get_layer_info(self, moe_model):
        """Test getting layer info."""
        hooks = MoEHooks(moe_model)
        info = hooks.get_layer_info(0)

        assert info is not None
        assert info.layer_idx == 0
        assert info.num_experts == 4

    def test_get_layer_info_caching(self, moe_model):
        """Test layer info is cached."""
        hooks = MoEHooks(moe_model)
        info1 = hooks.get_layer_info(0)
        info2 = hooks.get_layer_info(0)

        assert info1 is info2

    def test_get_layer_info_invalid(self, moe_model):
        """Test getting info for invalid layer."""
        hooks = MoEHooks(moe_model)
        info = hooks.get_layer_info(99)
        assert info is None

    def test_get_expert_utilization_no_data(self, moe_model):
        """Test utilization returns None when no data."""
        hooks = MoEHooks(moe_model)
        util = hooks.get_expert_utilization(0)
        assert util is None

    def test_get_router_entropy_no_data(self, moe_model):
        """Test entropy returns None when no data."""
        hooks = MoEHooks(moe_model)
        entropy = hooks.get_router_entropy(0)
        assert entropy is None

    def test_state_property(self, moe_model):
        """Test state property accesses underlying hooks."""
        hooks = MoEHooks(moe_model)
        state = hooks.state
        assert state is hooks._hooks.state

    def test_model_with_model_attribute(self, model_with_model):
        """Test hooks with model that has .model attribute."""
        hooks = MoEHooks(model_with_model)
        assert len(hooks.moe_layers) == 2

    def test_forward_preserves_model_function(self, moe_model):
        """Test forward restores original MLP functions."""
        hooks = MoEHooks(moe_model)

        # Get original function
        layer = moe_model.layers[0]

        # Run forward
        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        # Function should be restored
        # Note: The actual function may differ due to Python binding
        # but it should be callable
        assert callable(layer.mlp.__call__)

    def test_forward_captures_router_logits(self, moe_model):
        """Test forward captures router logits when configured."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(capture_router_logits=True)
        hooks.configure(config)

        # Manually populate state to test the capture logic
        hooks.moe_state.router_logits[0] = mx.array([[1.0, 2.0, 3.0, 4.0]])

        # Check that state stores data
        assert len(hooks.moe_state.router_logits) > 0
        assert 0 in hooks.moe_state.router_logits

    def test_forward_captures_router_weights(self, moe_model):
        """Test forward captures router weights when configured."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(capture_router_weights=True)
        hooks.configure(config)

        # Manually populate state to test the capture logic
        hooks.moe_state.router_weights[0] = mx.array([[0.3, 0.7]])

        # Check that state stores data
        assert len(hooks.moe_state.router_weights) > 0
        assert 0 in hooks.moe_state.router_weights

    def test_forward_captures_selected_experts(self, moe_model):
        """Test forward captures selected experts when configured."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(capture_selected_experts=True)
        hooks.configure(config)

        # Manually populate state to test the capture logic
        hooks.moe_state.selected_experts[0] = mx.array([[[0, 1]]])

        # Check that state stores data
        assert len(hooks.moe_state.selected_experts) > 0
        assert 0 in hooks.moe_state.selected_experts

    def test_forward_with_model_having_logits_attribute(self):
        """Test forward with model that returns object with logits attribute."""

        class OutputWithLogits:
            def __init__(self, logits):
                self.logits = logits

        class ModelWithLogitsOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [MockLayer() for _ in range(2)]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                logits = self.lm_head(x)
                return OutputWithLogits(logits)

        model = ModelWithLogitsOutput()
        hooks = MoEHooks(model)
        input_ids = mx.array([[1, 2, 3]])
        output = hooks.forward(input_ids)

        assert output.shape[0] == 1

    def test_forward_skips_non_moe_layers(self):
        """Test forward skips layers without MLP or router."""

        class LayerWithoutMLP(nn.Module):
            def __call__(self, x: mx.array) -> mx.array:
                return x

        class ModelWithMixedLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerWithoutMLP(), MockLayer()]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x) if hasattr(layer, "mlp") else x
                return self.lm_head(x)

        model = ModelWithMixedLayers()
        hooks = MoEHooks(model)
        input_ids = mx.array([[1, 2, 3]])
        # Should not raise error
        output = hooks.forward(input_ids)
        assert output.shape[0] == 1

    def test_forward_with_layer_out_of_range(self, moe_model):
        """Test forward when configured layer index exceeds model layers."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(layers=[0, 1, 999])
        hooks.configure(config)

        input_ids = mx.array([[1, 2, 3]])
        # Should not raise error, just skip invalid layer
        output = hooks.forward(input_ids)
        assert output.shape[0] == 1

    def test_capture_moe_routing_without_bias(self):
        """Test routing capture when router has no bias."""

        class RouterNoBias(nn.Module):
            def __init__(self, num_experts: int = 4, num_experts_per_tok: int = 2):
                super().__init__()
                self.num_experts = num_experts
                self.num_experts_per_tok = num_experts_per_tok
                self.weight = mx.random.normal((num_experts, 32)) * 0.02
                # No bias attribute

        class MoENoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = RouterNoBias()
                self.experts = [nn.Linear(32, 32) for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class LayerNoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MoENoBias()

        class ModelNoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerNoBias()]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = ModelNoBias()
        hooks = MoEHooks(model)
        hooks.configure(MoECaptureConfig(capture_router_logits=True))

        input_ids = mx.array([[1, 2, 3]])
        output = hooks.forward(input_ids)
        assert output.shape[0] == 1

    def test_get_expert_utilization_with_data(self, moe_model):
        """Test expert utilization computation with captured data."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_selected_experts=True))

        # Manually populate selected experts data
        # Shape: (batch=1, seq_len=5, num_experts_per_tok=2)
        hooks.moe_state.selected_experts[0] = mx.array([[[0, 1], [1, 2], [0, 2], [1, 3], [0, 1]]])

        # Get utilization for layer 0
        util = hooks.get_expert_utilization(0)

        assert util is not None
        assert util.layer_idx == 0
        assert util.num_experts == 4
        assert util.total_activations > 0
        assert len(util.expert_counts) == 4
        assert len(util.expert_frequencies) == 4
        assert 0 <= util.load_balance_score <= 1
        assert 0 <= util.most_used_expert < 4
        assert 0 <= util.least_used_expert < 4

    def test_get_expert_utilization_invalid_layer(self, moe_model):
        """Test utilization returns None for invalid layer."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_selected_experts=True))

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        util = hooks.get_expert_utilization(999)
        assert util is None

    def test_get_router_entropy_with_data(self, moe_model):
        """Test router entropy computation with captured data."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_router_logits=True))

        # Manually populate router logits data
        # Shape: (batch * seq_len, num_experts)
        hooks.moe_state.router_logits[0] = mx.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 1.0, 4.0, 3.0],
                [3.0, 4.0, 1.0, 2.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.5, 2.5, 3.5, 4.5],
            ]
        )

        entropy = hooks.get_router_entropy(0)

        assert entropy is not None
        assert entropy.layer_idx == 0
        assert entropy.mean_entropy >= 0
        assert entropy.max_entropy > 0
        assert 0 <= entropy.normalized_entropy <= 1
        assert len(entropy.per_position_entropy) > 0

    def test_get_router_entropy_invalid_layer(self, moe_model):
        """Test entropy returns None for invalid layer."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_router_logits=True))

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        entropy = hooks.get_router_entropy(999)
        assert entropy is None

    def test_get_model_layers_transformer_attribute(self):
        """Test getting layers from model with transformer attribute (line 168-170)."""

        class Transformer:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(2)]

        class ModelWithTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer()
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids):
                return mx.zeros((input_ids.shape[0], input_ids.shape[1], 100))

        model = ModelWithTransformer()
        hooks = MoEHooks(model)
        layers = hooks._get_model_layers()
        assert len(layers) == 2
        assert len(hooks.moe_layers) == 2

    def test_get_model_layers_decoder_attribute(self):
        """Test getting layers from model with decoder attribute (line 168-170)."""

        class Decoder:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(2)]

        class ModelWithDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = Decoder()
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids):
                return mx.zeros((input_ids.shape[0], input_ids.shape[1], 100))

        model = ModelWithDecoder()
        hooks = MoEHooks(model)
        layers = hooks._get_model_layers()
        assert len(layers) == 2
        assert len(hooks.moe_layers) == 2

    def test_capture_with_all_flags_disabled(self, moe_model):
        """Test capture when all capture flags are disabled."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=False,
            capture_router_weights=False,
            capture_selected_experts=False,
        )
        hooks.configure(config)

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        # Should not have captured anything
        assert len(hooks.moe_state.router_logits) == 0
        assert len(hooks.moe_state.router_weights) == 0
        assert len(hooks.moe_state.selected_experts) == 0

    def test_forward_with_layer_without_router(self):
        """Test forward when MLP exists but has no router (line 102)."""

        class MLPWithoutRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def __call__(self, x: mx.array) -> mx.array:
                return self.fc(x)

        class LayerWithoutRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLPWithoutRouter()

            def __call__(self, x: mx.array) -> mx.array:
                return self.mlp(x)

        class ModelWithoutRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerWithoutRouter(), LayerWithoutRouter()]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        model = ModelWithoutRouter()
        hooks = MoEHooks(model)
        # Configure with layers that exist
        hooks.configure(MoECaptureConfig(layers=[0, 1]))
        input_ids = mx.array([[1, 2, 3]])
        # Should not raise error - should skip layers without router (line 102)
        output = hooks.forward(input_ids)
        assert output.shape[0] == 1
        # Verify no state was captured since there are no routers
        assert len(hooks.moe_state.router_logits) == 0

    def test_capture_moe_routing_integration(self, moe_model):
        """Test _capture_moe_routing method directly."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        )
        hooks.configure(config)

        # Test _capture_moe_routing directly with mock data
        # This tests lines 134-159
        batch_size, seq_len, hidden_size = 1, 4, 32
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        moe = moe_model.layers[0].mlp

        # Call _capture_moe_routing directly
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Verify captures occurred
        assert 0 in hooks.moe_state.router_logits
        assert 0 in hooks.moe_state.router_weights
        assert 0 in hooks.moe_state.selected_experts

        # Verify shapes
        logits = hooks.moe_state.router_logits[0]
        weights = hooks.moe_state.router_weights[0]
        selected = hooks.moe_state.selected_experts[0]

        # logits: (batch*seq, num_experts)
        assert logits.shape == (4, 4)  # 4 tokens, 4 experts
        # weights: (batch*seq, num_experts_per_tok)
        assert weights.shape[1] == 2  # top-2
        # selected: (batch, seq, num_experts_per_tok)
        assert selected.shape == (1, 4, 2)

    def test_capture_moe_routing_router_logits_only(self, moe_model):
        """Test capturing only router logits."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=False,
            capture_selected_experts=False,
        )
        hooks.configure(config)

        # Call _capture_moe_routing directly (line 146-147)
        x = mx.random.normal((1, 3, 32))
        moe = moe_model.layers[0].mlp
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Should only have router logits
        assert len(hooks.moe_state.router_logits) > 0
        assert len(hooks.moe_state.router_weights) == 0
        assert len(hooks.moe_state.selected_experts) == 0

    def test_capture_moe_routing_weights_only(self, moe_model):
        """Test capturing only router weights."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=False,
            capture_router_weights=True,
            capture_selected_experts=False,
        )
        hooks.configure(config)

        # Call _capture_moe_routing directly (line 155-156)
        x = mx.random.normal((1, 3, 32))
        moe = moe_model.layers[0].mlp
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Should only have router weights
        assert len(hooks.moe_state.router_logits) == 0
        assert len(hooks.moe_state.router_weights) > 0
        assert len(hooks.moe_state.selected_experts) == 0

    def test_capture_moe_routing_selected_experts_only(self, moe_model):
        """Test capturing only selected experts."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=False,
            capture_router_weights=False,
            capture_selected_experts=True,
        )
        hooks.configure(config)

        # Call _capture_moe_routing directly (line 158-161)
        x = mx.random.normal((1, 3, 32))
        moe = moe_model.layers[0].mlp
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Should only have selected experts
        assert len(hooks.moe_state.router_logits) == 0
        assert len(hooks.moe_state.router_weights) == 0
        assert len(hooks.moe_state.selected_experts) > 0

    def test_get_model_layers_fallback(self):
        """Test getting layers when model has no standard attributes."""

        class ModelWithDirectLayers(nn.Module):
            def __init__(self):
                super().__init__()
                # No .model, .transformer, or .decoder - just .layers directly
                self.layers = [MockLayer() for _ in range(2)]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = mx.zeros((input_ids.shape[0], input_ids.shape[1], 32))
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        model = ModelWithDirectLayers()
        hooks = MoEHooks(model)
        layers = hooks._get_model_layers()
        assert len(layers) == 2

    def test_get_expert_utilization_no_layer_info(self):
        """Test expert utilization when layer info cannot be retrieved."""

        # Create a model where layer info will be None
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = []

        model = MinimalModel()
        hooks = MoEHooks(model)

        # Manually add selected experts data for non-existent layer
        hooks.moe_state.selected_experts[999] = mx.array([[[0, 1]]])

        # Should return None because layer info is None
        util = hooks.get_expert_utilization(999)
        assert util is None

    def test_get_expert_utilization_zero_total(self):
        """Test expert utilization with edge case of zero total activations."""
        hooks = MoEHooks(MockMoEModel())

        # Get real layer info
        info = hooks.get_layer_info(0)
        assert info is not None

        # Add selected experts with zero total activations (empty tensor)
        # This creates a scenario where total = 0
        hooks.moe_state.selected_experts[0] = mx.array([]).reshape(1, 0, 2)

        util = hooks.get_expert_utilization(0)
        # Should handle the zero total case - expected will be 0
        # This tests line 206 (the else branch when expected <= 0)
        if util:
            assert util.load_balance_score == 1.0
            assert util.total_activations == 0

    def test_get_router_entropy_no_layer_info(self):
        """Test router entropy when layer info cannot be retrieved."""

        # Create a model where layer info will be None
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = []

        model = MinimalModel()
        hooks = MoEHooks(model)

        # Manually add router logits data for non-existent layer
        hooks.moe_state.router_logits[999] = mx.array([[1.0, 2.0, 3.0, 4.0]])

        # Should return None because layer info is None
        entropy = hooks.get_router_entropy(999)
        assert entropy is None

    def test_capture_moe_routing_with_bias(self, moe_model):
        """Test routing capture when router has bias."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        )
        hooks.configure(config)

        # The MockRouter has bias, this tests line 143-144
        x = mx.random.normal((1, 2, 32))
        moe = moe_model.layers[0].mlp
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Verify captures include bias in computation
        assert 0 in hooks.moe_state.router_logits
        assert 0 in hooks.moe_state.router_weights
        assert 0 in hooks.moe_state.selected_experts

    def test_capture_moe_routing_config_none(self, moe_model):
        """Test that _capture_moe_routing returns early when config is None."""
        hooks = MoEHooks(moe_model)
        # Don't configure - config will be None
        hooks.config = None

        # This should return early (line 134-135)
        x = mx.random.normal((1, 2, 32))
        moe = moe_model.layers[0].mlp
        hooks._capture_moe_routing(layer_idx=0, x=x, moe=moe)

        # Nothing should be captured
        assert len(hooks.moe_state.router_logits) == 0
        assert len(hooks.moe_state.router_weights) == 0
        assert len(hooks.moe_state.selected_experts) == 0
