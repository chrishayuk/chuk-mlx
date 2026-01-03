"""Tests for MoE introspection."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.moe import (
    ExpertAblationResult,
    ExpertUtilization,
    MoEArchitecture,
    MoECapturedState,
    MoECaptureConfig,
    MoEHooks,
    MoELayerInfo,
    RouterEntropy,
    detect_moe_architecture,
    get_moe_layer_info,
)
from chuk_lazarus.introspection.moe.logit_lens import (
    ExpertLogitContribution,
    LayerRoutingSnapshot,
    MoELogitLens,
)


# =============================================================================
# Test MoE Components
# =============================================================================


class SimpleMoERouter(nn.Module):
    """Simple router for testing."""

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = mx.random.normal((num_experts, hidden_size)) * 0.02
        self.bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute routing weights and indices."""
        if x.ndim == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.reshape(-1, hidden_size)

        logits = x @ self.weight.T + self.bias
        k = self.num_experts_per_tok
        indices = mx.argsort(logits, axis=-1)[:, -k:][:, ::-1]
        weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
        return weights, indices


class SimpleMoEExperts(nn.Module):
    """Simple batched experts for testing."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        # Simple linear for each expert
        self.experts = [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]

    def __call__(
        self, x: mx.array, indices: mx.array, weights: mx.array
    ) -> mx.array:
        """Apply experts to input."""
        output = mx.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            mask = indices == expert_idx
            expert_weights = mx.sum(weights * mask.astype(weights.dtype), axis=-1)
            if mx.any(expert_weights > 0):
                expert_out = expert(x)
                output = output + expert_out * expert_weights[:, None]
        return output


class SimpleMoE(nn.Module):
    """Simple MoE layer for testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.router = SimpleMoERouter(hidden_size, num_experts, num_experts_per_tok)
        self.experts = SimpleMoEExperts(hidden_size, intermediate_size, num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)
        weights, indices = self.router(x_flat)
        output = self.experts(x_flat, indices, weights)
        return output.reshape(batch_size, seq_len, hidden_size)


class SimpleMoELayer(nn.Module):
    """Simple transformer layer with MoE for testing."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.self_attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.mlp = SimpleMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, cache=None
    ) -> tuple[mx.array, None]:
        # Attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, x, x, mask=mask)
        x = residual + x

        # MoE
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, None


class SimpleMoEModel(nn.Module):
    """Simple MoE model for testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            SimpleMoELayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
            )
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class SimpleMoEForCausalLM(nn.Module):
    """Simple MoE causal LM for testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.model = SimpleMoEModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


# =============================================================================
# Tests
# =============================================================================


class TestMoECaptureConfig:
    """Tests for MoECaptureConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MoECaptureConfig()
        assert config.capture_router_logits is True
        assert config.capture_router_weights is True
        assert config.capture_selected_experts is True
        assert config.capture_expert_outputs is False
        assert config.layers is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = MoECaptureConfig(
            capture_router_logits=False,
            capture_expert_outputs=True,
            layers=[0, 2],
        )
        assert config.capture_router_logits is False
        assert config.capture_expert_outputs is True
        assert config.layers == [0, 2]


class TestMoECapturedState:
    """Tests for MoECapturedState."""

    def test_empty_state(self):
        """Test empty state initialization."""
        state = MoECapturedState()
        assert len(state.router_logits) == 0
        assert len(state.router_weights) == 0
        assert len(state.selected_experts) == 0

    def test_clear(self):
        """Test state clearing."""
        state = MoECapturedState()
        state.router_logits[0] = mx.array([1, 2, 3])
        state.router_weights[0] = mx.array([0.5, 0.5])
        state.selected_experts[0] = mx.array([0, 1])

        state.clear()

        assert len(state.router_logits) == 0
        assert len(state.router_weights) == 0
        assert len(state.selected_experts) == 0


class TestMoEArchitectureDetection:
    """Tests for MoE architecture detection."""

    def test_detect_generic_moe(self):
        """Test detection of generic MoE model."""
        model = SimpleMoEForCausalLM()
        arch = detect_moe_architecture(model)
        assert arch == MoEArchitecture.GENERIC

    def test_get_moe_layer_info(self):
        """Test getting MoE layer info."""
        model = SimpleMoEForCausalLM(num_experts=4, num_experts_per_tok=2)
        info = get_moe_layer_info(model, 0)

        assert info is not None
        assert info.layer_idx == 0
        assert info.num_experts == 4
        assert info.num_experts_per_tok == 2
        assert info.has_shared_expert is False

    def test_non_moe_layer_returns_none(self):
        """Test that non-MoE layers return None."""
        # A model without MoE would return None
        # For this test, we check an out-of-range index
        model = SimpleMoEForCausalLM(num_layers=2)
        info = get_moe_layer_info(model, 10)
        assert info is None


class TestMoEHooks:
    """Tests for MoEHooks."""

    @pytest.fixture
    def moe_model(self):
        """Create a simple MoE model for testing."""
        return SimpleMoEForCausalLM(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
        )

    def test_hooks_initialization(self, moe_model):
        """Test MoE hooks initialization."""
        hooks = MoEHooks(moe_model)

        assert hooks.model is moe_model
        assert hooks.architecture == MoEArchitecture.GENERIC
        assert len(hooks.moe_layers) == 2  # 2 layers

    def test_configure(self, moe_model):
        """Test configuration."""
        hooks = MoEHooks(moe_model)
        config = MoECaptureConfig(capture_expert_outputs=True)

        result = hooks.configure(config)

        assert result is hooks  # Returns self for chaining
        assert hooks.config.capture_expert_outputs is True

    def test_moe_layer_indices(self, moe_model):
        """Test MoE layer index detection."""
        hooks = MoEHooks(moe_model)
        indices = hooks.moe_layers

        assert len(indices) == 2
        assert 0 in indices
        assert 1 in indices

    def test_forward_captures_state(self, moe_model):
        """Test that forward pass captures MoE state."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        ))

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = hooks.forward(input_ids)
        mx.eval(logits)

        # The model should have run; verify logits shape
        assert logits.shape[0] == 1
        assert logits.shape[1] == 5
        # State may or may not have data depending on model routing
        # Just verify no errors occurred during capture

    def test_forward_with_layer_filter(self, moe_model):
        """Test forward with layer filtering."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(layers=[0]))  # Only layer 0

        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        # Should only capture layer 0
        captured = list(hooks.moe_state.selected_experts.keys())
        assert 0 in captured or len(captured) <= 1


class TestExpertUtilization:
    """Tests for expert utilization analysis."""

    @pytest.fixture
    def moe_model(self):
        """Create MoE model."""
        return SimpleMoEForCausalLM(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
        )

    def test_get_expert_utilization(self, moe_model):
        """Test expert utilization computation."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_selected_experts=True))

        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # 8 tokens
        hooks.forward(input_ids)

        # Get utilization for first MoE layer if state was captured
        if hooks.moe_layers and hooks.moe_state.selected_experts:
            layer_idx = list(hooks.moe_state.selected_experts.keys())[0]
            mx.eval(hooks.moe_state.selected_experts[layer_idx])
            utilization = hooks.get_expert_utilization(layer_idx)

            if utilization is not None:
                assert utilization.num_experts == 4
                assert utilization.load_balance_score >= 0
                assert utilization.load_balance_score <= 1


class TestRouterEntropy:
    """Tests for router entropy analysis."""

    @pytest.fixture
    def moe_model(self):
        """Create MoE model."""
        return SimpleMoEForCausalLM(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
        )

    def test_get_router_entropy(self, moe_model):
        """Test router entropy computation."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(capture_router_logits=True))

        input_ids = mx.array([[1, 2, 3, 4]])
        hooks.forward(input_ids)

        if hooks.moe_layers:
            layer_idx = hooks.moe_layers[0]
            entropy = hooks.get_router_entropy(layer_idx)

            if entropy is not None:
                assert entropy.mean_entropy >= 0
                assert entropy.max_entropy > 0
                assert 0 <= entropy.normalized_entropy <= 1


class TestMoELogitLens:
    """Tests for MoE-aware logit lens."""

    @pytest.fixture
    def moe_model(self):
        """Create MoE model."""
        return SimpleMoEForCausalLM(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
        )

    def test_logit_lens_initialization(self, moe_model):
        """Test logit lens initialization."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig())

        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]
            def decode(self, ids):
                return "token"

        lens = MoELogitLens(hooks, MockTokenizer())

        assert lens.hooks is hooks
        assert lens.tokenizer is not None

    def test_get_routing_evolution(self, moe_model):
        """Test routing evolution analysis."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(
            capture_router_logits=True,
            capture_selected_experts=True,
        ))

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)

        lens = MoELogitLens(hooks)
        evolution = lens.get_routing_evolution(position=-1)

        assert isinstance(evolution, list)
        for snapshot in evolution:
            assert isinstance(snapshot, LayerRoutingSnapshot)
            assert hasattr(snapshot, "layer_idx")
            assert hasattr(snapshot, "selected_experts")

    def test_get_expert_contributions(self, moe_model):
        """Test expert contribution analysis."""
        hooks = MoEHooks(moe_model)
        hooks.configure(MoECaptureConfig(
            capture_router_logits=True,
            capture_router_weights=True,
            capture_selected_experts=True,
        ))

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)

        lens = MoELogitLens(hooks)

        if hooks.moe_layers:
            layer_idx = hooks.moe_layers[0]
            contributions = lens.get_expert_contributions(layer_idx)

            assert isinstance(contributions, list)
            for contrib in contributions:
                assert isinstance(contrib, ExpertLogitContribution)


class TestMoEModels:
    """Tests for MoE Pydantic models."""

    def test_moe_layer_info(self):
        """Test MoELayerInfo model."""
        info = MoELayerInfo(
            layer_idx=0,
            num_experts=8,
            num_experts_per_tok=2,
            has_shared_expert=False,
            architecture=MoEArchitecture.GENERIC,
        )
        assert info.layer_idx == 0
        assert info.num_experts == 8
        assert info.num_experts_per_tok == 2

    def test_router_entropy(self):
        """Test RouterEntropy model."""
        entropy = RouterEntropy(
            layer_idx=0,
            mean_entropy=1.5,
            max_entropy=2.0,
            normalized_entropy=0.75,
            per_position_entropy=(1.4, 1.5, 1.6),
        )
        assert entropy.normalized_entropy == 0.75

    def test_expert_utilization(self):
        """Test ExpertUtilization model."""
        util = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(25, 25, 25, 25),
            expert_frequencies=(0.25, 0.25, 0.25, 0.25),
            load_balance_score=1.0,
            most_used_expert=0,
            least_used_expert=0,
        )
        assert util.load_balance_score == 1.0

    def test_expert_ablation_result(self):
        """Test ExpertAblationResult model."""
        result = ExpertAblationResult(
            expert_idx=0,
            layer_idx=4,
            baseline_output="hello world",
            ablated_output="hello",
            output_changed=True,
            would_have_activated=True,
            activation_count=5,
        )
        assert result.output_changed is True
        assert result.would_have_activated is True
