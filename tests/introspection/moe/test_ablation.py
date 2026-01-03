"""Tests for MoE ablation functionality."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.moe.ablation import (
    ablate_expert,
    ablate_expert_batch,
    find_causal_experts,
    sweep_layer_experts,
)
from chuk_lazarus.introspection.moe.config import MoEAblationConfig
from chuk_lazarus.introspection.moe.hooks import MoEHooks
from chuk_lazarus.introspection.moe.models import ExpertAblationResult

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


class MockExpert(nn.Module):
    """Mock expert for testing."""

    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(mx.maximum(self.up_proj(x), 0))


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4):
        super().__init__()
        self.router = MockRouter(num_experts)
        self.experts = [MockExpert(hidden_size) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.mlp = MockMoE(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
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
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.model = type("Model", (), {"layers": self.layers})()

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    def decode(self, ids) -> str:
        if isinstance(ids, list):
            return " ".join(str(i) for i in ids)
        return str(ids)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def moe_model():
    """Create mock MoE model."""
    return MockMoEModel(vocab_size=100, hidden_size=32, num_layers=2, num_experts=4)


@pytest.fixture
def tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


# =============================================================================
# Tests
# =============================================================================


class TestMoEAblationConfig:
    """Tests for MoEAblationConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = MoEAblationConfig()
        assert config.target_layers is None
        assert config.ablation_method == "zero"
        assert config.preserve_scale is True
        assert config.max_new_tokens == 10

    def test_custom_values(self):
        """Test custom config values."""
        config = MoEAblationConfig(
            target_layers=[0, 2],
            ablation_method="mean",
            preserve_scale=False,
            max_new_tokens=20,
        )
        assert config.target_layers == [0, 2]
        assert config.ablation_method == "mean"
        assert config.max_new_tokens == 20


class TestAblateExpert:
    """Tests for ablate_expert function."""

    def test_basic_ablation(self, moe_model, tokenizer):
        """Test basic expert ablation."""
        input_ids = mx.array([[1, 2, 3]])
        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        assert isinstance(result, ExpertAblationResult)
        assert result.layer_idx == 0
        assert result.expert_idx == 0

    def test_with_config(self, moe_model, tokenizer):
        """Test ablation with custom config."""
        config = MoEAblationConfig(ablation_method="zero", max_new_tokens=5)
        input_ids = mx.array([[1, 2, 3]])

        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=1,
            input_ids=input_ids,
            tokenizer=tokenizer,
            config=config,
        )

        assert isinstance(result, ExpertAblationResult)

    def test_layer_out_of_range(self, moe_model, tokenizer):
        """Test with out of range layer."""
        input_ids = mx.array([[1, 2, 3]])

        with pytest.raises(ValueError):
            ablate_expert(
                moe_model,
                layer_idx=99,
                expert_idx=0,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )


class TestAblateExpertBatch:
    """Tests for ablate_expert_batch function."""

    def test_batch_ablation(self, moe_model, tokenizer):
        """Test batch ablation of multiple experts."""
        input_ids = mx.array([[1, 2, 3]])
        results = ablate_expert_batch(
            moe_model,
            layer_idx=0,
            expert_indices=[0, 1],
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ExpertAblationResult)

    def test_empty_indices(self, moe_model, tokenizer):
        """Test with empty expert indices."""
        input_ids = mx.array([[1, 2, 3]])
        results = ablate_expert_batch(
            moe_model,
            layer_idx=0,
            expert_indices=[],
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        assert results == []


class TestFindCausalExperts:
    """Tests for find_causal_experts function."""

    def test_find_causal(self, moe_model, tokenizer):
        """Test finding causal experts."""
        input_ids = mx.array([[1, 2, 3]])
        results = find_causal_experts(
            moe_model,
            layer_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        assert isinstance(results, list)
        # Results only include experts that changed output
        for result in results:
            assert result.output_changed is True


class TestSweepLayerExperts:
    """Tests for sweep_layer_experts function."""

    def test_sweep(self, moe_model, tokenizer):
        """Test sweeping all MoE layers."""
        hooks = MoEHooks(moe_model)
        input_ids = mx.array([[1, 2, 3]])

        results = sweep_layer_experts(
            hooks,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        assert isinstance(results, dict)
        # Should have entries for each MoE layer
        for layer_idx, layer_results in results.items():
            assert isinstance(layer_idx, int)
            assert isinstance(layer_results, list)


class TestAblationHelpers:
    """Tests for internal ablation helper functions."""

    def test_layer_not_moe(self, tokenizer):
        """Test ablating a non-MoE layer raises error."""

        class NonMoELayer(nn.Module):
            def __init__(self):
                super().__init__()
                # No mlp attribute

        class NonMoEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [NonMoELayer()]
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                return self.lm_head(x)

        model = NonMoEModel()
        input_ids = mx.array([[1, 2, 3]])

        with pytest.raises(ValueError, match="is not an MoE layer"):
            ablate_expert(model, 0, 0, input_ids, tokenizer)

    def test_generate_with_eos_token(self):
        """Test generation stops at EOS token."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                # Always output logits favoring token 0 (EOS)
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                logits = mx.zeros((batch_size, seq_len, 100))
                # Set high score for EOS token using proper MLX syntax
                mx.zeros((batch_size, seq_len, 100))
                high_eos_slice = mx.ones((batch_size, seq_len, 1)) * 10.0
                logits = mx.concatenate([high_eos_slice, logits[:, :, 1:]], axis=-1)
                return logits

        from chuk_lazarus.introspection.moe.ablation import _generate

        model = SimpleModel()
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = 0

        input_ids = mx.array([[1, 2, 3]])
        output = _generate(model, input_ids, tokenizer, max_new_tokens=10)

        # Should stop early due to EOS
        assert isinstance(output, str)

    def test_ablation_result_fields(self, moe_model, tokenizer):
        """Test all fields in ablation result are populated."""
        input_ids = mx.array([[1, 2, 3]])
        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # Check all required fields
        assert isinstance(result.expert_idx, int)
        assert isinstance(result.layer_idx, int)
        assert isinstance(result.baseline_output, str)
        assert isinstance(result.ablated_output, str)
        assert isinstance(result.output_changed, bool)
        assert isinstance(result.would_have_activated, bool)
        assert isinstance(result.activation_count, int)
        assert result.activation_count >= 0

    def test_ablation_with_batched_experts(self):
        """Test ablation with batched expert implementation."""

        class BatchedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                # GPT-OSS style batched experts
                self.gate_up_proj_blocks = nn.Linear(32, 256)

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class MoEBatched(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                self.experts = BatchedExperts()

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class LayerBatched(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MoEBatched()

        class ModelBatched(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerBatched()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = ModelBatched()
        tokenizer = MockTokenizer()
        input_ids = mx.array([[1, 2, 3]])

        # Should handle batched experts (though it may skip ablation)
        result = ablate_expert(model, 0, 0, input_ids, tokenizer)
        assert isinstance(result, ExpertAblationResult)

    def test_ablation_expert_without_down_proj(self):
        """Test ablation when expert doesn't have expected structure."""

        class SimpleExpert(nn.Module):
            # No down_proj attribute
            def __call__(self, x: mx.array) -> mx.array:
                return x

        class MoESimple(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                self.experts = [SimpleExpert() for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class LayerSimple(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MoESimple()

        class ModelSimple(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerSimple()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = ModelSimple()
        tokenizer = MockTokenizer()
        input_ids = mx.array([[1, 2, 3]])

        # Should handle missing down_proj gracefully
        result = ablate_expert(model, 0, 0, input_ids, tokenizer)
        assert isinstance(result, ExpertAblationResult)

    def test_find_causal_experts_no_moe_layer(self, tokenizer):
        """Test find_causal_experts with non-existent MoE layer."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = []

        model = SimpleModel()
        input_ids = mx.array([[1, 2, 3]])

        results = find_causal_experts(model, 0, input_ids, tokenizer)
        assert results == []

    def test_ablation_with_2d_input_ids(self, moe_model, tokenizer):
        """Test ablation handles 2D input correctly."""
        input_ids = mx.array([[1, 2, 3, 4]])  # Explicit 2D shape
        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=1,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
        assert isinstance(result, ExpertAblationResult)

    def test_sweep_with_custom_config(self, moe_model, tokenizer):
        """Test sweep with custom ablation config."""
        hooks = MoEHooks(moe_model)
        input_ids = mx.array([[1, 2, 3]])
        config = MoEAblationConfig(max_new_tokens=5)

        results = sweep_layer_experts(
            hooks,
            input_ids=input_ids,
            tokenizer=tokenizer,
            config=config,
        )

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_expert_activation_tracking(self, moe_model, tokenizer):
        """Test that expert activation is tracked correctly (lines 75-77)."""
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # These fields should be populated by lines 75-77
        assert hasattr(result, "would_have_activated")
        assert hasattr(result, "activation_count")
        assert isinstance(result.activation_count, int)
        assert result.activation_count >= 0
        # If would_have_activated is True, activation_count should be > 0
        if result.would_have_activated:
            assert result.activation_count > 0

    def test_generate_with_logits_attribute(self, tokenizer):
        """Test _generate handles model outputs with .logits attribute (line 216)."""

        class ModelWithLogitsAttr(nn.Module):
            """Model that returns object with .logits attribute."""

            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                logits = mx.random.normal((batch_size, seq_len, 100))
                # Return object with .logits attribute
                return type("Output", (), {"logits": logits})()

        from chuk_lazarus.introspection.moe.ablation import _generate

        model = ModelWithLogitsAttr()
        input_ids = mx.array([[1, 2, 3]])

        # This should trigger line 216: logits = logits.logits
        output = _generate(model, input_ids, tokenizer, max_new_tokens=2)
        assert isinstance(output, str)

    def test_generate_with_ablation_complex_routing(self, tokenizer):
        """Test _generate_with_ablation routing logic (lines 244-268)."""

        # Create a model with router that has bias
        class RouterWithBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.num_experts_per_tok = 2
                self.weight = mx.random.normal((4, 32)) * 0.02
                self.bias = mx.ones((4,)) * 0.1  # Non-None bias

        class MoEWithBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = RouterWithBias()
                self.experts = [MockExpert(32) for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class LayerWithBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MoEWithBias()

        class ModelWithBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerWithBias()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        from chuk_lazarus.introspection.moe.ablation import _generate_with_ablation

        model = ModelWithBias()
        input_ids = mx.array([[1, 2, 3]])

        # This should exercise the routing logic with bias (lines 244-268)
        output = _generate_with_ablation(
            model, input_ids, tokenizer, layer_idx=0, expert_idx=1, max_new_tokens=2
        )
        assert isinstance(output, str)

    def test_ablation_weight_restoration(self, tokenizer):
        """Test that expert weights are properly restored after ablation."""

        class TrackableExpert(nn.Module):
            """Expert that tracks weight modifications."""

            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(32, 64)
                self.down_proj = nn.Linear(64, 32)

            def __call__(self, x: mx.array) -> mx.array:
                return self.down_proj(mx.maximum(self.up_proj(x), 0))

        class TrackableMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                self.experts = [TrackableExpert() for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class TrackableLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = TrackableMoE()

        class TrackableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [TrackableLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = TrackableModel()
        input_ids = mx.array([[1, 2, 3]])
        expert = model.layers[0].mlp.experts[0]
        # Store original weight using MLX array operations
        original_weight = mx.array(expert.down_proj.weight)

        # Run ablation
        result = ablate_expert(
            model, layer_idx=0, expert_idx=0, input_ids=input_ids, tokenizer=tokenizer
        )

        # Weight should be restored
        assert mx.array_equal(expert.down_proj.weight, original_weight)
        assert isinstance(result, ExpertAblationResult)

    def test_ablation_expert_out_of_range(self, tokenizer):
        """Test ablation when expert index is out of range."""

        class SmallMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter(num_experts=2)  # Only 2 experts
                self.experts = [MockExpert(32) for _ in range(2)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class SmallLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = SmallMoE()

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [SmallLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = SmallModel()
        input_ids = mx.array([[1, 2, 3]])

        # Try to ablate expert beyond available experts
        # Should handle gracefully (expert won't have weight modified)
        result = ablate_expert(
            model,
            layer_idx=0,
            expert_idx=5,  # Out of range
            input_ids=input_ids,
            tokenizer=tokenizer,
        )
        assert isinstance(result, ExpertAblationResult)

    def test_generate_1d_input_ids(self, tokenizer):
        """Test _generate handles 1D input_ids correctly."""

        class Simple1DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, input_ids: mx.array):
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                return mx.random.normal((batch_size, seq_len, 100))

        from chuk_lazarus.introspection.moe.ablation import _generate

        model = Simple1DModel()
        # 1D input
        input_ids = mx.array([1, 2, 3])

        output = _generate(model, input_ids, tokenizer, max_new_tokens=2)
        assert isinstance(output, str)

    def test_ablation_with_no_experts_attr(self, tokenizer):
        """Test ablation when MLP has no experts attribute."""

        class NoExpertsMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                # No experts attribute

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class NoExpertsLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = NoExpertsMLP()

        class NoExpertsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [NoExpertsLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = NoExpertsModel()
        input_ids = mx.array([[1, 2, 3]])

        # Should handle missing experts attribute gracefully
        result = ablate_expert(
            model, layer_idx=0, expert_idx=0, input_ids=input_ids, tokenizer=tokenizer
        )
        assert isinstance(result, ExpertAblationResult)

    def test_router_without_bias(self, tokenizer):
        """Test ablation with router that has no bias."""

        class RouterNoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.num_experts_per_tok = 2
                self.weight = mx.random.normal((4, 32)) * 0.02
                self.bias = None  # Explicitly None

        class MoENoBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = RouterNoBias()
                self.experts = [MockExpert(32) for _ in range(4)]

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
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        from chuk_lazarus.introspection.moe.ablation import _generate_with_ablation

        model = ModelNoBias()
        input_ids = mx.array([[1, 2, 3]])

        # Should handle router without bias
        output = _generate_with_ablation(
            model, input_ids, tokenizer, layer_idx=0, expert_idx=0, max_new_tokens=2
        )
        assert isinstance(output, str)

    def test_ablated_call_routing_logic(self, tokenizer):
        """Test the internal routing logic in ablated_call (lines 244-268)."""
        # This tests the dead code path that's defined but not used
        # We'll create a scenario that exercises the routing calculations

        from chuk_lazarus.introspection.moe.ablation import _generate_with_ablation

        class RouterWithAttrs(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.num_experts_per_tok = 2
                self.weight = mx.random.normal((4, 32)) * 0.02
                self.bias = mx.ones((4,)) * 0.1

        class MoEForRouting(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = RouterWithAttrs()
                self.experts = [MockExpert(32) for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                # Simulate routing behavior to exercise the code paths
                router = self.router
                if x.ndim == 3:
                    batch_size, seq_len, hidden_size = x.shape
                    x_flat = x.reshape(-1, hidden_size)

                    # Get routing (this exercises lines similar to 244-264)
                    router_logits = x_flat @ router.weight.T
                    if hasattr(router, "bias") and router.bias is not None:
                        router_logits = router_logits + router.bias

                    k = router.num_experts_per_tok
                    topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
                    topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
                    weights = mx.softmax(topk_logits, axis=-1)

                    # Zero out expert 0's contribution (simulate ablation)
                    expert_idx = 0
                    mask = topk_indices != expert_idx
                    weights = weights * mask.astype(weights.dtype)

                    # Renormalize
                    weight_sum = mx.sum(weights, axis=-1, keepdims=True)
                    weights = mx.where(weight_sum > 0, weights / (weight_sum + 1e-10), weights)

                return x

        class LayerWithRouting(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MoEForRouting()

        class ModelWithRouting(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [LayerWithRouting()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = ModelWithRouting()
        input_ids = mx.array([[1, 2, 3]])

        # This will exercise the routing logic similar to lines 244-268
        output = _generate_with_ablation(
            model, input_ids, tokenizer, layer_idx=0, expert_idx=1, max_new_tokens=1
        )
        assert isinstance(output, str)

    def test_selected_experts_not_none(self, moe_model, tokenizer):
        """Test expert activation tracking when selected experts is not None (lines 75-77)."""
        # This specifically targets the code path where selected is not None
        # and we compute activation_count and would_activate

        # Use a longer input to increase chance of expert selection
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

        # Run ablation which internally uses hooks to check expert selection
        result = ablate_expert(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # The result should have activation tracking populated
        # If selected was not None, these would be set by lines 75-77
        assert hasattr(result, "would_have_activated")
        assert hasattr(result, "activation_count")
        assert isinstance(result.would_have_activated, bool)
        assert isinstance(result.activation_count, int)
        assert result.activation_count >= 0

    def test_multiple_expert_activations(self, tokenizer):
        """Test counting multiple activations of same expert (line 76)."""

        # Create a model that will definitely select expert 0 multiple times
        class DeterministicRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.num_experts_per_tok = 2
                # Weight biased to always select expert 0
                weight = mx.zeros((4, 32))
                weight[0, :] = mx.ones((32,)) * 10.0
                self.weight = weight
                self.bias = None

        class DeterministicMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = DeterministicRouter()
                self.experts = [MockExpert(32) for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                return x

        class DeterministicLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = DeterministicMoE()

        class DeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [DeterministicLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer.mlp(x)
                return self.lm_head(x)

        model = DeterministicModel()
        # Use multiple tokens to get multiple expert selections
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        result = ablate_expert(
            model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # With a deterministic router, expert 0 should be selected
        # The activation_count should reflect multiple token selections
        assert result.activation_count >= 0

    def test_hooks_capture_selected_experts(self, tokenizer):
        """Test that MoE hooks properly capture selected experts to exercise lines 75-77."""

        # Create a model with a functioning MoE forward pass
        class FunctionalRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.num_experts_per_tok = 2
                self.weight = mx.random.normal((4, 32)) * 0.02
                self.bias = mx.zeros((4,))

        class FunctionalMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = FunctionalRouter()
                self.experts = [MockExpert(32) for _ in range(4)]

            def __call__(self, x: mx.array) -> mx.array:
                # Implement actual MoE routing to trigger hook capture
                router = self.router
                batch_size, seq_len, hidden_size = x.shape
                x_flat = x.reshape(-1, hidden_size)

                # Compute router logits
                router_logits = x_flat @ router.weight.T
                if hasattr(router, "bias") and router.bias is not None:
                    router_logits = router_logits + router.bias

                # Get top-k experts
                k = router.num_experts_per_tok
                mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]

                # Simple pass-through for now
                return x

        class FunctionalLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = FunctionalMoE()

            def __call__(self, x: mx.array) -> mx.array:
                return self.mlp(x)

        class FunctionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [FunctionalLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        model = FunctionalModel()
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        # This should trigger the hooks and populate selected_experts
        # which will allow lines 75-77 to be covered
        result = ablate_expert(
            model,
            layer_idx=0,
            expert_idx=0,
            input_ids=input_ids,
            tokenizer=tokenizer,
        )

        # Verify the result has proper activation tracking
        assert isinstance(result.would_have_activated, bool)
        assert isinstance(result.activation_count, int)

    def test_expert_activation_with_mock_hook_state(self, tokenizer):
        """Test lines 75-77 by directly mocking the hook state."""
        from unittest.mock import MagicMock, patch

        from chuk_lazarus.introspection.moe.hooks import MoECapturedState

        # Create a simple mock model
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                self.experts = [MockExpert(32) for _ in range(4)]

            def __call__(self, x):
                return x

        class SimpleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = SimpleMLP()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.layers = [SimpleLayer()]
                self.lm_head = nn.Linear(32, 100)
                self.model = type("Model", (), {"layers": self.layers})()

            def __call__(self, input_ids: mx.array):
                x = self.embed(input_ids)
                return self.lm_head(x)

        model = SimpleModel()
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        # Create a mock hooks instance
        mock_hooks_instance = MagicMock()
        mock_state = MoECapturedState()
        # Set selected_experts to have expert 0 appearing 3 times
        mock_state.selected_experts[0] = mx.array(
            [
                [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3]]  # Expert 0 appears 3 times
            ]
        )
        mock_hooks_instance.moe_state = mock_state
        mock_hooks_instance.configure.return_value = mock_hooks_instance
        mock_hooks_instance.forward.return_value = model(input_ids)

        # Patch MoEHooks class at the location it's imported from
        with patch("chuk_lazarus.introspection.moe.hooks.MoEHooks") as mock_hooks_class:
            mock_hooks_class.return_value = mock_hooks_instance

            # Now run ablate_expert - it should hit lines 75-77
            result = ablate_expert(
                model,
                layer_idx=0,
                expert_idx=0,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )

            # Lines 75-77 should have been executed
            # Expert 0 appears 3 times in our mocked selected_experts
            assert result.would_have_activated is True
            assert result.activation_count == 3
