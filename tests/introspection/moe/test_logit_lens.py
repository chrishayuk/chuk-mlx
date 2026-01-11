"""Tests for MoE logit lens analysis."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.moe.config import MoECaptureConfig
from chuk_lazarus.introspection.moe.hooks import MoEHooks
from chuk_lazarus.introspection.moe.logit_lens import (
    ExpertLogitContribution,
    LayerRoutingSnapshot,
    MoELogitLens,
    analyze_expert_vocabulary,
)

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

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    def decode(self, ids) -> str:
        return "token"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def moe_model():
    """Create mock MoE model."""
    return MockMoEModel(vocab_size=100, hidden_size=32, num_layers=2, num_experts=4)


@pytest.fixture
def hooks_with_data(moe_model):
    """Create hooks with pre-populated test data."""
    hooks = MoEHooks(moe_model)
    hooks.configure(MoECaptureConfig())

    # Populate state
    hooks.moe_state.selected_experts[0] = mx.array(
        [
            [[0, 1], [0, 2], [1, 3], [0, 1], [0, 1]],
        ]
    )
    hooks.moe_state.selected_experts[1] = mx.array(
        [
            [[2, 3], [1, 2], [0, 3], [2, 3], [1, 2]],
        ]
    )
    hooks.moe_state.router_weights[0] = mx.array(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.6, 0.4],
        ]
    )
    hooks.moe_state.router_logits[0] = mx.array(
        [
            [1.0, 2.0, 0.5, 0.3],
            [1.5, 1.5, 1.0, 0.5],
            [0.5, 2.0, 1.0, 1.5],
            [1.0, 2.0, 0.5, 0.3],
            [1.0, 2.0, 0.5, 0.3],
        ]
    )

    return hooks


@pytest.fixture
def hooks_with_2d_experts(moe_model):
    """Create hooks with 2D expert selection (no batch dimension)."""
    hooks = MoEHooks(moe_model)
    hooks.configure(MoECaptureConfig())

    # Populate state with 2D arrays [seq, k]
    hooks.moe_state.selected_experts[0] = mx.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [0, 1],
            [0, 1],
        ]
    )
    # No router_weights to cover line 121
    return hooks


@pytest.fixture
def hooks_with_3d_no_weights(moe_model):
    """Create hooks with 3D expert selection but no weights."""
    hooks = MoEHooks(moe_model)
    hooks.configure(MoECaptureConfig())

    # 3D arrays [batch, seq, k] but no weights
    hooks.moe_state.selected_experts[0] = mx.array(
        [
            [[0, 1], [0, 2], [1, 3], [0, 1], [0, 1]],
        ]
    )
    # No router_weights to cover line 118
    return hooks


# =============================================================================
# Tests
# =============================================================================


class TestExpertLogitContribution:
    """Tests for ExpertLogitContribution model."""

    def test_creation(self):
        """Test model creation."""
        contrib = ExpertLogitContribution(
            layer_idx=0,
            expert_idx=2,
            top_tokens=("def", "class"),
            top_logits=(1.5, 1.2),
            top_token_ids=(100, 101),
            activation_weight=0.6,
        )
        assert contrib.layer_idx == 0
        assert contrib.expert_idx == 2
        assert contrib.activation_weight == 0.6

    def test_defaults(self):
        """Test default values."""
        contrib = ExpertLogitContribution(
            layer_idx=0,
            expert_idx=0,
            activation_weight=0.5,
        )
        assert contrib.top_tokens == ()
        assert contrib.top_logits == ()


class TestLayerRoutingSnapshot:
    """Tests for LayerRoutingSnapshot model."""

    def test_creation(self):
        """Test model creation."""
        snapshot = LayerRoutingSnapshot(
            layer_idx=4,
            selected_experts=(0, 2),
            expert_weights=(0.6, 0.4),
            router_entropy=1.5,
            top_token="hello",
            top_token_prob=0.9,
        )
        assert snapshot.layer_idx == 4
        assert snapshot.selected_experts == (0, 2)
        assert snapshot.router_entropy == 1.5

    def test_defaults(self):
        """Test default values."""
        snapshot = LayerRoutingSnapshot(
            layer_idx=0,
            router_entropy=1.0,
        )
        assert snapshot.selected_experts == ()
        assert snapshot.top_token == ""
        assert snapshot.top_token_prob == 0.0


class TestMoELogitLens:
    """Tests for MoELogitLens class."""

    def test_initialization(self, hooks_with_data):
        """Test logit lens initialization."""
        lens = MoELogitLens(hooks_with_data, MockTokenizer())

        assert lens.hooks is hooks_with_data
        assert lens.tokenizer is not None

    def test_initialization_no_tokenizer(self, hooks_with_data):
        """Test initialization without tokenizer."""
        lens = MoELogitLens(hooks_with_data)

        assert lens.hooks is hooks_with_data
        assert lens.tokenizer is None

    def test_get_expert_contributions(self, hooks_with_data):
        """Test getting expert contributions."""
        lens = MoELogitLens(hooks_with_data)
        contributions = lens.get_expert_contributions(layer_idx=0, position=-1)

        assert isinstance(contributions, list)
        for contrib in contributions:
            assert isinstance(contrib, ExpertLogitContribution)

    def test_get_expert_contributions_missing_layer(self, hooks_with_data):
        """Test with missing layer."""
        lens = MoELogitLens(hooks_with_data)
        contributions = lens.get_expert_contributions(layer_idx=99)

        assert contributions == []

    def test_get_expert_contributions_2d_experts(self, hooks_with_2d_experts):
        """Test getting expert contributions with 2D expert array (no batch)."""
        lens = MoELogitLens(hooks_with_2d_experts)
        contributions = lens.get_expert_contributions(layer_idx=0, position=-1)

        assert isinstance(contributions, list)
        # Should still work with 2D arrays
        for contrib in contributions:
            assert isinstance(contrib, ExpertLogitContribution)
            # With no weights, each expert gets uniform weight
            assert contrib.activation_weight == pytest.approx(0.5, rel=0.1)

    def test_get_expert_contributions_3d_no_weights(self, hooks_with_3d_no_weights):
        """Test getting expert contributions with 3D array but no weights."""
        lens = MoELogitLens(hooks_with_3d_no_weights)
        contributions = lens.get_expert_contributions(layer_idx=0, position=-1)

        assert isinstance(contributions, list)
        for contrib in contributions:
            assert isinstance(contrib, ExpertLogitContribution)
            # With no weights, each expert gets uniform weight
            assert contrib.activation_weight == pytest.approx(0.5, rel=0.1)

    def test_get_routing_evolution(self, hooks_with_data):
        """Test getting routing evolution."""
        lens = MoELogitLens(hooks_with_data)
        evolution = lens.get_routing_evolution(position=-1)

        assert isinstance(evolution, list)
        assert len(evolution) == 2  # Two layers with data
        for snapshot in evolution:
            assert isinstance(snapshot, LayerRoutingSnapshot)

    def test_get_routing_evolution_empty(self, moe_model):
        """Test routing evolution with no data."""
        hooks = MoEHooks(moe_model)
        lens = MoELogitLens(hooks)
        evolution = lens.get_routing_evolution()

        assert evolution == []

    def test_get_routing_evolution_2d_experts(self, hooks_with_2d_experts):
        """Test routing evolution with 2D expert arrays."""
        lens = MoELogitLens(hooks_with_2d_experts)
        evolution = lens.get_routing_evolution(position=-1)

        assert isinstance(evolution, list)
        assert len(evolution) == 1  # One layer with data
        for snapshot in evolution:
            assert isinstance(snapshot, LayerRoutingSnapshot)

    def test_find_routing_divergence(self, hooks_with_data):
        """Test finding routing divergence."""
        lens = MoELogitLens(hooks_with_data)
        divergences = lens.find_routing_divergence(position=-1)

        assert isinstance(divergences, list)
        for div in divergences:
            assert isinstance(div, tuple)
            assert len(div) == 3  # (layer_a, layer_b, diff_set)

    def test_print_routing_evolution(self, hooks_with_data, capsys):
        """Test printing routing evolution."""
        lens = MoELogitLens(hooks_with_data)
        lens.print_routing_evolution(position=-1)

        captured = capsys.readouterr()
        assert "Routing Evolution" in captured.out or "Layer" in captured.out

    def test_print_routing_evolution_empty(self, moe_model, capsys):
        """Test printing with no data."""
        hooks = MoEHooks(moe_model)
        lens = MoELogitLens(hooks)
        lens.print_routing_evolution()

        captured = capsys.readouterr()
        assert "No routing data" in captured.out


class TestAnalyzeExpertVocabulary:
    """Tests for analyze_expert_vocabulary function."""

    def test_basic_analysis(self, moe_model):
        """Test basic vocabulary analysis."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=0,
            expert_idx=0,
            tokenizer=MockTokenizer(),
            top_k=10,
        )

        assert isinstance(result, dict)
        assert "expert_idx" in result
        assert result["expert_idx"] == 0

    def test_layer_out_of_range(self, moe_model):
        """Test with out of range layer."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=99,
            expert_idx=0,
            tokenizer=MockTokenizer(),
        )

        assert "error" in result

    def test_expert_out_of_range(self, moe_model):
        """Test with out of range expert."""
        result = analyze_expert_vocabulary(
            moe_model,
            layer_idx=0,
            expert_idx=99,
            tokenizer=MockTokenizer(),
        )

        assert "error" in result

    def test_layer_without_mlp(self):
        """Test with layer that has no MLP."""

        class NoMlpLayer(nn.Module):
            def __call__(self, x):
                return x

        class NoMlpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [NoMlpLayer()]

        model = NoMlpModel()
        result = analyze_expert_vocabulary(
            model,
            layer_idx=0,
            expert_idx=0,
            tokenizer=MockTokenizer(),
        )

        assert result.get("error") == "no mlp"

    def test_mlp_without_experts_list(self):
        """Test with MLP that has no experts list."""

        class MlpNoExperts(nn.Module):
            def __call__(self, x):
                return x

        class LayerWithMlpNoExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MlpNoExperts()

        class ModelMlpNoExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [LayerWithMlpNoExperts()]

        model = ModelMlpNoExperts()
        result = analyze_expert_vocabulary(
            model,
            layer_idx=0,
            expert_idx=0,
            tokenizer=MockTokenizer(),
        )

        assert result.get("error") == "no experts list"

    def test_expert_without_down_proj(self):
        """Test with expert that has no down_proj."""

        class ExpertNoDownProj(nn.Module):
            def __call__(self, x):
                return x

        class MlpWithBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = [ExpertNoDownProj()]

        class LayerWithBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MlpWithBadExperts()

        class ModelWithBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [LayerWithBadExperts()]

        model = ModelWithBadExperts()
        result = analyze_expert_vocabulary(
            model,
            layer_idx=0,
            expert_idx=0,
            tokenizer=MockTokenizer(),
        )

        assert result.get("error") == "no down_proj"


# =============================================================================
# Tests for functions with missing coverage
# =============================================================================


class MockMoEModelWithModel(nn.Module):
    """Mock MoE model with model.layers structure."""

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


class TestComputeExpertVocabContribution:
    """Tests for compute_expert_vocab_contribution function."""

    def test_basic_computation(self):
        """Test basic vocab contribution computation."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        model = MockMoEModelWithModel(vocab_size=50, num_experts=4)
        result = compute_expert_vocab_contribution(
            model, tokenizer=MockTokenizer(), layer_idx=0, top_k=5
        )

        assert result.layer_idx == 0
        assert result.num_experts >= 0
        assert isinstance(result.expert_contributions, tuple)

    def test_layer_out_of_range(self):
        """Test with invalid layer index."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        model = MockMoEModelWithModel()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=100)

        assert result.layer_idx == 100
        assert len(result.expert_contributions) == 0

    def test_layer_without_mlp(self):
        """Test with layer that has no MLP."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class NoMlpLayer(nn.Module):
            def __call__(self, x):
                return x

        class NoMlpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [NoMlpLayer()]})()
                self.lm_head = nn.Linear(32, 50)

        model = NoMlpModel()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=0)

        assert len(result.expert_contributions) == 0

    def test_mlp_without_experts(self):
        """Test with MLP that has no experts."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class MlpWithoutExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def __call__(self, x):
                return x

        class LayerWithMlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MlpWithoutExperts()

            def __call__(self, x):
                return x

        class ModelWithMlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [LayerWithMlp()]})()
                self.lm_head = nn.Linear(32, 50)

        model = ModelWithMlp()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=0)

        assert len(result.expert_contributions) == 0

    def test_with_vocab_sample_size(self):
        """Test with vocabulary sampling."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        model = MockMoEModelWithModel(vocab_size=100, num_experts=4)
        result = compute_expert_vocab_contribution(
            model, tokenizer=MockTokenizer(), layer_idx=0, vocab_sample_size=20
        )

        assert result.layer_idx == 0
        # Should still work with sampled vocabulary

    def test_expert_without_down_proj(self):
        """Test with expert that has no down_proj (line 480)."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class ExpertNoDownProj(nn.Module):
            def __init__(self):
                super().__init__()
                # Has up_proj but no down_proj
                self.up_proj = nn.Linear(32, 64)

            def __call__(self, x):
                return x

        class MlpWithPartialExperts(nn.Module):
            def __init__(self):
                super().__init__()
                # Mix of good experts and experts without down_proj
                self.experts = [MockExpert(32), ExpertNoDownProj(), MockExpert(32)]

        class LayerWithPartialExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MlpWithPartialExperts()

        class ModelWithPartialExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [LayerWithPartialExperts()]})()
                self.lm_head = nn.Linear(32, 50)

        model = ModelWithPartialExperts()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=0)

        # Should still work, just skip the expert without down_proj
        assert result.layer_idx == 0
        # Should have 2 expert contributions (skipping the one without down_proj)
        assert len(result.expert_contributions) == 2

    def test_tokenizer_decode_failure(self):
        """Test when tokenizer.decode fails (lines 500-501)."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class FailingTokenizer:
            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                raise ValueError("Decode failed")

        model = MockMoEModelWithModel(vocab_size=50, num_experts=2)
        result = compute_expert_vocab_contribution(
            model, tokenizer=FailingTokenizer(), layer_idx=0, top_k=3
        )

        # Should handle decode failure gracefully
        assert result.layer_idx == 0
        # Token names should be fallback format like "[token_id]"
        for contrib in result.expert_contributions:
            for token in contrib.top_tokens:
                assert token.startswith("[") and token.endswith("]")

    def test_model_without_lm_head(self):
        """Test model without lm_head."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class NoLmHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [MockLayer()]})()

            def __call__(self, x):
                return x

        model = NoLmHeadModel()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=0)

        assert len(result.expert_contributions) == 0

    def test_all_experts_without_down_proj(self):
        """Test when all experts lack down_proj (lines 548-549)."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_expert_vocab_contribution,
        )

        class ExpertNoDownProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(32, 64)

            def __call__(self, x):
                return x

        class MlpAllBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = [ExpertNoDownProj(), ExpertNoDownProj()]

        class LayerAllBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MlpAllBadExperts()

        class ModelAllBadExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [LayerAllBadExperts()]})()
                self.lm_head = nn.Linear(32, 50)

        model = ModelAllBadExperts()
        result = compute_expert_vocab_contribution(model, tokenizer=MockTokenizer(), layer_idx=0)

        # All experts skipped -> empty contributions, coverage and overlap = 0.0
        assert result.layer_idx == 0
        assert len(result.expert_contributions) == 0
        assert result.vocab_coverage == 0.0
        assert result.expert_overlap == 0.0


class TestComputeTokenExpertMapping:
    """Tests for compute_token_expert_mapping function."""

    def test_invalid_layer(self):
        """Test with invalid layer."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_token_expert_mapping,
        )

        model = MockMoEModelWithModel()
        result = compute_token_expert_mapping(
            model, tokenizer=MockTokenizer(), layer_idx=100, tokens_to_analyze=["test"]
        )

        assert result.layer_idx == 100
        assert len(result.token_preferences) == 0

    def test_layer_without_mlp(self):
        """Test with layer that has no MLP."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_token_expert_mapping,
        )

        class NoMlpLayer(nn.Module):
            def __call__(self, x):
                return x

        class NoMlpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [NoMlpLayer()]})()
                self.lm_head = nn.Linear(32, 50)

        model = NoMlpModel()
        result = compute_token_expert_mapping(
            model, tokenizer=MockTokenizer(), layer_idx=0, tokens_to_analyze=["test"]
        )

        assert len(result.token_preferences) == 0

    def test_without_lm_head(self):
        """Test model without lm_head returns empty preferences."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_token_expert_mapping,
        )

        class NoLmHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [MockLayer()]})()

            def __call__(self, x):
                return x

        model = NoLmHeadModel()
        result = compute_token_expert_mapping(
            model, tokenizer=MockTokenizer(), layer_idx=0, tokens_to_analyze=["test"]
        )

        assert len(result.token_preferences) == 0

    def test_tokenizer_encode_failure(self):
        """Test when tokenizer.encode fails (lines 663-664)."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_token_expert_mapping,
        )

        class FailingTokenizer:
            def encode(self, text):
                raise ValueError("Encode failed")

            def decode(self, ids):
                return "token"

        model = MockMoEModelWithModel()
        result = compute_token_expert_mapping(
            model, tokenizer=FailingTokenizer(), layer_idx=0, tokens_to_analyze=["test"]
        )

        # All encodes failed, so no tokens mapped
        assert result.num_tokens == 0
        assert len(result.token_preferences) == 0

    def test_tokenizer_encode_returns_empty(self):
        """Test when tokenizer.encode returns empty list (line 667)."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            compute_token_expert_mapping,
        )

        class EmptyTokenizer:
            def encode(self, text):
                return []

            def decode(self, ids):
                return "token"

        model = MockMoEModelWithModel()
        result = compute_token_expert_mapping(
            model, tokenizer=EmptyTokenizer(), layer_idx=0, tokens_to_analyze=["test"]
        )

        # No tokens could be encoded
        assert result.num_tokens == 0
        assert len(result.token_preferences) == 0


class TestFindExpertSpecialists:
    """Tests for find_expert_specialists function."""

    def test_basic_specialist_finding(self):
        """Test finding specialist experts."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
            find_expert_specialists,
        )

        # Create mock analysis with specialist experts
        contributions = [
            ExpertVocabContribution(
                expert_idx=0,
                layer_idx=0,
                top_tokens=("cat", "dog", "pet"),
                top_scores=(0.9, 0.8, 0.7),
                top_token_ids=(10, 20, 30),
                vocab_entropy=0.5,
                specialization_score=0.9,  # High specialization
            ),
            ExpertVocabContribution(
                expert_idx=1,
                layer_idx=0,
                top_tokens=("a", "the", "is"),
                top_scores=(0.3, 0.3, 0.3),
                top_token_ids=(1, 2, 3),
                vocab_entropy=0.95,
                specialization_score=0.1,  # Low specialization
            ),
        ]

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=2,
            expert_contributions=tuple(contributions),
        )

        specialists = find_expert_specialists(analysis, min_specialization=0.5)

        # Should find expert 0 as specialist
        assert len(specialists) >= 0

    def test_no_specialists(self):
        """Test when no specialists exist."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
            find_expert_specialists,
        )

        # All generalists
        contributions = [
            ExpertVocabContribution(
                expert_idx=0,
                layer_idx=0,
                top_tokens=("a", "the"),
                top_scores=(0.5, 0.5),
                top_token_ids=(1, 2),
                vocab_entropy=0.95,
                specialization_score=0.1,
            ),
        ]

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=1,
            expert_contributions=tuple(contributions),
        )

        specialists = find_expert_specialists(analysis, min_specialization=0.9)

        assert len(specialists) == 0


class TestPrintExpertVocabSummary:
    """Tests for print_expert_vocab_summary function."""

    def test_basic_print(self, capsys):
        """Test basic summary printing."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
            print_expert_vocab_summary,
        )

        contributions = [
            ExpertVocabContribution(
                expert_idx=0,
                layer_idx=0,
                top_tokens=("cat", "dog"),
                top_scores=(0.9, 0.8),
                top_token_ids=(10, 20),
                vocab_entropy=0.5,
                specialization_score=0.8,
            ),
        ]

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=1,
            expert_contributions=tuple(contributions),
        )

        print_expert_vocab_summary(analysis)

        captured = capsys.readouterr()
        assert "Expert Vocabulary Contributions" in captured.out

    def test_empty_analysis(self, capsys):
        """Test printing empty analysis."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            LayerVocabAnalysis,
            print_expert_vocab_summary,
        )

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=4,
            expert_contributions=(),
        )

        print_expert_vocab_summary(analysis)

        captured = capsys.readouterr()
        # With no contributions, it still prints header with 0 coverage
        assert "Expert Vocabulary Contributions" in captured.out


class TestPrintTokenExpertPreferences:
    """Tests for print_token_expert_preferences function."""

    def test_basic_print(self, capsys):
        """Test basic preferences printing."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            TokenExpertPreference,
            VocabExpertMapping,
            print_token_expert_preferences,
        )

        preferences = [
            TokenExpertPreference(
                token="hello",
                token_id=10,
                preferred_expert=0,
                expert_scores=(0.8, 0.1, 0.1),
                category="word",
            ),
        ]

        mapping = VocabExpertMapping(
            layer_idx=0,
            num_experts=3,
            num_tokens=1,
            token_preferences=tuple(preferences),
        )

        print_token_expert_preferences(mapping)

        captured = capsys.readouterr()
        assert "Token-Expert" in captured.out or "Preferences" in captured.out

    def test_empty_preferences(self, capsys):
        """Test printing empty preferences."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            VocabExpertMapping,
            print_token_expert_preferences,
        )

        mapping = VocabExpertMapping(
            layer_idx=0,
            num_experts=3,
            num_tokens=0,
            token_preferences=(),
        )

        print_token_expert_preferences(mapping)

        captured = capsys.readouterr()
        # With no preferences, it shows "no dominant tokens" for each expert
        assert "Token-Expert Preferences" in captured.out
        assert "no dominant tokens" in captured.out


class TestGetModelLayers:
    """Tests for _get_model_layers helper function."""

    def test_with_model_layers(self):
        """Test with model.layers structure."""
        from chuk_lazarus.introspection.moe.logit_lens import _get_model_layers

        model = MockMoEModelWithModel()
        layers = _get_model_layers(model)

        assert len(layers) == 2

    def test_with_direct_layers(self):
        """Test with direct layers attribute."""
        from chuk_lazarus.introspection.moe.logit_lens import _get_model_layers

        model = MockMoEModel()
        layers = _get_model_layers(model)

        assert len(layers) == 2

    def test_with_no_layers(self):
        """Test with model that has no layers."""
        from chuk_lazarus.introspection.moe.logit_lens import _get_model_layers

        class NoLayersModel(nn.Module):
            def __call__(self, x):
                return x

        model = NoLayersModel()
        layers = _get_model_layers(model)

        assert layers == []


class TestGetLmHead:
    """Tests for _get_lm_head helper function."""

    def test_with_lm_head(self):
        """Test model with lm_head."""
        from chuk_lazarus.introspection.moe.logit_lens import _get_lm_head

        model = MockMoEModelWithModel()
        lm_head = _get_lm_head(model)

        assert lm_head is not None

    def test_without_lm_head(self):
        """Test model without lm_head."""
        from chuk_lazarus.introspection.moe.logit_lens import _get_lm_head

        class NoLmHeadModel(nn.Module):
            def __call__(self, x):
                return x

        model = NoLmHeadModel()
        lm_head = _get_lm_head(model)

        assert lm_head is None


class TestCategorizeTokens:
    """Tests for _categorize_tokens helper function."""

    def test_punctuation(self):
        """Test punctuation categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens([".", ",", "!"])

        assert all(c == "punctuation" for c in categories)

    def test_numbers(self):
        """Test number categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["123", "0", "456"])

        # Returns top categories - should include "numbers"
        assert "numbers" in categories

    def test_words(self):
        """Test word categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["hello", "world"])

        # Returns top categories - lowercase words
        assert "lowercase" in categories

    def test_mixed(self):
        """Test mixed token types."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["hello", "123", "."])

        # Returns top 3 categories sorted by count
        assert len(categories) <= 3
        # Each type appears once, so all should be present
        assert "lowercase" in categories
        assert "numbers" in categories
        assert "punctuation" in categories

    def test_whitespace(self):
        """Test whitespace categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["", " ", "  "])

        assert "whitespace" in categories

    def test_uppercase(self):
        """Test uppercase word categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["HELLO", "WORLD", "FOO"])

        assert "uppercase" in categories

    def test_mixed_case(self):
        """Test mixed case word categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["HelloWorld", "CamelCase", "MixedCase"])

        assert "mixed_case" in categories

    def test_operators(self):
        """Test operator categorization."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["+", "-", "*", "/"])

        assert "operators" in categories

    def test_mixed_characters(self):
        """Test tokens with mixed character types (not pure alpha/digit/punct)."""
        from chuk_lazarus.introspection.moe.logit_lens import _categorize_tokens

        categories = _categorize_tokens(["hello123", "a1b2c3", "abc_xyz"])

        assert "mixed" in categories


class TestComputeVocabScores:
    """Tests for _compute_vocab_scores helper function."""

    def test_basic_computation(self):
        """Test basic vocab score computation."""
        from chuk_lazarus.introspection.moe.logit_lens import _compute_vocab_scores

        down_weight = mx.random.normal((32, 64))
        lm_weight = mx.random.normal((100, 32))

        scores = _compute_vocab_scores(down_weight, lm_weight)

        assert scores.shape[0] == 100
        assert all(s >= 0 for s in scores.tolist())
