"""Tests for MoE Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.moe.enums import (
    ExpertCategory,
    ExpertRole,
    MoEArchitecture,
)
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    CompressionPlan,
    ExpertAblationResult,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertIdentity,
    ExpertPair,
    ExpertPattern,
    ExpertTaxonomy,
    ExpertUtilization,
    GenerationStats,
    LayerDivergenceResult,
    LayerRouterWeights,
    LayerRoutingAnalysis,
    MoELayerInfo,
    MoEModelInfo,
    RouterEntropy,
    RouterWeightCapture,
    TokenExpertMapping,
    TopKVariationResult,
    VocabExpertAnalysis,
)


class TestMoELayerInfo:
    """Tests for MoELayerInfo model."""

    def test_minimal_creation(self):
        """Test minimal creation with required fields."""
        info = MoELayerInfo(
            layer_idx=0,
            num_experts=8,
            num_experts_per_tok=2,
        )
        assert info.layer_idx == 0
        assert info.num_experts == 8
        assert info.num_experts_per_tok == 2
        assert info.has_shared_expert is False
        assert info.architecture == MoEArchitecture.GENERIC

    def test_full_creation(self):
        """Test creation with all fields."""
        info = MoELayerInfo(
            layer_idx=4,
            num_experts=32,
            num_experts_per_tok=4,
            has_shared_expert=True,
            architecture=MoEArchitecture.LLAMA4,
            router_type="linear",
            uses_softmax=True,
            uses_sigmoid=False,
        )
        assert info.layer_idx == 4
        assert info.num_experts == 32
        assert info.num_experts_per_tok == 4
        assert info.has_shared_expert is True
        assert info.architecture == MoEArchitecture.LLAMA4

    def test_frozen(self):
        """Test model is frozen."""
        info = MoELayerInfo(layer_idx=0, num_experts=8, num_experts_per_tok=2)
        with pytest.raises(Exception):
            info.layer_idx = 1


class TestRouterEntropy:
    """Tests for RouterEntropy model."""

    def test_minimal_creation(self):
        """Test minimal creation."""
        entropy = RouterEntropy(
            layer_idx=0,
            mean_entropy=1.5,
            max_entropy=2.0,
            normalized_entropy=0.75,
        )
        assert entropy.layer_idx == 0
        assert entropy.mean_entropy == 1.5
        assert entropy.normalized_entropy == 0.75
        assert entropy.per_position_entropy == ()

    def test_with_position_entropy(self):
        """Test with per-position entropy."""
        entropy = RouterEntropy(
            layer_idx=2,
            mean_entropy=1.0,
            max_entropy=2.0,
            normalized_entropy=0.5,
            per_position_entropy=(0.9, 1.0, 1.1),
        )
        assert len(entropy.per_position_entropy) == 3

    def test_normalized_entropy_bounds(self):
        """Test normalized entropy is between 0 and 1."""
        entropy = RouterEntropy(
            layer_idx=0,
            mean_entropy=0.5,
            max_entropy=1.0,
            normalized_entropy=0.5,
        )
        assert 0 <= entropy.normalized_entropy <= 1


class TestExpertUtilization:
    """Tests for ExpertUtilization model."""

    def test_full_creation(self):
        """Test full creation."""
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
        assert util.num_experts == 4
        assert util.total_activations == 100
        assert util.load_balance_score == 1.0

    def test_imbalanced_utilization(self):
        """Test imbalanced utilization."""
        util = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(40, 30, 20, 10),
            expert_frequencies=(0.4, 0.3, 0.2, 0.1),
            load_balance_score=0.7,
            most_used_expert=0,
            least_used_expert=3,
        )
        assert util.most_used_expert == 0
        assert util.least_used_expert == 3


class TestExpertIdentity:
    """Tests for ExpertIdentity model."""

    def test_minimal_creation(self):
        """Test minimal creation."""
        identity = ExpertIdentity(
            expert_idx=0,
            layer_idx=4,
            primary_category=ExpertCategory.CODE,
            confidence=0.9,
            activation_rate=0.5,
        )
        assert identity.expert_idx == 0
        assert identity.primary_category == ExpertCategory.CODE
        assert identity.role == ExpertRole.GENERALIST

    def test_full_creation(self):
        """Test full creation with all fields."""
        identity = ExpertIdentity(
            expert_idx=2,
            layer_idx=4,
            primary_category=ExpertCategory.MATH,
            secondary_categories=(ExpertCategory.NUMBERS, ExpertCategory.CODE),
            role=ExpertRole.SPECIALIST,
            confidence=0.95,
            activation_rate=0.3,
            top_tokens=("def", "class", "import"),
            description="Math and code specialist",
        )
        assert identity.role == ExpertRole.SPECIALIST
        assert len(identity.secondary_categories) == 2
        assert len(identity.top_tokens) == 3


class TestExpertPair:
    """Tests for ExpertPair model."""

    def test_creation(self):
        """Test pair creation."""
        pair = ExpertPair(
            expert_a=0,
            expert_b=3,
            coactivation_count=50,
            coactivation_rate=0.25,
        )
        assert pair.expert_a == 0
        assert pair.expert_b == 3
        assert pair.coactivation_count == 50
        assert pair.coactivation_rate == 0.25


class TestCoactivationAnalysis:
    """Tests for CoactivationAnalysis model."""

    def test_minimal_creation(self):
        """Test minimal creation."""
        analysis = CoactivationAnalysis(
            layer_idx=4,
            total_activations=1000,
        )
        assert analysis.layer_idx == 4
        assert analysis.top_pairs == ()

    def test_with_pairs(self):
        """Test with pair data."""
        pair = ExpertPair(
            expert_a=0,
            expert_b=1,
            coactivation_count=100,
            coactivation_rate=0.5,
        )
        analysis = CoactivationAnalysis(
            layer_idx=4,
            total_activations=1000,
            top_pairs=(pair,),
            generalist_experts=(2, 5),
        )
        assert len(analysis.top_pairs) == 1
        assert len(analysis.generalist_experts) == 2


class TestExpertAblationResult:
    """Tests for ExpertAblationResult model."""

    def test_creation(self):
        """Test result creation."""
        result = ExpertAblationResult(
            expert_idx=0,
            layer_idx=4,
            baseline_output="The quick brown fox",
            ablated_output="The quick brown",
            output_changed=True,
            would_have_activated=True,
            activation_count=5,
        )
        assert result.output_changed is True
        assert result.would_have_activated is True
        assert result.activation_count == 5

    def test_no_change(self):
        """Test when output doesn't change."""
        result = ExpertAblationResult(
            expert_idx=1,
            layer_idx=4,
            baseline_output="Hello world",
            ablated_output="Hello world",
            output_changed=False,
            would_have_activated=False,
            activation_count=0,
        )
        assert result.output_changed is False
        assert result.would_have_activated is False


class TestCompressionPlan:
    """Tests for CompressionPlan model."""

    def test_creation(self):
        """Test plan creation."""
        plan = CompressionPlan(
            source_num_experts=8,
            target_num_experts=4,
            merge_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
            estimated_quality_loss=0.05,
            estimated_size_reduction=0.5,
        )
        assert plan.source_num_experts == 8
        assert plan.target_num_experts == 4
        assert len(plan.merge_groups) == 4
        assert plan.estimated_quality_loss == 0.05

    def test_no_merge(self):
        """Test plan with no merging."""
        plan = CompressionPlan(
            source_num_experts=4,
            target_num_experts=4,
            merge_groups=(),
            estimated_quality_loss=0.0,
            estimated_size_reduction=0.0,
        )
        assert len(plan.merge_groups) == 0


class TestMoEModelInfo:
    """Tests for MoEModelInfo model."""

    def test_minimal_creation(self):
        """Test minimal creation."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=4,
        )
        assert info.num_experts == 8
        assert info.total_layers == 4
        assert info.architecture == MoEArchitecture.GENERIC

    def test_full_creation(self):
        """Test full creation with all fields."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
            has_shared_expert=True,
        )
        assert info.architecture == MoEArchitecture.GPT_OSS
        assert info.has_shared_expert is True

    def test_is_moe_property_true(self):
        """Test is_moe property when model has MoE layers."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=4,
        )
        assert info.is_moe is True

    def test_is_moe_property_false(self):
        """Test is_moe property when model has no MoE layers."""
        info = MoEModelInfo(
            moe_layers=(),
            num_experts=0,
            num_experts_per_tok=0,
            total_layers=4,
        )
        assert info.is_moe is False

    def test_frozen(self):
        """Test model is frozen."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=4,
        )
        with pytest.raises(ValidationError):
            info.num_experts = 16


class TestGenerationStats:
    """Tests for GenerationStats model."""

    def test_creation(self):
        """Test creation."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        assert stats.expert_idx == 6
        assert stats.tokens_generated == 20
        assert stats.layers_modified == 8
        assert stats.moe_type == "gpt_oss_batched"
        assert stats.prompt_tokens == 0

    def test_with_prompt_tokens(self):
        """Test with prompt tokens."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
            prompt_tokens=10,
        )
        assert stats.prompt_tokens == 10

    def test_normal_generation(self):
        """Test normal generation (expert_idx = -1)."""
        stats = GenerationStats(
            expert_idx=-1,
            tokens_generated=50,
            layers_modified=0,
            moe_type="gpt_oss_batched",
        )
        assert stats.expert_idx == -1


class TestExpertChatResult:
    """Tests for ExpertChatResult model."""

    def test_creation(self):
        """Test result creation."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        result = ExpertChatResult(
            prompt="127 * 89 = ",
            response="11303",
            expert_idx=6,
            stats=stats,
        )
        assert result.prompt == "127 * 89 = "
        assert result.response == "11303"
        assert result.expert_idx == 6
        assert result.layer_idx is None

    def test_with_layer_idx(self):
        """Test result with specific layer."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=1,
            moe_type="gpt_oss_batched",
        )
        result = ExpertChatResult(
            prompt="Test",
            response="Response",
            expert_idx=6,
            layer_idx=4,
            stats=stats,
        )
        assert result.layer_idx == 4


class TestExpertComparisonResult:
    """Tests for ExpertComparisonResult model."""

    def test_creation(self):
        """Test creation."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        result1 = ExpertChatResult(
            prompt="Test",
            response="Response1",
            expert_idx=6,
            stats=stats,
        )
        result2 = ExpertChatResult(
            prompt="Test",
            response="Response2",
            expert_idx=7,
            stats=GenerationStats(
                expert_idx=7,
                tokens_generated=25,
                layers_modified=8,
                moe_type="gpt_oss_batched",
            ),
        )
        comparison = ExpertComparisonResult(
            prompt="Test",
            expert_results=(result1, result2),
        )
        assert comparison.prompt == "Test"
        assert len(comparison.expert_results) == 2

    def test_get_result_for_expert_found(self):
        """Test getting result for specific expert."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        result = ExpertChatResult(
            prompt="Test",
            response="Response1",
            expert_idx=6,
            stats=stats,
        )
        comparison = ExpertComparisonResult(
            prompt="Test",
            expert_results=(result,),
        )
        found = comparison.get_result_for_expert(6)
        assert found is not None
        assert found.expert_idx == 6

    def test_get_result_for_expert_not_found(self):
        """Test getting result for non-existent expert."""
        comparison = ExpertComparisonResult(
            prompt="Test",
            expert_results=(),
        )
        assert comparison.get_result_for_expert(99) is None


class TestTopKVariationResult:
    """Tests for TopKVariationResult model."""

    def test_creation(self):
        """Test creation."""
        result = TopKVariationResult(
            prompt="Test prompt",
            k_value=2,
            default_k=4,
            response="Response with k=2",
            normal_response="Response with k=4",
        )
        assert result.k_value == 2
        assert result.default_k == 4
        assert result.response != result.normal_response


class TestRouterWeightCapture:
    """Tests for RouterWeightCapture model."""

    def test_creation(self):
        """Test creation."""
        capture = RouterWeightCapture(
            layer_idx=0,
            position_idx=5,
            token="Hello",
            expert_indices=(6, 7, 20, 1),
            weights=(0.4, 0.3, 0.2, 0.1),
        )
        assert capture.layer_idx == 0
        assert capture.position_idx == 5
        assert capture.token == "Hello"
        assert len(capture.expert_indices) == 4
        assert len(capture.weights) == 4

    def test_top_expert_property(self):
        """Test top_expert property."""
        capture = RouterWeightCapture(
            layer_idx=0,
            position_idx=0,
            token="Test",
            expert_indices=(15, 7, 3, 1),
            weights=(0.5, 0.3, 0.15, 0.05),
        )
        assert capture.top_expert == 15

    def test_top_expert_empty(self):
        """Test top_expert with no experts."""
        capture = RouterWeightCapture(
            layer_idx=0,
            position_idx=0,
            token="",
            expert_indices=(),
            weights=(),
        )
        assert capture.top_expert is None


class TestLayerRouterWeights:
    """Tests for LayerRouterWeights model."""

    def test_creation(self):
        """Test creation."""
        capture = RouterWeightCapture(
            layer_idx=0,
            position_idx=0,
            token="Hello",
            expert_indices=(6, 7),
            weights=(0.6, 0.4),
        )
        layer = LayerRouterWeights(
            layer_idx=0,
            positions=(capture,),
        )
        assert layer.layer_idx == 0
        assert len(layer.positions) == 1

    def test_multiple_positions(self):
        """Test with multiple positions."""
        captures = tuple(
            RouterWeightCapture(
                layer_idx=0,
                position_idx=i,
                token=f"tok_{i}",
                expert_indices=(i % 8,),
                weights=(1.0,),
            )
            for i in range(10)
        )
        layer = LayerRouterWeights(layer_idx=0, positions=captures)
        assert len(layer.positions) == 10


class TestLayerRoutingAnalysis:
    """Tests for LayerRoutingAnalysis model."""

    def test_creation(self):
        """Test creation."""
        entropy = RouterEntropy(
            layer_idx=0,
            mean_entropy=1.5,
            max_entropy=2.0,
            normalized_entropy=0.75,
        )
        util = ExpertUtilization(
            layer_idx=0,
            num_experts=8,
            total_activations=100,
            expert_counts=(12, 13, 12, 13, 12, 13, 12, 13),
            expert_frequencies=(0.12, 0.13, 0.12, 0.13, 0.12, 0.13, 0.12, 0.13),
            load_balance_score=0.98,
            most_used_expert=1,
            least_used_expert=0,
        )
        analysis = LayerRoutingAnalysis(
            layer_idx=0,
            entropy=entropy,
            utilization=util,
        )
        assert analysis.layer_idx == 0
        assert analysis.entropy.mean_entropy == 1.5
        assert analysis.coactivation is None

    def test_with_coactivation(self):
        """Test with coactivation analysis."""
        entropy = RouterEntropy(
            layer_idx=0,
            mean_entropy=1.0,
            max_entropy=1.5,
            normalized_entropy=0.67,
        )
        util = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=50,
            expert_counts=(12, 13, 12, 13),
            expert_frequencies=(0.24, 0.26, 0.24, 0.26),
            load_balance_score=0.95,
            most_used_expert=1,
            least_used_expert=0,
        )
        coact = CoactivationAnalysis(
            layer_idx=0,
            total_activations=50,
            generalist_experts=(0, 1),
        )
        analysis = LayerRoutingAnalysis(
            layer_idx=0,
            entropy=entropy,
            utilization=util,
            coactivation=coact,
        )
        assert analysis.coactivation is not None
        assert analysis.coactivation.generalist_experts == (0, 1)


class TestLayerDivergenceResult:
    """Tests for LayerDivergenceResult model."""

    def test_creation(self):
        """Test creation."""
        result = LayerDivergenceResult(
            layer_a=0,
            layer_b=7,
            divergence_score=0.45,
            shared_experts=(6, 7, 20),
            unique_to_a=(1, 2),
            unique_to_b=(25, 30),
        )
        assert result.layer_a == 0
        assert result.layer_b == 7
        assert result.divergence_score == 0.45
        assert len(result.shared_experts) == 3

    def test_no_divergence(self):
        """Test identical layers."""
        result = LayerDivergenceResult(
            layer_a=0,
            layer_b=1,
            divergence_score=0.0,
            shared_experts=(0, 1, 2, 3, 4, 5, 6, 7),
            unique_to_a=(),
            unique_to_b=(),
        )
        assert result.divergence_score == 0.0


class TestExpertPattern:
    """Tests for ExpertPattern model."""

    def test_creation(self):
        """Test creation."""
        pattern = ExpertPattern(
            expert_idx=6,
            layer_idx=0,
            pattern_type="numeric",
            trigger_tokens=("1", "2", "3", "127", "89"),
            confidence=0.92,
            sample_activations=150,
            description="Activates on numeric tokens",
        )
        assert pattern.expert_idx == 6
        assert pattern.pattern_type == "numeric"
        assert len(pattern.trigger_tokens) == 5
        assert pattern.confidence == 0.92

    def test_minimal_creation(self):
        """Test minimal creation."""
        pattern = ExpertPattern(
            expert_idx=0,
            layer_idx=0,
            pattern_type="unknown",
            confidence=0.5,
            sample_activations=10,
        )
        assert pattern.trigger_tokens == ()
        assert pattern.description == ""


class TestExpertTaxonomy:
    """Tests for ExpertTaxonomy model."""

    def test_creation(self):
        """Test creation."""
        identity = ExpertIdentity(
            expert_idx=6,
            layer_idx=0,
            primary_category=ExpertCategory.MATH,
            role=ExpertRole.SPECIALIST,
            confidence=0.9,
            activation_rate=0.15,
        )
        taxonomy = ExpertTaxonomy(
            model_id="test-model",
            num_layers=8,
            num_experts=32,
            expert_identities=(identity,),
        )
        assert taxonomy.model_id == "test-model"
        assert taxonomy.num_layers == 8
        assert len(taxonomy.expert_identities) == 1

    def test_get_experts_by_role(self):
        """Test filtering by role."""
        specialists = tuple(
            ExpertIdentity(
                expert_idx=i,
                layer_idx=0,
                primary_category=ExpertCategory.MATH,
                role=ExpertRole.SPECIALIST,
                confidence=0.9,
                activation_rate=0.1,
            )
            for i in range(3)
        )
        generalists = tuple(
            ExpertIdentity(
                expert_idx=i + 10,
                layer_idx=0,
                primary_category=ExpertCategory.GENERALIST,
                role=ExpertRole.GENERALIST,
                confidence=0.7,
                activation_rate=0.2,
            )
            for i in range(2)
        )
        taxonomy = ExpertTaxonomy(
            model_id="test",
            num_layers=1,
            num_experts=32,
            expert_identities=specialists + generalists,
        )
        found_specialists = taxonomy.get_experts_by_role(ExpertRole.SPECIALIST)
        assert len(found_specialists) == 3
        found_generalists = taxonomy.get_experts_by_role(ExpertRole.GENERALIST)
        assert len(found_generalists) == 2

    def test_get_experts_by_category(self):
        """Test filtering by category."""
        math_experts = tuple(
            ExpertIdentity(
                expert_idx=i,
                layer_idx=0,
                primary_category=ExpertCategory.MATH,
                role=ExpertRole.SPECIALIST,
                confidence=0.9,
                activation_rate=0.1,
            )
            for i in range(2)
        )
        code_experts = tuple(
            ExpertIdentity(
                expert_idx=i + 10,
                layer_idx=0,
                primary_category=ExpertCategory.CODE,
                role=ExpertRole.SPECIALIST,
                confidence=0.85,
                activation_rate=0.15,
            )
            for i in range(3)
        )
        taxonomy = ExpertTaxonomy(
            model_id="test",
            num_layers=1,
            num_experts=32,
            expert_identities=math_experts + code_experts,
        )
        found_math = taxonomy.get_experts_by_category(ExpertCategory.MATH)
        assert len(found_math) == 2
        found_code = taxonomy.get_experts_by_category(ExpertCategory.CODE)
        assert len(found_code) == 3

    def test_get_layer_analysis_found(self):
        """Test getting layer analysis."""
        entropy = RouterEntropy(
            layer_idx=2,
            mean_entropy=1.0,
            max_entropy=1.5,
            normalized_entropy=0.67,
        )
        util = ExpertUtilization(
            layer_idx=2,
            num_experts=8,
            total_activations=100,
            expert_counts=(12, 13, 12, 13, 12, 13, 12, 13),
            expert_frequencies=(0.12, 0.13, 0.12, 0.13, 0.12, 0.13, 0.12, 0.13),
            load_balance_score=0.98,
            most_used_expert=1,
            least_used_expert=0,
        )
        analysis = LayerRoutingAnalysis(
            layer_idx=2,
            entropy=entropy,
            utilization=util,
        )
        taxonomy = ExpertTaxonomy(
            model_id="test",
            num_layers=8,
            num_experts=32,
            layer_analyses=(analysis,),
        )
        found = taxonomy.get_layer_analysis(2)
        assert found is not None
        assert found.layer_idx == 2

    def test_get_layer_analysis_not_found(self):
        """Test getting non-existent layer analysis."""
        taxonomy = ExpertTaxonomy(
            model_id="test",
            num_layers=8,
            num_experts=32,
        )
        assert taxonomy.get_layer_analysis(99) is None


class TestTokenExpertMapping:
    """Tests for TokenExpertMapping model."""

    def test_creation(self):
        """Test creation."""
        mapping = TokenExpertMapping(
            token="hello",
            token_id=1234,
            preferred_experts=(6, 7, 15),
            activation_counts=(50, 45, 30),
        )
        assert mapping.token == "hello"
        assert mapping.token_id == 1234
        assert len(mapping.preferred_experts) == 3

    def test_minimal_creation(self):
        """Test minimal creation."""
        mapping = TokenExpertMapping(
            token="x",
            token_id=0,
        )
        assert mapping.preferred_experts == ()
        assert mapping.activation_counts == ()


class TestVocabExpertAnalysis:
    """Tests for VocabExpertAnalysis model."""

    def test_creation(self):
        """Test creation."""
        mapping1 = TokenExpertMapping(
            token="hello",
            token_id=1234,
            preferred_experts=(6, 7),
            activation_counts=(50, 45),
        )
        mapping2 = TokenExpertMapping(
            token="world",
            token_id=1235,
            preferred_experts=(6, 20),
            activation_counts=(40, 35),
        )
        analysis = VocabExpertAnalysis(
            layer_idx=0,
            total_tokens_analyzed=1000,
            mappings=(mapping1, mapping2),
            expert_vocab_sizes=(100, 50, 80, 120),
        )
        assert analysis.layer_idx == 0
        assert analysis.total_tokens_analyzed == 1000
        assert len(analysis.mappings) == 2
        assert len(analysis.expert_vocab_sizes) == 4

    def test_minimal_creation(self):
        """Test minimal creation."""
        analysis = VocabExpertAnalysis(
            layer_idx=4,
            total_tokens_analyzed=0,
        )
        assert analysis.mappings == ()
        assert analysis.expert_vocab_sizes == ()
