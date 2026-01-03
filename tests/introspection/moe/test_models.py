"""Tests for MoE Pydantic models."""

import pytest

from chuk_lazarus.introspection.moe.enums import (
    ExpertCategory,
    ExpertRole,
    MoEArchitecture,
)
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    CompressionPlan,
    ExpertAblationResult,
    ExpertIdentity,
    ExpertPair,
    ExpertUtilization,
    MoELayerInfo,
    RouterEntropy,
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
