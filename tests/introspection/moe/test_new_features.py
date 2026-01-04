"""Tests for new MoE introspection features.

Tests for:
- Activation overlap computation
- Visualization utilities
- Cross-layer expert tracking
"""

import numpy as np

from chuk_lazarus.introspection.moe.compression import (
    ActivationOverlapResult,
    ExpertActivationStats,
    compute_activation_overlap,
    find_merge_candidates_with_activations,
)
from chuk_lazarus.introspection.moe.tracking import (
    CrossLayerAnalysis,
    ExpertPipeline,
    ExpertPipelineNode,
    analyze_cross_layer_routing,
    compute_layer_alignment,
    identify_functional_pipelines,
    track_expert_across_layers,
)
from chuk_lazarus.introspection.moe.visualization import (
    multi_layer_routing_matrix,
    routing_heatmap_ascii,
    routing_weights_to_matrix,
    utilization_bar_ascii,
)

# =============================================================================
# Tests for Activation Overlap
# =============================================================================


class TestExpertActivationStats:
    """Tests for ExpertActivationStats model."""

    def test_creation(self):
        """Test model creation."""
        stats = ExpertActivationStats(
            expert_idx=0,
            layer_idx=1,
            activation_count=100,
            token_positions=(1, 5, 10, 15),
            total_samples=200,
        )
        assert stats.expert_idx == 0
        assert stats.layer_idx == 1
        assert stats.activation_count == 100

    def test_activation_rate(self):
        """Test activation rate computation."""
        stats = ExpertActivationStats(
            expert_idx=0,
            layer_idx=0,
            activation_count=50,
            total_samples=200,
        )
        assert stats.activation_rate == 0.25

    def test_activation_rate_zero_samples(self):
        """Test activation rate with zero samples."""
        stats = ExpertActivationStats(
            expert_idx=0,
            layer_idx=0,
            activation_count=0,
            total_samples=0,
        )
        assert stats.activation_rate == 0.0


class TestActivationOverlapResult:
    """Tests for ActivationOverlapResult model."""

    def test_creation(self):
        """Test model creation."""
        result = ActivationOverlapResult(
            expert_a=0,
            expert_b=1,
            layer_idx=0,
            jaccard_similarity=0.5,
            overlap_count=10,
            union_count=20,
            a_only_count=5,
            b_only_count=5,
        )
        assert result.expert_a == 0
        assert result.expert_b == 1
        assert result.jaccard_similarity == 0.5


class TestComputeActivationOverlap:
    """Tests for compute_activation_overlap function."""

    def test_basic_overlap(self):
        """Test basic overlap computation."""
        a_activations = {0, 1, 2, 3, 4}
        b_activations = {3, 4, 5, 6, 7}

        result = compute_activation_overlap(
            a_activations,
            b_activations,
            expert_a=0,
            expert_b=1,
            layer_idx=0,
        )

        assert result.overlap_count == 2  # {3, 4}
        assert result.union_count == 8  # {0,1,2,3,4,5,6,7}
        assert result.jaccard_similarity == 0.25  # 2/8

    def test_no_overlap(self):
        """Test with no overlap."""
        a_activations = {0, 1, 2}
        b_activations = {3, 4, 5}

        result = compute_activation_overlap(
            a_activations,
            b_activations,
            expert_a=0,
            expert_b=1,
            layer_idx=0,
        )

        assert result.overlap_count == 0
        assert result.jaccard_similarity == 0.0

    def test_complete_overlap(self):
        """Test with complete overlap."""
        activations = {0, 1, 2, 3}

        result = compute_activation_overlap(
            activations,
            activations,
            expert_a=0,
            expert_b=1,
            layer_idx=0,
        )

        assert result.jaccard_similarity == 1.0

    def test_empty_sets(self):
        """Test with empty sets."""
        result = compute_activation_overlap(
            set(),
            set(),
            expert_a=0,
            expert_b=1,
            layer_idx=0,
        )
        assert result.jaccard_similarity == 0.0


class TestFindMergeCandidatesWithActivations:
    """Tests for find_merge_candidates_with_activations function."""

    def test_basic_finding(self):
        """Test finding candidates with both metrics."""
        from chuk_lazarus.introspection.moe.compression import ExpertSimilarity

        similarities = [
            ExpertSimilarity(
                expert_a=0,
                expert_b=1,
                layer_idx=0,
                weight_cosine_similarity=0.9,
                activation_overlap=0.8,
            ),
            ExpertSimilarity(
                expert_a=0,
                expert_b=2,
                layer_idx=0,
                weight_cosine_similarity=0.5,
                activation_overlap=0.2,
            ),
        ]

        candidates = find_merge_candidates_with_activations(
            similarities,
            weight_threshold=0.8,
            activation_threshold=0.5,
        )

        assert len(candidates) == 1
        assert candidates[0][:2] == (0, 1)

    def test_require_both(self):
        """Test require_both flag."""
        from chuk_lazarus.introspection.moe.compression import ExpertSimilarity

        similarities = [
            ExpertSimilarity(
                expert_a=0,
                expert_b=1,
                layer_idx=0,
                weight_cosine_similarity=0.9,  # High weight
                activation_overlap=0.3,  # Low activation
            ),
        ]

        # Without require_both - should find candidate
        candidates = find_merge_candidates_with_activations(
            similarities,
            weight_threshold=0.8,
            activation_threshold=0.5,
            require_both=False,
        )
        assert len(candidates) == 1

        # With require_both - should not find candidate
        candidates = find_merge_candidates_with_activations(
            similarities,
            weight_threshold=0.8,
            activation_threshold=0.5,
            require_both=True,
        )
        assert len(candidates) == 0


# =============================================================================
# Tests for Visualization
# =============================================================================


class TestRoutingWeightsToMatrix:
    """Tests for routing_weights_to_matrix function."""

    def test_basic_conversion(self):
        """Test basic conversion to matrix."""
        from chuk_lazarus.introspection.moe.models import (
            LayerRouterWeights,
            RouterWeightCapture,
        )

        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="Hello",
                    expert_indices=(0, 1),
                    weights=(0.6, 0.4),
                ),
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=1,
                    token="world",
                    expert_indices=(1, 2),
                    weights=(0.7, 0.3),
                ),
            ),
        )

        matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts=4)

        assert matrix.shape == (2, 4)
        assert tokens == ["Hello", "world"]
        assert matrix[0, 0] == 0.6
        assert matrix[0, 1] == 0.4
        assert matrix[1, 1] == 0.7


class TestMultiLayerRoutingMatrix:
    """Tests for multi_layer_routing_matrix function."""

    def test_mean_aggregation(self):
        """Test mean aggregation across layers."""
        from chuk_lazarus.introspection.moe.models import (
            LayerRouterWeights,
            RouterWeightCapture,
        )

        layer0 = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(1.0,),
                ),
            ),
        )
        layer1 = LayerRouterWeights(
            layer_idx=1,
            positions=(
                RouterWeightCapture(
                    layer_idx=1,
                    position_idx=0,
                    token="A",
                    expert_indices=(1,),
                    weights=(1.0,),
                ),
            ),
        )

        matrix = multi_layer_routing_matrix([layer0, layer1], num_experts=2, aggregation="mean")

        assert matrix.shape == (1, 2)
        assert matrix[0, 0] == 0.5  # Average of 1.0 and 0.0
        assert matrix[0, 1] == 0.5


class TestAsciiVisualization:
    """Tests for ASCII visualization functions."""

    def test_routing_heatmap_ascii(self):
        """Test ASCII heatmap generation."""
        from chuk_lazarus.introspection.moe.models import (
            LayerRouterWeights,
            RouterWeightCapture,
        )

        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="Test",
                    expert_indices=(0,),
                    weights=(0.9,),
                ),
            ),
        )

        output = routing_heatmap_ascii(layer_weights, num_experts=4)

        assert "Layer 0" in output
        assert "Heatmap" in output

    def test_utilization_bar_ascii(self):
        """Test ASCII bar chart generation."""
        from chuk_lazarus.introspection.moe.models import ExpertUtilization

        utilization = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(30, 25, 25, 20),
            expert_frequencies=(0.30, 0.25, 0.25, 0.20),
            load_balance_score=0.95,
            most_used_expert=0,
            least_used_expert=3,
        )

        output = utilization_bar_ascii(utilization)

        assert "Layer 0" in output
        assert "Load Balance" in output


# =============================================================================
# Tests for Cross-Layer Tracking
# =============================================================================


class TestExpertPipelineNode:
    """Tests for ExpertPipelineNode model."""

    def test_creation(self):
        """Test model creation."""
        node = ExpertPipelineNode(
            layer_idx=0,
            expert_idx=5,
            activation_rate=0.8,
            confidence=0.9,
        )
        assert node.layer_idx == 0
        assert node.expert_idx == 5


class TestExpertPipeline:
    """Tests for ExpertPipeline model."""

    def test_creation(self):
        """Test model creation."""
        from chuk_lazarus.introspection.moe.enums import ExpertCategory

        pipeline = ExpertPipeline(
            name="Math Pipeline",
            category=ExpertCategory.MATH,
            nodes=(
                ExpertPipelineNode(layer_idx=0, expert_idx=1, activation_rate=0.8),
                ExpertPipelineNode(layer_idx=1, expert_idx=2, activation_rate=0.7),
            ),
            consistency_score=0.9,
            coverage=0.5,
        )

        assert pipeline.name == "Math Pipeline"
        assert len(pipeline.nodes) == 2

    def test_experts_by_layer(self):
        """Test experts_by_layer property."""
        from chuk_lazarus.introspection.moe.enums import ExpertCategory

        pipeline = ExpertPipeline(
            name="Test",
            category=ExpertCategory.GENERALIST,
            nodes=(
                ExpertPipelineNode(layer_idx=0, expert_idx=1, activation_rate=0.8),
                ExpertPipelineNode(layer_idx=2, expert_idx=3, activation_rate=0.7),
            ),
        )

        by_layer = pipeline.experts_by_layer
        assert by_layer[0] == 1
        assert by_layer[2] == 3

    def test_get_expert_at_layer(self):
        """Test get_expert_at_layer method."""
        from chuk_lazarus.introspection.moe.enums import ExpertCategory

        pipeline = ExpertPipeline(
            name="Test",
            category=ExpertCategory.GENERALIST,
            nodes=(ExpertPipelineNode(layer_idx=0, expert_idx=5, activation_rate=0.8),),
        )

        assert pipeline.get_expert_at_layer(0) == 5
        assert pipeline.get_expert_at_layer(1) is None


class TestComputeLayerAlignment:
    """Tests for compute_layer_alignment function."""

    def test_identical_profiles(self):
        """Test alignment with identical profiles."""
        profile = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = compute_layer_alignment(profile, profile, layer_a=0, layer_b=1)

        assert result.alignment_score > 0.9
        assert len(result.matched_pairs) == 3

    def test_different_profiles(self):
        """Test alignment with different profiles."""
        profile_a = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        profile_b = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )

        result = compute_layer_alignment(profile_a, profile_b, layer_a=0, layer_b=1)

        # Correlation should be negative or zero
        assert result.alignment_score <= 0.5


class TestTrackExpertAcrossLayers:
    """Tests for track_expert_across_layers function."""

    def test_tracking(self):
        """Test expert tracking across layers."""
        # Create profiles where expert 0 in layer 0 correlates with expert 0 in layer 1
        profiles = {
            0: np.array(
                [
                    [0.8, 0.2],
                    [0.9, 0.1],
                    [0.7, 0.3],
                ]
            ),
            1: np.array(
                [
                    [0.85, 0.15],
                    [0.88, 0.12],
                    [0.75, 0.25],
                ]
            ),
        }

        nodes = track_expert_across_layers(profiles, start_layer=0, start_expert=0)

        assert len(nodes) >= 1
        assert nodes[0].expert_idx == 0

    def test_empty_profiles(self):
        """Test with empty profiles."""
        nodes = track_expert_across_layers({}, start_layer=0, start_expert=0)
        assert nodes == []


class TestIdentifyFunctionalPipelines:
    """Tests for identify_functional_pipelines function."""

    def test_identifies_pipelines(self):
        """Test pipeline identification."""
        # Create profiles with clear expert patterns
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.9, 0.1],
                ]
            ),
            1: np.array(
                [
                    [0.85, 0.15, 0.0, 0.0],
                    [0.0, 0.0, 0.85, 0.15],
                ]
            ),
            2: np.array(
                [
                    [0.8, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.2],
                ]
            ),
        }

        pipelines = identify_functional_pipelines(profiles, min_coverage=0.5)

        assert len(pipelines) >= 1

    def test_empty_profiles(self):
        """Test with empty profiles."""
        pipelines = identify_functional_pipelines({})
        assert pipelines == []


class TestAnalyzeCrossLayerRouting:
    """Tests for analyze_cross_layer_routing function."""

    def test_analysis(self):
        """Test cross-layer analysis."""
        from chuk_lazarus.introspection.moe.models import (
            LayerRouterWeights,
            RouterWeightCapture,
        )

        layer0 = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0, 1),
                    weights=(0.6, 0.4),
                ),
            ),
        )
        layer1 = LayerRouterWeights(
            layer_idx=1,
            positions=(
                RouterWeightCapture(
                    layer_idx=1,
                    position_idx=0,
                    token="A",
                    expert_indices=(0, 1),
                    weights=(0.7, 0.3),
                ),
            ),
        )

        result = analyze_cross_layer_routing([layer0, layer1], num_experts=4)

        assert isinstance(result, CrossLayerAnalysis)
        assert result.num_layers == 2
        assert result.num_experts == 4

    def test_empty_analysis(self):
        """Test with empty input."""
        result = analyze_cross_layer_routing([], num_experts=4)

        assert result.num_layers == 0
        assert result.global_consistency == 0.0


# =============================================================================
# Tests for Expert Vocabulary Contribution
# =============================================================================


class TestExpertVocabContribution:
    """Tests for ExpertVocabContribution model."""

    def test_creation(self):
        """Test model creation."""
        from chuk_lazarus.introspection.moe.logit_lens import ExpertVocabContribution

        contrib = ExpertVocabContribution(
            expert_idx=0,
            layer_idx=5,
            top_tokens=("the", "a", "is"),
            top_token_ids=(100, 50, 75),
            top_scores=(0.9, 0.8, 0.7),
            vocab_entropy=5.5,
            specialization_score=0.3,
            dominant_categories=("lowercase", "mixed"),
        )

        assert contrib.expert_idx == 0
        assert contrib.layer_idx == 5
        assert len(contrib.top_tokens) == 3
        assert contrib.specialization_score == 0.3

    def test_default_values(self):
        """Test default values."""
        from chuk_lazarus.introspection.moe.logit_lens import ExpertVocabContribution

        contrib = ExpertVocabContribution(
            expert_idx=0,
            layer_idx=0,
        )

        assert contrib.top_tokens == ()
        assert contrib.vocab_entropy == 0.0
        assert contrib.specialization_score == 0.0


class TestLayerVocabAnalysis:
    """Tests for LayerVocabAnalysis model."""

    def test_creation(self):
        """Test model creation."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
        )

        contrib = ExpertVocabContribution(
            expert_idx=0,
            layer_idx=5,
            top_tokens=("hello",),
            specialization_score=0.5,
        )

        analysis = LayerVocabAnalysis(
            layer_idx=5,
            num_experts=8,
            expert_contributions=(contrib,),
            vocab_coverage=0.1,
            expert_overlap=0.05,
        )

        assert analysis.layer_idx == 5
        assert analysis.num_experts == 8
        assert len(analysis.expert_contributions) == 1
        assert analysis.vocab_coverage == 0.1


class TestTokenExpertPreference:
    """Tests for TokenExpertPreference model."""

    def test_creation(self):
        """Test model creation."""
        from chuk_lazarus.introspection.moe.logit_lens import TokenExpertPreference

        pref = TokenExpertPreference(
            token="hello",
            token_id=12345,
            preferred_experts=(0, 3, 5),
            preference_scores=(0.9, 0.5, 0.3),
        )

        assert pref.token == "hello"
        assert pref.token_id == 12345
        assert pref.preferred_experts[0] == 0


class TestVocabExpertMapping:
    """Tests for VocabExpertMapping model."""

    def test_creation(self):
        """Test model creation."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            TokenExpertPreference,
            VocabExpertMapping,
        )

        pref = TokenExpertPreference(
            token="test",
            token_id=100,
            preferred_experts=(2,),
            preference_scores=(0.8,),
        )

        mapping = VocabExpertMapping(
            layer_idx=10,
            num_experts=8,
            num_tokens=1,
            token_preferences=(pref,),
            expert_vocab_sizes=(0, 0, 1, 0, 0, 0, 0, 0),
        )

        assert mapping.layer_idx == 10
        assert mapping.num_tokens == 1
        assert mapping.expert_vocab_sizes[2] == 1


class TestFindExpertSpecialists:
    """Tests for find_expert_specialists function."""

    def test_finds_specialists(self):
        """Test finding specialists."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
            find_expert_specialists,
        )

        contribs = (
            ExpertVocabContribution(
                expert_idx=0,
                layer_idx=0,
                specialization_score=0.8,
                dominant_categories=("numbers",),
            ),
            ExpertVocabContribution(
                expert_idx=1,
                layer_idx=0,
                specialization_score=0.2,
                dominant_categories=("mixed",),
            ),
            ExpertVocabContribution(
                expert_idx=2,
                layer_idx=0,
                specialization_score=0.5,
                dominant_categories=("punctuation",),
            ),
        )

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=3,
            expert_contributions=contribs,
        )

        specialists = find_expert_specialists(analysis, min_specialization=0.3)

        assert len(specialists) == 2
        assert specialists[0][0] == 0  # Expert 0 has highest specialization
        assert specialists[0][1] == "numbers"
        assert specialists[0][2] == 0.8

    def test_no_specialists(self):
        """Test when no specialists found."""
        from chuk_lazarus.introspection.moe.logit_lens import (
            ExpertVocabContribution,
            LayerVocabAnalysis,
            find_expert_specialists,
        )

        contribs = (
            ExpertVocabContribution(
                expert_idx=0,
                layer_idx=0,
                specialization_score=0.1,
            ),
        )

        analysis = LayerVocabAnalysis(
            layer_idx=0,
            num_experts=1,
            expert_contributions=contribs,
        )

        specialists = find_expert_specialists(analysis, min_specialization=0.5)
        assert len(specialists) == 0
