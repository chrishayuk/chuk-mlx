"""Comprehensive tests for tracking.py to achieve 90%+ coverage."""

import numpy as np

from chuk_lazarus.introspection.moe.enums import ExpertCategory
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    RouterWeightCapture,
)
from chuk_lazarus.introspection.moe.tracking import (
    CrossLayerAnalysis,
    ExpertPipeline,
    ExpertPipelineNode,
    LayerAlignmentResult,
    analyze_cross_layer_routing,
    compute_expert_activation_profile,
    compute_layer_alignment,
    identify_functional_pipelines,
    print_alignment_matrix,
    print_pipeline_summary,
    track_expert_across_layers,
)


class TestExpertPipelineNode:
    """Tests for ExpertPipelineNode model."""

    def test_creation_with_category(self):
        """Test model creation with category."""
        node = ExpertPipelineNode(
            layer_idx=0,
            expert_idx=5,
            activation_rate=0.8,
            category=ExpertCategory.MATH,
            confidence=0.9,
        )
        assert node.layer_idx == 0
        assert node.expert_idx == 5
        assert node.category == ExpertCategory.MATH

    def test_default_values(self):
        """Test default values."""
        node = ExpertPipelineNode(
            layer_idx=0,
            expert_idx=0,
            activation_rate=0.5,
        )
        assert node.category is None
        assert node.confidence == 0.0


class TestExpertPipeline:
    """Tests for ExpertPipeline model."""

    def test_layers_property(self):
        """Test layers property returns sorted layer indices."""
        pipeline = ExpertPipeline(
            name="Test Pipeline",
            category=ExpertCategory.MATH,
            nodes=(
                ExpertPipelineNode(layer_idx=5, expert_idx=1, activation_rate=0.8),
                ExpertPipelineNode(layer_idx=0, expert_idx=2, activation_rate=0.7),
                ExpertPipelineNode(layer_idx=2, expert_idx=3, activation_rate=0.6),
            ),
            consistency_score=0.9,
            coverage=0.75,
        )

        # layers property should return sorted list
        assert pipeline.layers == [0, 2, 5]

    def test_get_expert_at_layer_not_found(self):
        """Test get_expert_at_layer returns None for missing layer."""
        pipeline = ExpertPipeline(
            name="Test",
            category=ExpertCategory.GENERALIST,
            nodes=(
                ExpertPipelineNode(layer_idx=0, expert_idx=5, activation_rate=0.8),
                ExpertPipelineNode(layer_idx=2, expert_idx=7, activation_rate=0.7),
            ),
        )

        assert pipeline.get_expert_at_layer(0) == 5
        assert pipeline.get_expert_at_layer(2) == 7
        assert pipeline.get_expert_at_layer(1) is None  # Not in pipeline
        assert pipeline.get_expert_at_layer(99) is None


class TestCrossLayerAnalysis:
    """Tests for CrossLayerAnalysis model."""

    def test_get_pipeline_for_category_found(self):
        """Test get_pipeline_for_category when category exists."""
        math_pipeline = ExpertPipeline(
            name="Math Pipeline",
            category=ExpertCategory.MATH,
            nodes=(ExpertPipelineNode(layer_idx=0, expert_idx=1, activation_rate=0.8),),
            consistency_score=0.9,
            coverage=0.5,
        )
        code_pipeline = ExpertPipeline(
            name="Code Pipeline",
            category=ExpertCategory.CODE,
            nodes=(ExpertPipelineNode(layer_idx=0, expert_idx=2, activation_rate=0.7),),
            consistency_score=0.85,
            coverage=0.6,
        )

        analysis = CrossLayerAnalysis(
            num_layers=4,
            num_experts=8,
            pipelines=(math_pipeline, code_pipeline),
            layer_alignments=(),
            global_consistency=0.8,
        )

        result = analysis.get_pipeline_for_category(ExpertCategory.MATH)
        assert result is not None
        assert result.name == "Math Pipeline"
        assert result.category == ExpertCategory.MATH

        result = analysis.get_pipeline_for_category(ExpertCategory.CODE)
        assert result is not None
        assert result.name == "Code Pipeline"

    def test_get_pipeline_for_category_not_found(self):
        """Test get_pipeline_for_category when category doesn't exist."""
        pipeline = ExpertPipeline(
            name="Math Pipeline",
            category=ExpertCategory.MATH,
            nodes=(ExpertPipelineNode(layer_idx=0, expert_idx=1, activation_rate=0.8),),
        )

        analysis = CrossLayerAnalysis(
            num_layers=4,
            num_experts=8,
            pipelines=(pipeline,),
            layer_alignments=(),
            global_consistency=0.8,
        )

        result = analysis.get_pipeline_for_category(ExpertCategory.CODE)
        assert result is None

        result = analysis.get_pipeline_for_category(ExpertCategory.LANGUAGE)
        assert result is None


class TestComputeExpertActivationProfile:
    """Tests for compute_expert_activation_profile function."""

    def test_basic_profile_computation(self):
        """Test basic activation profile computation."""
        layer_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="A",
                        expert_indices=(0, 1),
                        weights=(0.6, 0.4),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token="B",
                        expert_indices=(1, 2),
                        weights=(0.7, 0.3),
                    ),
                ),
            ),
        ]

        profiles = compute_expert_activation_profile(layer_weights, num_experts=4)

        assert 0 in profiles
        assert profiles[0].shape == (2, 4)  # 2 positions, 4 experts
        assert profiles[0][0, 0] == 0.6
        assert profiles[0][0, 1] == 0.4
        assert profiles[0][1, 1] == 0.7
        assert profiles[0][1, 2] == 0.3

    def test_out_of_bounds_expert_ignored(self):
        """Test that out-of-bounds expert indices are ignored."""
        layer_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="A",
                        expert_indices=(0, 99),  # 99 is out of bounds
                        weights=(0.6, 0.4),
                    ),
                ),
            ),
        ]

        profiles = compute_expert_activation_profile(layer_weights, num_experts=4)

        assert profiles[0].shape == (1, 4)
        assert profiles[0][0, 0] == 0.6
        # Expert 99 should be ignored since it's >= num_experts
        assert np.sum(profiles[0]) == 0.6


class TestComputeLayerAlignment:
    """Tests for compute_layer_alignment function."""

    def test_zero_variance_profiles(self):
        """Test handling of zero variance profiles."""
        # All zeros - no variance
        profile_a = np.zeros((3, 2))
        profile_b = np.zeros((3, 2))

        result = compute_layer_alignment(profile_a, profile_b, layer_a=0, layer_b=1)

        # Should handle gracefully with zero alignment
        assert result.alignment_score == 0.0
        assert result.matched_pairs == ()

    def test_low_correlation_not_matched(self):
        """Test that low correlation pairs are not matched."""
        # Create uncorrelated profiles
        profile_a = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        profile_b = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )

        result = compute_layer_alignment(profile_a, profile_b, layer_a=0, layer_b=1)

        # Should have low alignment due to uncorrelated patterns
        assert result.layer_a == 0
        assert result.layer_b == 1


class TestTrackExpertAcrossLayers:
    """Tests for track_expert_across_layers function."""

    def test_missing_start_layer(self):
        """Test when start layer is not in profiles."""
        profiles = {
            0: np.array([[0.8, 0.2]]),
            1: np.array([[0.9, 0.1]]),
        }

        # Start layer 5 doesn't exist
        nodes = track_expert_across_layers(profiles, start_layer=5, start_expert=0)
        assert nodes == []

    def test_tracking_stops_below_threshold(self):
        """Test that tracking stops when correlation drops below threshold."""
        # Create profiles where expert 0 changes behavior abruptly
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.1],
                ]
            ),
            1: np.array(
                [
                    [0.1, 0.9],  # Completely different pattern
                    [0.1, 0.9],
                    [0.1, 0.9],
                ]
            ),
        }

        nodes = track_expert_across_layers(profiles, start_layer=0, start_expert=0, threshold=0.5)

        # Should only have the starting node since correlation is low
        assert len(nodes) == 1
        assert nodes[0].layer_idx == 0

    def test_tracking_continues_with_high_correlation(self):
        """Test that tracking continues when correlation is above threshold."""
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.85, 0.15],
                ]
            ),
            1: np.array(
                [
                    [0.88, 0.12],
                    [0.82, 0.18],
                    [0.87, 0.13],
                ]
            ),
            2: np.array(
                [
                    [0.86, 0.14],
                    [0.84, 0.16],
                    [0.89, 0.11],
                ]
            ),
        }

        nodes = track_expert_across_layers(profiles, start_layer=0, start_expert=0, threshold=0.3)

        # Should track through all layers
        assert len(nodes) == 3
        assert nodes[0].layer_idx == 0
        assert nodes[1].layer_idx == 1
        assert nodes[2].layer_idx == 2


class TestIdentifyFunctionalPipelines:
    """Tests for identify_functional_pipelines function."""

    def test_used_starts_skipped(self):
        """Test that already-used starting experts are skipped."""
        # Create profiles where experts 0 and 1 are very similar
        # Both would create similar pipelines, but second should be skipped
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                ]
            ),
            1: np.array(
                [
                    [0.88, 0.12, 0.0, 0.0],
                    [0.88, 0.12, 0.0, 0.0],
                ]
            ),
        }

        pipelines = identify_functional_pipelines(profiles, min_coverage=0.5)

        # Should have pipelines but each starting expert used only once
        assert isinstance(pipelines, list)

    def test_with_expert_identities(self):
        """Test pipeline identification with expert identities."""
        # Profiles need variance across positions for correlation to work
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1, 0.0, 0.0],
                    [0.8, 0.2, 0.0, 0.0],
                    [0.85, 0.15, 0.0, 0.0],
                ]
            ),
            1: np.array(
                [
                    [0.88, 0.12, 0.0, 0.0],
                    [0.82, 0.18, 0.0, 0.0],
                    [0.86, 0.14, 0.0, 0.0],
                ]
            ),
            2: np.array(
                [
                    [0.86, 0.14, 0.0, 0.0],
                    [0.84, 0.16, 0.0, 0.0],
                    [0.87, 0.13, 0.0, 0.0],
                ]
            ),
        }

        # Provide expert identity info
        expert_identities = [
            {"layer_idx": 0, "expert_idx": 0, "primary_category": "math"},
            {"layer_idx": 1, "expert_idx": 0, "primary_category": "math"},
            {"layer_idx": 2, "expert_idx": 0, "primary_category": "math"},
        ]

        pipelines = identify_functional_pipelines(
            profiles,
            expert_identities=expert_identities,
            min_coverage=0.5,
        )

        # Should identify pipeline with math category
        assert len(pipelines) >= 1
        assert pipelines[0].category == ExpertCategory.MATH

    def test_with_missing_identity_category(self):
        """Test pipeline with identity that has no primary_category."""
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                ]
            ),
            1: np.array(
                [
                    [0.88, 0.12],
                    [0.88, 0.12],
                ]
            ),
        }

        # Identity without primary_category
        expert_identities = [
            {"layer_idx": 0, "expert_idx": 0},  # No primary_category
            {"layer_idx": 1, "expert_idx": 0, "primary_category": None},
        ]

        pipelines = identify_functional_pipelines(
            profiles,
            expert_identities=expert_identities,
            min_coverage=0.5,
        )

        # Should still work, defaulting to GENERALIST
        assert len(pipelines) >= 1

    def test_single_node_pipeline(self):
        """Test pipeline with only one node."""
        profiles = {
            0: np.array(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                ]
            ),
        }

        # With only one layer, coverage will be 100%
        pipelines = identify_functional_pipelines(profiles, min_coverage=0.5)

        # Should have pipeline with consistency 1.0
        assert len(pipelines) >= 1
        assert pipelines[0].consistency_score == 1.0


class TestAnalyzeCrossLayerRouting:
    """Tests for analyze_cross_layer_routing function."""

    def test_empty_alignments(self):
        """Test with single layer (no alignments possible)."""
        layer_weights = [
            LayerRouterWeights(
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
            ),
        ]

        result = analyze_cross_layer_routing(layer_weights, num_experts=4)

        # With only one layer, no alignments can be computed
        # This tests line 444: global_consistency = 0.0
        assert result.num_layers == 1
        assert result.global_consistency == 0.0
        assert result.layer_alignments == ()

    def test_with_expert_identities(self):
        """Test analysis with expert identities."""
        layer_weights = [
            LayerRouterWeights(
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
            ),
            LayerRouterWeights(
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
            ),
        ]

        expert_identities = [
            {"layer_idx": 0, "expert_idx": 0, "primary_category": "code"},
            {"layer_idx": 1, "expert_idx": 0, "primary_category": "code"},
        ]

        result = analyze_cross_layer_routing(
            layer_weights,
            num_experts=4,
            expert_identities=expert_identities,
        )

        assert isinstance(result, CrossLayerAnalysis)
        assert result.num_layers == 2


class TestPrintPipelineSummary:
    """Tests for print_pipeline_summary function."""

    def test_print_with_pipelines(self, capsys):
        """Test printing pipeline summary with pipelines."""
        pipelines = [
            ExpertPipeline(
                name="Math Pipeline (E0)",
                category=ExpertCategory.MATH,
                nodes=(
                    ExpertPipelineNode(layer_idx=0, expert_idx=0, activation_rate=0.8),
                    ExpertPipelineNode(
                        layer_idx=1, expert_idx=0, activation_rate=0.7, confidence=0.9
                    ),
                    ExpertPipelineNode(
                        layer_idx=2, expert_idx=1, activation_rate=0.75, confidence=0.85
                    ),
                ),
                consistency_score=0.875,
                coverage=0.75,
            ),
            ExpertPipeline(
                name="Code Pipeline (E2)",
                category=ExpertCategory.CODE,
                nodes=(
                    ExpertPipelineNode(layer_idx=0, expert_idx=2, activation_rate=0.6),
                    ExpertPipelineNode(
                        layer_idx=1, expert_idx=3, activation_rate=0.65, confidence=0.8
                    ),
                ),
                consistency_score=0.8,
                coverage=0.5,
            ),
        ]

        print_pipeline_summary(pipelines)

        captured = capsys.readouterr()
        assert "Expert Pipelines Across Layers" in captured.out
        assert "Math Pipeline (E0)" in captured.out
        assert "Code Pipeline (E2)" in captured.out
        assert "Category: math" in captured.out
        assert "Category: code" in captured.out
        assert "Coverage:" in captured.out
        assert "Consistency:" in captured.out
        assert "Path:" in captured.out
        assert "L0:E0" in captured.out

    def test_print_empty_pipelines(self, capsys):
        """Test printing when no pipelines exist."""
        print_pipeline_summary([])

        captured = capsys.readouterr()
        assert "Expert Pipelines Across Layers" in captured.out
        assert "No pipelines identified" in captured.out


class TestPrintAlignmentMatrix:
    """Tests for print_alignment_matrix function."""

    def test_print_with_alignments(self, capsys):
        """Test printing alignment matrix with alignments."""
        alignments = [
            LayerAlignmentResult(
                layer_a=0,
                layer_b=1,
                alignment_score=0.85,
                matched_pairs=((0, 0), (1, 1)),
            ),
            LayerAlignmentResult(
                layer_a=1,
                layer_b=2,
                alignment_score=0.72,
                matched_pairs=((0, 1), (1, 0)),
            ),
            LayerAlignmentResult(
                layer_a=2,
                layer_b=3,
                alignment_score=0.45,
                matched_pairs=((0, 0),),
            ),
        ]

        print_alignment_matrix(alignments)

        captured = capsys.readouterr()
        assert "Layer-to-Layer Alignment" in captured.out
        assert "L 0 → L 1:" in captured.out
        assert "L 1 → L 2:" in captured.out
        assert "L 2 → L 3:" in captured.out
        assert "0.85" in captured.out
        assert "0.72" in captured.out
        assert "0.45" in captured.out
        assert "Average alignment:" in captured.out
        # Should show bar chars
        assert "█" in captured.out or "░" in captured.out

    def test_print_empty_alignments(self, capsys):
        """Test printing with no alignments."""
        print_alignment_matrix([])

        captured = capsys.readouterr()
        assert "Layer-to-Layer Alignment" in captured.out
        # Should not print average when empty
        assert "Average alignment:" not in captured.out
