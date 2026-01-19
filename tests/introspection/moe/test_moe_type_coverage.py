"""Additional tests for moe_type.py to improve coverage."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from chuk_lazarus.introspection.moe import MoEType
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.moe_type import (
    MoETypeAnalysis,
    MoETypeService,
    ProjectionRankAnalysis,
    TrainingOriginSignals,
)


class TestProjectionRankAnalysisProperties:
    """Tests for ProjectionRankAnalysis computed properties."""

    def test_rank_ratio_with_max_rank_zero(self):
        """Test rank_ratio property when max_rank would cause division by zero (line 51)."""
        # Create analysis with max_rank=1 (minimum allowed) and effective_rank=0
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(10, 10),
            max_rank=1,
            effective_rank_95=0,
        )
        assert analysis.rank_ratio == 0.0

    def test_compression_ratio_normal(self):
        """Test compression_ratio property with normal values (lines 53-61)."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(100, 200),
            max_rank=100,
            effective_rank_95=10,
        )
        # Original: 100 * 200 = 20000
        # Factorized: 10 * (100 + 200) = 3000
        # Ratio: 20000 / 3000 = 6.67
        assert abs(analysis.compression_ratio - (20000 / 3000)) < 0.01

    def test_compression_ratio_zero_effective_rank(self):
        """Test compression_ratio property when effective_rank_95 is 0 (line 56-57)."""
        analysis = ProjectionRankAnalysis(
            name="up",
            shape=(50, 50),
            max_rank=50,
            effective_rank_95=0,
        )
        assert analysis.compression_ratio == float("inf")


class TestTrainingOriginSignals:
    """Tests for TrainingOriginSignals computed properties."""

    def test_upcycled_score_high_similarity_low_rank(self):
        """Test upcycled_score with high similarity and low rank (lines 99-109)."""
        signals = TrainingOriginSignals(
            expert_similarity=0.5,  # High similarity
            rank_ratio=0.05,  # Low rank ratio
            layer_consistency=1.0,
            expert_norm_variance=0.01,  # Low variance
            router_entropy=0.0,
        )
        assert signals.upcycled_score > 0.5  # Should indicate upcycled

    def test_upcycled_score_low_similarity_high_rank(self):
        """Test upcycled_score with low similarity and high rank."""
        signals = TrainingOriginSignals(
            expert_similarity=0.0,  # Low similarity
            rank_ratio=0.7,  # High rank ratio
            layer_consistency=0.5,
            expert_norm_variance=0.5,  # High variance
            router_entropy=0.0,
        )
        assert signals.upcycled_score < 0.3  # Should NOT indicate upcycled

    def test_upcycled_score_boundary_values(self):
        """Test upcycled_score with boundary values for clamping (lines 103-106)."""
        # Very low similarity (below 0.1 threshold)
        signals1 = TrainingOriginSignals(
            expert_similarity=0.0,
            rank_ratio=0.0,
            expert_norm_variance=0.0,
        )
        # sim_score should be clamped to 0
        assert signals1.upcycled_score >= 0

        # Very high similarity
        signals2 = TrainingOriginSignals(
            expert_similarity=1.0,
            rank_ratio=0.0,
            expert_norm_variance=0.0,
        )
        assert signals2.upcycled_score <= 1.0

    def test_pretrained_score_high_rank_low_similarity(self):
        """Test pretrained_score with high rank and low similarity (lines 111-119)."""
        signals = TrainingOriginSignals(
            expert_similarity=0.0,  # Very low similarity
            rank_ratio=0.6,  # High rank ratio
        )
        assert signals.pretrained_score > 0.5  # Should indicate pretrained

    def test_pretrained_score_low_rank_high_similarity(self):
        """Test pretrained_score with low rank and high similarity."""
        signals = TrainingOriginSignals(
            expert_similarity=0.5,  # High similarity
            rank_ratio=0.1,  # Low rank ratio
        )
        assert signals.pretrained_score < 0.3  # Should NOT indicate pretrained

    def test_pretrained_score_boundary_values(self):
        """Test pretrained_score with boundary values for clamping (lines 115-116)."""
        # Values at boundary
        signals = TrainingOriginSignals(
            expert_similarity=0.1,  # At boundary
            rank_ratio=0.3,  # At boundary
        )
        # Should be around 0 (both at boundary values)
        assert 0 <= signals.pretrained_score <= 1.0


class TestMoETypeAnalysisProperties:
    """Tests for MoETypeAnalysis computed properties."""

    def _create_analysis(
        self,
        moe_type: MoEType = MoEType.UNKNOWN,
        confidence: float = 0.5,
        gate_eff_rank: int = 10,
        up_eff_rank: int = 100,
        down_eff_rank: int = 100,
    ) -> MoETypeAnalysis:
        """Helper to create MoETypeAnalysis."""
        return MoETypeAnalysis(
            model_id="test-model",
            layer_idx=0,
            num_experts=8,
            moe_type=moe_type,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(100, 100), max_rank=100, effective_rank_95=gate_eff_rank
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(100, 400), max_rank=100, effective_rank_95=up_eff_rank
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(400, 100), max_rank=100, effective_rank_95=down_eff_rank
            ),
            mean_cosine_similarity=0.5,
            std_cosine_similarity=0.1,
            confidence=confidence,
        )

    def test_estimated_compression(self):
        """Test estimated_compression property (lines 170-177)."""
        analysis = self._create_analysis(gate_eff_rank=10, up_eff_rank=50, down_eff_rank=50)
        # Gate: 100*100 = 10000, compressed: 10*200 = 2000
        # Up: 100*400 = 40000, compressed: 50*500 = 25000
        # Down: 400*100 = 40000, compressed: 50*500 = 25000
        # Total original: 90000, Total compressed: 52000
        # Ratio: 90000 / 52000 = ~1.73
        assert analysis.estimated_compression > 1.0

    def test_estimated_compression_zero_compressed(self):
        """Test estimated_compression when compressed size would be 0 (line 177)."""
        analysis = self._create_analysis(gate_eff_rank=0, up_eff_rank=0, down_eff_rank=0)
        # All zero effective ranks means compressed size = 0
        # Should return 1.0 to avoid division by zero
        assert analysis.estimated_compression == 1.0

    def test_is_compressible_upcycled(self):
        """Test is_compressible property for UPCYCLED (line 180-182)."""
        analysis = self._create_analysis(moe_type=MoEType.UPCYCLED)
        assert analysis.is_compressible is True

    def test_is_compressible_pretrained(self):
        """Test is_compressible property for PRETRAINED_MOE."""
        analysis = self._create_analysis(moe_type=MoEType.PRETRAINED_MOE)
        assert analysis.is_compressible is False

    def test_is_compressible_unknown(self):
        """Test is_compressible property for UNKNOWN."""
        analysis = self._create_analysis(moe_type=MoEType.UNKNOWN)
        assert analysis.is_compressible is False

    def test_training_origin_upcycled(self):
        """Test training_origin property for UPCYCLED (lines 185-192)."""
        analysis = self._create_analysis(moe_type=MoEType.UPCYCLED)
        assert analysis.training_origin == "Upcycled from dense model"

    def test_training_origin_pretrained(self):
        """Test training_origin property for PRETRAINED_MOE (line 189-190)."""
        analysis = self._create_analysis(moe_type=MoEType.PRETRAINED_MOE)
        assert analysis.training_origin == "Pretrained as MoE from scratch"

    def test_training_origin_unknown(self):
        """Test training_origin property for UNKNOWN (line 191-192)."""
        analysis = self._create_analysis(moe_type=MoEType.UNKNOWN)
        assert analysis.training_origin == "Unknown training origin"

    def test_confidence_label_high(self):
        """Test confidence_label property for high confidence (lines 194-204)."""
        analysis = self._create_analysis(confidence=0.9)
        assert analysis.confidence_label == "High"

    def test_confidence_label_medium(self):
        """Test confidence_label property for medium confidence (line 199-200)."""
        analysis = self._create_analysis(confidence=0.6)
        assert analysis.confidence_label == "Medium"

    def test_confidence_label_low(self):
        """Test confidence_label property for low confidence (line 201-202)."""
        analysis = self._create_analysis(confidence=0.35)
        assert analysis.confidence_label == "Low"

    def test_confidence_label_very_low(self):
        """Test confidence_label property for very low confidence (line 203-204)."""
        analysis = self._create_analysis(confidence=0.1)
        assert analysis.confidence_label == "Very Low"

    def test_confidence_label_boundaries(self):
        """Test confidence_label at exact boundaries."""
        # Exactly 0.8 -> High
        assert self._create_analysis(confidence=0.8).confidence_label == "High"
        # Just under 0.8 -> Medium
        assert self._create_analysis(confidence=0.79).confidence_label == "Medium"
        # Exactly 0.5 -> Medium
        assert self._create_analysis(confidence=0.5).confidence_label == "Medium"
        # Just under 0.5 -> Low
        assert self._create_analysis(confidence=0.49).confidence_label == "Low"
        # Exactly 0.3 -> Low
        assert self._create_analysis(confidence=0.3).confidence_label == "Low"
        # Just under 0.3 -> Very Low
        assert self._create_analysis(confidence=0.29).confidence_label == "Very Low"


class TestMoETypeServiceClassifyExtended:
    """Extended tests for _classify method."""

    def test_classify_returns_training_signals(self):
        """Test that _classify returns TrainingOriginSignals."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.1,
            similarity=0.3,
        )
        assert isinstance(signals, TrainingOriginSignals)
        assert signals.expert_similarity == 0.3
        assert signals.rank_ratio == 0.1

    def test_classify_with_expert_norm_variance(self):
        """Test _classify with expert_norm_variance parameter."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.01,
            similarity=0.5,
            expert_norm_variance=0.02,
        )
        assert signals.expert_norm_variance == 0.02

    def test_classify_unknown_ambiguous_scores(self):
        """Test _classify returns UNKNOWN when scores are ambiguous (line 370-372)."""
        # Create a case where neither score is > 0.6
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.15,  # Medium rank
            similarity=0.15,  # Medium similarity
        )
        assert moe_type == MoEType.UNKNOWN


class TestMoETypeServiceHelperMethods:
    """Tests for MoETypeService helper methods."""

    def test_get_experts_layer_out_of_range(self):
        """Test _get_experts when layer_idx is out of range (line 404-405)."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": []})()

        model = MockModel()
        result = MoETypeService._get_experts(model, 99)
        assert result is None

    def test_get_experts_no_mlp(self):
        """Test _get_experts when layer has no mlp (line 409-410)."""

        class MockLayer:
            pass  # No mlp attribute

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [MockLayer()]})()

        model = MockModel()
        result = MoETypeService._get_experts(model, 0)
        assert result is None

    def test_get_experts_no_experts(self):
        """Test _get_experts when mlp has no experts (line 412)."""

        class MockMLP:
            pass  # No experts attribute

        class MockLayer:
            def __init__(self):
                self.mlp = MockMLP()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [MockLayer()]})()

        model = MockModel()
        result = MoETypeService._get_experts(model, 0)
        assert result is None

    def test_get_experts_success(self):
        """Test _get_experts successful extraction."""

        class MockExperts:
            pass

        class MockMLP:
            def __init__(self):
                self.experts = MockExperts()

        class MockLayer:
            def __init__(self):
                self.mlp = MockMLP()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("Model", (), {"layers": [MockLayer()]})()

        model = MockModel()
        result = MoETypeService._get_experts(model, 0)
        assert isinstance(result, MockExperts)

    def test_extract_weights_unknown_architecture(self):
        """Test _extract_weights with unsupported architecture (line 435)."""

        # Create experts that don't match any architecture
        class UnknownExperts:
            pass  # Not a list, doesn't have gate_up_proj_blocks

        with pytest.raises(ValueError, match="Unknown expert structure"):
            MoETypeService._extract_weights(UnknownExperts(), MoEArchitecture.GENERIC)


class TestExtractListWeights:
    """Tests for _extract_list_weights method."""

    def test_extract_list_weights_basic(self):
        """Test _extract_list_weights with basic expert structure (lines 504-531)."""

        class MockExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(32, 64, bias=False)
                self.up_proj = nn.Linear(32, 64, bias=False)
                self.down_proj = nn.Linear(64, 32, bias=False)

        experts = [MockExpert() for _ in range(4)]

        gate, up, down, num_experts = MoETypeService._extract_list_weights(experts)

        assert num_experts == 4
        assert gate.shape == (4, 64, 32)
        assert up.shape == (4, 64, 32)
        assert down.shape == (4, 32, 64)


class TestAnalyzeProjection:
    """Tests for _analyze_projection method."""

    def test_analyze_projection_basic(self):
        """Test _analyze_projection with small weights (lines 533-592)."""
        # Create simple weights (4 experts, 10x10)
        weights = mx.random.normal((4, 10, 10))

        analysis = MoETypeService._analyze_projection(weights, "gate")

        assert analysis.name == "gate"
        assert analysis.shape == (10, 10)
        assert analysis.max_rank == 10
        assert 0 <= analysis.effective_rank_95 <= 10

    def test_analyze_projection_sampling(self):
        """Test _analyze_projection with many experts (triggers sampling, lines 557-563)."""
        # More than MAX_EXPERTS_TO_SAMPLE experts
        num_experts = MoETypeService.MAX_EXPERTS_TO_SAMPLE + 5
        weights = mx.random.normal((num_experts, 10, 10))

        analysis = MoETypeService._analyze_projection(weights, "up")

        assert analysis.name == "up"
        assert analysis.shape == (10, 10)

    def test_analyze_projection_svd_error(self):
        """Test _analyze_projection handles SVD errors gracefully (lines 579-580)."""
        # Create weights that might cause SVD issues (all same values)
        weights = mx.ones((2, 5, 5))

        # Should not raise, just return rank 0
        analysis = MoETypeService._analyze_projection(weights, "down")
        assert analysis.effective_rank_95 >= 0


class TestComputeSimilarities:
    """Tests for _compute_similarities method."""

    def test_compute_similarities_basic(self):
        """Test _compute_similarities with random weights (lines 610-651)."""
        weights = mx.random.normal((4, 10, 10))

        mean_sim, std_sim, matrix = MoETypeService._compute_similarities(weights)

        assert -1.0 <= mean_sim <= 1.0
        assert std_sim >= 0.0
        assert len(matrix) == 4
        assert len(matrix[0]) == 4

    def test_compute_similarities_identical_experts(self):
        """Test _compute_similarities when experts are identical."""
        # All experts are the same - should have similarity 1.0
        base = mx.random.normal((10, 10))
        weights = mx.stack([base, base, base], axis=0)

        mean_sim, std_sim, matrix = MoETypeService._compute_similarities(weights)

        # Identical experts should have very high similarity
        assert mean_sim > 0.99

    def test_compute_similarities_single_expert(self):
        """Test _compute_similarities with single expert (line 644-646)."""
        weights = mx.random.normal((1, 10, 10))

        mean_sim, std_sim, matrix = MoETypeService._compute_similarities(weights)

        # Single expert means no pairs to compare
        assert mean_sim == 0.0
        assert std_sim == 0.0


class TestComputeExpertNormVariance:
    """Tests for _compute_expert_norm_variance method."""

    def test_compute_expert_norm_variance_basic(self):
        """Test _compute_expert_norm_variance with random weights (lines 653-681)."""
        weights = mx.random.normal((4, 10, 10))

        variance = MoETypeService._compute_expert_norm_variance(weights)

        assert variance >= 0.0

    def test_compute_expert_norm_variance_identical_norms(self):
        """Test _compute_expert_norm_variance when all norms are equal."""
        # Create experts with same norm (uniformly scaled identity-like)
        weights = mx.stack([mx.eye(10) * i for i in [1, 1, 1, 1]], axis=0)

        variance = MoETypeService._compute_expert_norm_variance(weights)

        # Should have very low variance
        assert variance < 0.01

    def test_compute_expert_norm_variance_varied_norms(self):
        """Test _compute_expert_norm_variance with varied norms."""
        # Create experts with very different norms
        weights = mx.stack(
            [
                mx.eye(10) * 1.0,
                mx.eye(10) * 10.0,
                mx.eye(10) * 100.0,
                mx.eye(10) * 1000.0,
            ],
            axis=0,
        )

        variance = MoETypeService._compute_expert_norm_variance(weights)

        # Should have significant variance
        assert variance > 0.5

    def test_compute_expert_norm_variance_zero_mean(self):
        """Test _compute_expert_norm_variance when mean norm is 0 (line 679-680)."""
        # All zeros - mean norm is 0
        weights = mx.zeros((4, 10, 10))

        variance = MoETypeService._compute_expert_norm_variance(weights)

        assert variance == 0.0


class TestComputeEffectiveRank:
    """Tests for _compute_effective_rank method."""

    def test_compute_effective_rank_typical(self):
        """Test _compute_effective_rank with typical singular values (lines 594-608)."""
        # Decaying singular values
        S = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])

        rank = MoETypeService._compute_effective_rank(S)

        # Should need only a few to capture 95%
        assert 1 <= rank <= 4

    def test_compute_effective_rank_all_zeros(self):
        """Test _compute_effective_rank when total variance is 0 (line 605-606)."""
        S = np.zeros(10)

        rank = MoETypeService._compute_effective_rank(S)

        assert rank == 0


class TestExtractBatchedWeightsAndBiases:
    """Tests for _extract_batched_weights and _extract_biases methods."""

    def test_extract_biases_no_bias(self):
        """Test _extract_biases when no biases present (lines 481-502)."""

        class NoBiasExperts:
            pass  # No gate_up_proj_bias or down_proj_bias

        gate_bias, up_bias, down_bias = MoETypeService._extract_biases(NoBiasExperts())

        assert gate_bias is None
        assert up_bias is None
        assert down_bias is None

    def test_extract_biases_with_biases(self):
        """Test _extract_biases with biases present (lines 493-500)."""

        class BiasExperts:
            def __init__(self):
                # gate_up is interleaved (num_experts, 2*intermediate)
                self.gate_up_proj_bias = mx.random.normal((4, 128))  # 64 each
                self.down_proj_bias = mx.random.normal((4, 32))

        experts = BiasExperts()
        gate_bias, up_bias, down_bias = MoETypeService._extract_biases(experts)

        assert gate_bias.shape == (4, 64)  # Even indices
        assert up_bias.shape == (4, 64)  # Odd indices
        assert down_bias.shape == (4, 32)

    def test_extract_biases_only_gate_up(self):
        """Test _extract_biases with only gate_up_proj_bias."""

        class PartialBiasExperts:
            def __init__(self):
                self.gate_up_proj_bias = mx.random.normal((4, 64))
                self.down_proj_bias = None

        experts = PartialBiasExperts()
        gate_bias, up_bias, down_bias = MoETypeService._extract_biases(experts)

        assert gate_bias is not None
        assert up_bias is not None
        assert down_bias is None
