"""Tests for MoE type detection (pseudo vs native classification)."""

import numpy as np
import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.moe import MoEType
from chuk_lazarus.introspection.moe.moe_type import (
    MoETypeAnalysis,
    MoETypeService,
    ProjectionRankAnalysis,
)


class TestProjectionRankAnalysis:
    """Tests for ProjectionRankAnalysis Pydantic model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(2880, 2880),
            max_rank=2880,
            effective_rank_95=1,
        )
        assert analysis.name == "gate"
        assert analysis.shape == (2880, 2880)
        assert analysis.max_rank == 2880
        assert analysis.effective_rank_95 == 1

    def test_rank_ratio_low(self):
        """Test rank ratio for pseudo-MoE (low rank)."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(2880, 2880),
            max_rank=2880,
            effective_rank_95=1,
        )
        assert analysis.rank_ratio == pytest.approx(1 / 2880, rel=0.01)

    def test_rank_ratio_high(self):
        """Test rank ratio for native-MoE (high rank)."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(1024, 2048),
            max_rank=1024,
            effective_rank_95=755,
        )
        assert analysis.rank_ratio == pytest.approx(755 / 1024, rel=0.01)

    def test_rank_ratio_zero_max_rank(self):
        """Test rank ratio when max_rank is invalid."""
        # This shouldn't happen in practice but test edge case
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(0, 0),
            max_rank=1,  # Minimum valid
            effective_rank_95=0,
        )
        assert analysis.rank_ratio == 0.0

    def test_compression_ratio_high_rank(self):
        """Test compression ratio for low-rank projection."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(2880, 2880),
            max_rank=2880,
            effective_rank_95=1,
        )
        # Original: 2880 * 2880 = 8,294,400
        # Factorized: 1 * (2880 + 2880) = 5,760
        # Ratio: 8,294,400 / 5,760 = 1440
        assert analysis.compression_ratio > 1000

    def test_compression_ratio_full_rank(self):
        """Test compression ratio for full-rank projection."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(1024, 2048),
            max_rank=1024,
            effective_rank_95=755,
        )
        # Original: 1024 * 2048 = 2,097,152
        # Factorized: 755 * (1024 + 2048) = 2,319,360
        # Ratio: 2,097,152 / 2,319,360 < 1 (no compression)
        assert analysis.compression_ratio < 1

    def test_compression_ratio_zero_rank(self):
        """Test compression ratio when effective rank is 0."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(100, 100),
            max_rank=100,
            effective_rank_95=0,
        )
        assert analysis.compression_ratio == float("inf")

    def test_frozen_model(self):
        """Test that model is frozen (immutable)."""
        analysis = ProjectionRankAnalysis(
            name="gate",
            shape=(100, 100),
            max_rank=100,
            effective_rank_95=10,
        )
        with pytest.raises(ValidationError):
            analysis.name = "up"


class TestMoETypeAnalysis:
    """Tests for MoETypeAnalysis Pydantic model."""

    @pytest.fixture
    def pseudo_moe_analysis(self):
        """Create a pseudo-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="openai/gpt-oss-20b",
            layer_idx=0,
            num_experts=32,
            moe_type=MoEType.UPCYCLED,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(2880, 2880), max_rank=2880, effective_rank_95=1
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(2880, 2880), max_rank=2880, effective_rank_95=337
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2880, 2880), max_rank=2880, effective_rank_95=206
            ),
            mean_cosine_similarity=0.418,
            std_cosine_similarity=0.163,
        )

    @pytest.fixture
    def native_moe_analysis(self):
        """Create a native-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="allenai/OLMoE-1B-7B-0924",
            layer_idx=0,
            num_experts=64,
            moe_type=MoEType.PRETRAINED_MOE,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(1024, 2048), max_rank=1024, effective_rank_95=755
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(1024, 2048), max_rank=1024, effective_rank_95=772
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2048, 1024), max_rank=1024, effective_rank_95=785
            ),
            mean_cosine_similarity=0.0,
            std_cosine_similarity=0.001,
        )

    def test_pseudo_moe_is_compressible(self, pseudo_moe_analysis):
        """Test that pseudo-MoE is marked compressible."""
        assert pseudo_moe_analysis.is_compressible is True

    def test_native_moe_not_compressible(self, native_moe_analysis):
        """Test that native-MoE is not marked compressible."""
        assert native_moe_analysis.is_compressible is False

    def test_pseudo_moe_compression_estimate(self, pseudo_moe_analysis):
        """Test compression estimate for pseudo-MoE."""
        # Should have significant compression
        assert pseudo_moe_analysis.estimated_compression > 3

    def test_native_moe_compression_estimate(self, native_moe_analysis):
        """Test compression estimate for native-MoE (close to 1)."""
        # Should have minimal/no compression
        assert native_moe_analysis.estimated_compression < 1.5

    def test_model_dump_json(self, pseudo_moe_analysis):
        """Test JSON serialization."""
        json_str = pseudo_moe_analysis.model_dump_json()
        assert "openai/gpt-oss-20b" in json_str
        assert "upcycled" in json_str


class TestMoETypeServiceClassify:
    """Tests for MoETypeService._classify method."""

    def test_classify_pseudo_moe(self):
        """Test classification of pseudo-MoE (upcycled)."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.01,  # < 5%
            similarity=0.4,  # > 0.25
        )
        assert moe_type == MoEType.UPCYCLED
        assert confidence > 0.5  # Should have reasonable confidence

    def test_classify_native_moe(self):
        """Test classification of native-MoE (pretrained)."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.74,  # > 50%
            similarity=0.0,  # < 0.10
        )
        assert moe_type == MoEType.PRETRAINED_MOE
        assert confidence > 0.5  # Should have reasonable confidence

    def test_classify_unknown_high_rank_high_similarity(self):
        """Test classification returns unknown for ambiguous metrics."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.6,  # High rank
            similarity=0.3,  # High similarity
        )
        assert moe_type == MoEType.UNKNOWN

    def test_classify_unknown_low_rank_low_similarity(self):
        """Test classification returns unknown for edge case."""
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.02,  # Low rank
            similarity=0.05,  # Low similarity
        )
        assert moe_type == MoEType.UNKNOWN

    def test_classify_edge_pseudo_threshold(self):
        """Test classification at pseudo threshold boundary."""
        # Just under threshold
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.049,
            similarity=0.26,
        )
        assert moe_type == MoEType.UPCYCLED

    def test_classify_edge_native_threshold(self):
        """Test classification at native threshold boundary."""
        # Just above threshold - this is an ambiguous case (low similarity but medium rank)
        moe_type, confidence, signals = MoETypeService._classify(
            gate_rank_ratio=0.51,
            similarity=0.09,
        )
        # With the new signal-based classification, this ambiguous case may be UNKNOWN
        assert moe_type in (MoEType.PRETRAINED_MOE, MoEType.UNKNOWN)


class TestMoETypeServiceHelpers:
    """Tests for MoETypeService helper methods."""

    def test_compute_effective_rank_simple(self):
        """Test effective rank computation."""
        # Singular values that decay quickly
        S = np.array([10.0, 1.0, 0.1, 0.01])
        rank = MoETypeService._compute_effective_rank(S)
        # 10^2 + 1^2 = 101 captures most variance
        assert rank <= 2

    def test_compute_effective_rank_uniform(self):
        """Test effective rank for uniform singular values."""
        # All equal singular values
        S = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        rank = MoETypeService._compute_effective_rank(S)
        # Need most/all to capture 95%
        assert rank >= 4

    def test_compute_effective_rank_zero(self):
        """Test effective rank for zero matrix."""
        S = np.array([0.0, 0.0, 0.0])
        rank = MoETypeService._compute_effective_rank(S)
        assert rank == 0

    def test_compute_effective_rank_single(self):
        """Test effective rank for rank-1 matrix."""
        S = np.array([10.0, 0.0, 0.0, 0.0])
        rank = MoETypeService._compute_effective_rank(S)
        assert rank == 1


class TestMoETypeServiceThresholds:
    """Tests for MoETypeService threshold constants."""

    def test_pseudo_rank_threshold(self):
        """Verify pseudo rank threshold is reasonable."""
        assert MoETypeService.PSEUDO_RANK_RATIO_THRESHOLD == 0.05

    def test_pseudo_similarity_threshold(self):
        """Verify pseudo similarity threshold is reasonable."""
        assert MoETypeService.PSEUDO_SIMILARITY_THRESHOLD == 0.25

    def test_native_rank_threshold(self):
        """Verify native rank threshold is reasonable."""
        assert MoETypeService.NATIVE_RANK_RATIO_THRESHOLD == 0.50

    def test_native_similarity_threshold(self):
        """Verify native similarity threshold is reasonable."""
        assert MoETypeService.NATIVE_SIMILARITY_THRESHOLD == 0.10

    def test_variance_threshold(self):
        """Verify variance threshold is 95%."""
        assert MoETypeService.VARIANCE_THRESHOLD == 0.95

    def test_max_experts_to_sample(self):
        """Verify expert sampling limit."""
        assert MoETypeService.MAX_EXPERTS_TO_SAMPLE == 8
