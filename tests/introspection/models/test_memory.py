"""Tests for memory Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.enums import MemorizationLevel
from chuk_lazarus.introspection.models.facts import FactNeighborhood
from chuk_lazarus.introspection.models.memory import (
    AttractorNode,
    MemoryAnalysisResult,
    MemoryStats,
    RetrievalResult,
)


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_instantiation_minimal(self):
        """Test creating retrieval result with minimal fields."""
        result = RetrievalResult(
            query="2 * 3 = ",
            answer="6",
        )
        assert result.query == "2 * 3 = "
        assert result.answer == "6"
        assert result.category == ""
        assert result.predictions == []
        assert isinstance(result.neighborhood, FactNeighborhood)
        assert result.memorization_level == MemorizationLevel.NOT_MEMORIZED

    def test_instantiation_with_all_fields(self):
        """Test creating retrieval result with all fields."""
        predictions = [
            {"token": "6", "prob": 0.9, "rank": 1},
            {"token": "8", "prob": 0.05, "rank": 2},
        ]
        neighborhood = FactNeighborhood(
            correct_rank=1,
            correct_prob=0.9,
            same_category=[{"token": "12", "prob": 0.05}],
        )
        result = RetrievalResult(
            query="2 * 3 = ",
            answer="6",
            category="2x",
            predictions=predictions,
            neighborhood=neighborhood,
            memorization_level=MemorizationLevel.MEMORIZED,
        )
        assert result.category == "2x"
        assert len(result.predictions) == 2
        assert result.neighborhood.correct_rank == 1
        assert result.memorization_level == MemorizationLevel.MEMORIZED

    def test_default_values(self):
        """Test default values for optional fields."""
        result = RetrievalResult(query="test", answer="test")
        assert result.category == ""
        assert result.predictions == []
        assert result.memorization_level == MemorizationLevel.NOT_MEMORIZED

    def test_classify_memorization_memorized(self):
        """Test classify_memorization returns MEMORIZED for rank 1, high prob."""
        level = RetrievalResult.classify_memorization(rank=1, prob=0.5)
        assert level == MemorizationLevel.MEMORIZED

    def test_classify_memorization_memorized_threshold(self):
        """Test classify_memorization returns MEMORIZED at prob threshold."""
        level = RetrievalResult.classify_memorization(rank=1, prob=0.11)
        assert level == MemorizationLevel.MEMORIZED

    def test_classify_memorization_partial_rank1_low_prob(self):
        """Test classify_memorization returns PARTIAL for rank 1 but prob below memorized threshold."""
        # rank 1 with prob 0.09 is still rank <= 5 with prob > 0.01, so PARTIAL
        level = RetrievalResult.classify_memorization(rank=1, prob=0.09)
        assert level == MemorizationLevel.PARTIAL

    def test_classify_memorization_weak_rank1_very_low_prob(self):
        """Test classify_memorization returns WEAK for rank 1 with very low prob."""
        # rank 1 with prob 0.009 is still rank <= 15 with prob > 0.001, so WEAK
        level = RetrievalResult.classify_memorization(rank=1, prob=0.009)
        assert level == MemorizationLevel.WEAK

    def test_classify_memorization_partial(self):
        """Test classify_memorization returns PARTIAL for rank 2-5."""
        level = RetrievalResult.classify_memorization(rank=3, prob=0.05)
        assert level == MemorizationLevel.PARTIAL

    def test_classify_memorization_partial_threshold(self):
        """Test classify_memorization returns PARTIAL at prob threshold."""
        level = RetrievalResult.classify_memorization(rank=5, prob=0.011)
        assert level == MemorizationLevel.PARTIAL

    def test_classify_memorization_weak(self):
        """Test classify_memorization returns WEAK for rank 6-15."""
        level = RetrievalResult.classify_memorization(rank=10, prob=0.005)
        assert level == MemorizationLevel.WEAK

    def test_classify_memorization_weak_threshold(self):
        """Test classify_memorization returns WEAK at prob threshold."""
        level = RetrievalResult.classify_memorization(rank=15, prob=0.0011)
        assert level == MemorizationLevel.WEAK

    def test_classify_memorization_not_memorized_high_rank(self):
        """Test classify_memorization returns NOT_MEMORIZED for high rank."""
        level = RetrievalResult.classify_memorization(rank=20, prob=0.001)
        assert level == MemorizationLevel.NOT_MEMORIZED

    def test_classify_memorization_none_rank(self):
        """Test classify_memorization returns NOT_MEMORIZED for None rank."""
        level = RetrievalResult.classify_memorization(rank=None, prob=0.5)
        assert level == MemorizationLevel.NOT_MEMORIZED

    def test_classify_memorization_none_prob(self):
        """Test classify_memorization returns NOT_MEMORIZED for None prob."""
        level = RetrievalResult.classify_memorization(rank=1, prob=None)
        assert level == MemorizationLevel.NOT_MEMORIZED


class TestAttractorNode:
    """Tests for AttractorNode model."""

    def test_instantiation(self):
        """Test creating attractor node with all fields."""
        node = AttractorNode(
            answer="6",
            count=10,
            avg_probability=0.25,
        )
        assert node.answer == "6"
        assert node.count == 10
        assert node.avg_probability == 0.25

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            AttractorNode(answer="6", count=10)  # Missing avg_probability


class TestMemoryStats:
    """Tests for MemoryStats model."""

    def test_instantiation_with_defaults(self):
        """Test creating stats with default values."""
        stats = MemoryStats()
        assert stats.top1_correct == 0
        assert stats.top5_correct == 0
        assert stats.not_found == 0
        assert stats.total == 0
        assert stats.same_category_total == 0
        assert stats.same_category_alt_total == 0
        assert stats.other_answers_total == 0
        assert stats.non_answers_total == 0

    def test_instantiation_with_values(self):
        """Test creating stats with specific values."""
        stats = MemoryStats(
            top1_correct=8,
            top5_correct=10,
            not_found=2,
            total=12,
            same_category_total=15,
            same_category_alt_total=10,
            other_answers_total=5,
            non_answers_total=20,
        )
        assert stats.top1_correct == 8
        assert stats.top5_correct == 10
        assert stats.not_found == 2
        assert stats.total == 12

    def test_top1_accuracy_property(self):
        """Test top1_accuracy computation."""
        stats = MemoryStats(top1_correct=8, total=10)
        assert stats.top1_accuracy == 0.8

    def test_top1_accuracy_zero_total(self):
        """Test top1_accuracy returns 0 when total is 0."""
        stats = MemoryStats(top1_correct=0, total=0)
        assert stats.top1_accuracy == 0.0

    def test_top1_accuracy_perfect(self):
        """Test top1_accuracy returns 1.0 for perfect accuracy."""
        stats = MemoryStats(top1_correct=10, total=10)
        assert stats.top1_accuracy == 1.0

    def test_top5_accuracy_property(self):
        """Test top5_accuracy computation."""
        stats = MemoryStats(top5_correct=9, total=10)
        assert stats.top5_accuracy == 0.9

    def test_top5_accuracy_zero_total(self):
        """Test top5_accuracy returns 0 when total is 0."""
        stats = MemoryStats(top5_correct=0, total=0)
        assert stats.top5_accuracy == 0.0

    def test_top5_accuracy_greater_than_top1(self):
        """Test top5_accuracy can be greater than top1_accuracy."""
        stats = MemoryStats(top1_correct=7, top5_correct=9, total=10)
        assert stats.top5_accuracy > stats.top1_accuracy


class TestMemoryAnalysisResult:
    """Tests for MemoryAnalysisResult model."""

    def test_instantiation_minimal(self):
        """Test creating analysis result with minimal fields."""
        result = MemoryAnalysisResult(
            model_id="test-model",
            fact_type="multiplication",
            layer=5,
            num_facts=64,
        )
        assert result.model_id == "test-model"
        assert result.fact_type == "multiplication"
        assert result.layer == 5
        assert result.num_facts == 64
        assert isinstance(result.stats, MemoryStats)
        assert result.attractors == []
        assert result.results == []
        assert result.category_stats == {}
        assert result.asymmetries == []
        assert result.row_bias_count == 0
        assert result.col_bias_count == 0
        assert result.neutral_count == 0

    def test_instantiation_with_all_fields(self):
        """Test creating analysis result with all fields."""
        stats = MemoryStats(top1_correct=50, top5_correct=60, total=64)
        attractors = [
            AttractorNode(answer="6", count=10, avg_probability=0.2),
            AttractorNode(answer="12", count=8, avg_probability=0.15),
        ]
        retrieval_result = RetrievalResult(query="2*3=", answer="6")
        category_stats = {
            "2x": MemoryStats(top1_correct=8, total=8),
            "3x": MemoryStats(top1_correct=7, total=8),
        }
        asymmetries = [
            {"pair": ("2*3", "3*2"), "difficulty_diff": 0.5},
        ]

        result = MemoryAnalysisResult(
            model_id="test-model",
            fact_type="multiplication",
            layer=5,
            num_facts=64,
            stats=stats,
            attractors=attractors,
            results=[retrieval_result],
            category_stats=category_stats,
            asymmetries=asymmetries,
            row_bias_count=20,
            col_bias_count=15,
            neutral_count=29,
        )
        assert result.stats.top1_correct == 50
        assert len(result.attractors) == 2
        assert len(result.results) == 1
        assert len(result.category_stats) == 2
        assert len(result.asymmetries) == 1
        assert result.row_bias_count == 20
        assert result.col_bias_count == 15
        assert result.neutral_count == 29

    def test_default_values(self):
        """Test default values for optional fields."""
        result = MemoryAnalysisResult(
            model_id="test",
            fact_type="test",
            layer=0,
            num_facts=0,
        )
        assert isinstance(result.stats, MemoryStats)
        assert result.attractors == []
        assert result.results == []
        assert result.category_stats == {}
        assert result.asymmetries == []
        assert result.row_bias_count == 0
        assert result.col_bias_count == 0
        assert result.neutral_count == 0

    def test_with_multiple_results(self):
        """Test analysis result with multiple retrieval results."""
        results = [
            RetrievalResult(query=f"{i}*{j}=", answer=str(i * j))
            for i in range(2, 4)
            for j in range(2, 4)
        ]
        result = MemoryAnalysisResult(
            model_id="test",
            fact_type="multiplication",
            layer=5,
            num_facts=4,
            results=results,
        )
        assert len(result.results) == 4

    def test_category_stats_structure(self):
        """Test category_stats can hold multiple categories."""
        category_stats = {
            "2x": MemoryStats(top1_correct=8, total=8),
            "3x": MemoryStats(top1_correct=7, total=8),
            "4x": MemoryStats(top1_correct=6, total=8),
        }
        result = MemoryAnalysisResult(
            model_id="test",
            fact_type="multiplication",
            layer=5,
            num_facts=24,
            category_stats=category_stats,
        )
        assert len(result.category_stats) == 3
        assert result.category_stats["2x"].top1_accuracy == 1.0
        assert result.category_stats["3x"].top1_accuracy == 0.875

    def test_asymmetries_structure(self):
        """Test asymmetries list structure."""
        asymmetries = [
            {"pair": ("2*3", "3*2"), "rank_diff": 2, "prob_diff": 0.1},
            {"pair": ("4*5", "5*4"), "rank_diff": 0, "prob_diff": 0.01},
        ]
        result = MemoryAnalysisResult(
            model_id="test",
            fact_type="multiplication",
            layer=5,
            num_facts=64,
            asymmetries=asymmetries,
        )
        assert len(result.asymmetries) == 2
        assert result.asymmetries[0]["pair"] == ("2*3", "3*2")
