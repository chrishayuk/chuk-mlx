"""Tests for uncertainty Pydantic models."""

import numpy as np
import pytest

from chuk_lazarus.introspection.enums import ComputeStrategy, ConfidenceLevel
from chuk_lazarus.introspection.models.uncertainty import (
    CalibrationResult,
    MetacognitiveAnalysis,
    MetacognitiveResult,
    UncertaintyAnalysis,
    UncertaintyResult,
)


class TestMetacognitiveResult:
    """Tests for MetacognitiveResult model."""

    def test_instantiation_minimal(self):
        """Test creating metacognitive result with minimal fields."""
        result = MetacognitiveResult(
            problem="2 + 3 = ",
            decision_layer=5,
            decision_token="5",
            decision_prob=0.8,
            strategy=ComputeStrategy.DIRECT,
        )
        assert result.problem == "2 + 3 = "
        assert result.expected is None
        assert result.generated == ""
        assert result.decision_layer == 5
        assert result.decision_token == "5"
        assert result.decision_prob == 0.8
        assert result.strategy == ComputeStrategy.DIRECT
        assert result.is_digit is False
        assert result.correct_start is False
        assert result.final_token == ""
        assert result.final_prob == 0.0

    def test_instantiation_with_all_fields(self):
        """Test creating metacognitive result with all fields."""
        result = MetacognitiveResult(
            problem="123 * 456 = ",
            expected="56088",
            generated="Let me think step by step...",
            decision_layer=3,
            decision_token="L",
            decision_prob=0.7,
            strategy=ComputeStrategy.CHAIN_OF_THOUGHT,
            is_digit=False,
            correct_start=False,
            final_token="56088",
            final_prob=0.6,
        )
        assert result.expected == "56088"
        assert result.generated == "Let me think step by step..."
        assert result.strategy == ComputeStrategy.CHAIN_OF_THOUGHT
        assert result.is_digit is False
        assert result.correct_start is False
        assert result.final_token == "56088"
        assert result.final_prob == 0.6

    def test_default_values(self):
        """Test default values for optional fields."""
        result = MetacognitiveResult(
            problem="test",
            decision_layer=0,
            decision_token="x",
            decision_prob=0.5,
            strategy=ComputeStrategy.UNKNOWN,
        )
        assert result.expected is None
        assert result.generated == ""
        assert result.is_digit is False
        assert result.correct_start is False
        assert result.final_token == ""
        assert result.final_prob == 0.0

    def test_all_compute_strategies(self):
        """Test creating result with all compute strategies."""
        for strategy in ComputeStrategy:
            result = MetacognitiveResult(
                problem="test",
                decision_layer=0,
                decision_token="x",
                decision_prob=0.5,
                strategy=strategy,
            )
            assert result.strategy == strategy

    def test_direct_strategy_with_digit(self):
        """Test direct strategy with digit token."""
        result = MetacognitiveResult(
            problem="2 + 3 = ",
            expected="5",
            decision_layer=5,
            decision_token="5",
            decision_prob=0.9,
            strategy=ComputeStrategy.DIRECT,
            is_digit=True,
            correct_start=True,
        )
        assert result.is_digit is True
        assert result.correct_start is True


class TestMetacognitiveAnalysis:
    """Tests for MetacognitiveAnalysis model."""

    def test_instantiation_minimal(self):
        """Test creating metacognitive analysis with minimal fields."""
        analysis = MetacognitiveAnalysis(
            model_id="test-model",
            decision_layer=5,
            total_problems=10,
        )
        assert analysis.model_id == "test-model"
        assert analysis.decision_layer == 5
        assert analysis.total_problems == 10
        assert analysis.direct_count == 0
        assert analysis.cot_count == 0
        assert analysis.results == []

    def test_instantiation_with_all_fields(self):
        """Test creating metacognitive analysis with all fields."""
        results = [
            MetacognitiveResult(
                problem="2 + 3 = ",
                decision_layer=5,
                decision_token="5",
                decision_prob=0.9,
                strategy=ComputeStrategy.DIRECT,
                is_digit=True,
                correct_start=True,
            ),
            MetacognitiveResult(
                problem="123 * 456 = ",
                decision_layer=5,
                decision_token="L",
                decision_prob=0.7,
                strategy=ComputeStrategy.CHAIN_OF_THOUGHT,
            ),
        ]
        analysis = MetacognitiveAnalysis(
            model_id="test-model",
            decision_layer=5,
            total_problems=10,
            direct_count=6,
            cot_count=4,
            results=results,
        )
        assert analysis.direct_count == 6
        assert analysis.cot_count == 4
        assert len(analysis.results) == 2

    def test_direct_ratio_property(self):
        """Test direct_ratio computation."""
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=10,
            direct_count=7,
        )
        assert analysis.direct_ratio == 0.7

    def test_direct_ratio_zero_problems(self):
        """Test direct_ratio returns 0 when total_problems is 0."""
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=0,
            direct_count=0,
        )
        assert analysis.direct_ratio == 0.0

    def test_direct_ratio_all_direct(self):
        """Test direct_ratio returns 1.0 when all problems are direct."""
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=10,
            direct_count=10,
        )
        assert analysis.direct_ratio == 1.0

    def test_direct_accuracy_property(self):
        """Test direct_accuracy computation."""
        results = [
            MetacognitiveResult(
                problem="2 + 3",
                decision_layer=5,
                decision_token="5",
                decision_prob=0.9,
                strategy=ComputeStrategy.DIRECT,
                correct_start=True,
            ),
            MetacognitiveResult(
                problem="4 + 5",
                decision_layer=5,
                decision_token="9",
                decision_prob=0.85,
                strategy=ComputeStrategy.DIRECT,
                correct_start=True,
            ),
            MetacognitiveResult(
                problem="7 + 8",
                decision_layer=5,
                decision_token="14",
                decision_prob=0.6,
                strategy=ComputeStrategy.DIRECT,
                correct_start=False,
            ),
        ]
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=3,
            direct_count=3,
            results=results,
        )
        # 2 out of 3 direct answers are correct
        assert analysis.direct_accuracy == pytest.approx(2.0 / 3.0)

    def test_direct_accuracy_no_direct_results(self):
        """Test direct_accuracy returns 0 when no direct results."""
        results = [
            MetacognitiveResult(
                problem="123 * 456",
                decision_layer=5,
                decision_token="L",
                decision_prob=0.7,
                strategy=ComputeStrategy.CHAIN_OF_THOUGHT,
            ),
        ]
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=1,
            cot_count=1,
            results=results,
        )
        assert analysis.direct_accuracy == 0.0

    def test_direct_accuracy_all_correct(self):
        """Test direct_accuracy returns 1.0 when all direct answers correct."""
        results = [
            MetacognitiveResult(
                problem=f"{i} + {i}",
                decision_layer=5,
                decision_token=str(i * 2),
                decision_prob=0.9,
                strategy=ComputeStrategy.DIRECT,
                correct_start=True,
            )
            for i in range(5)
        ]
        analysis = MetacognitiveAnalysis(
            model_id="test",
            decision_layer=5,
            total_problems=5,
            direct_count=5,
            results=results,
        )
        assert analysis.direct_accuracy == 1.0


class TestUncertaintyResult:
    """Tests for UncertaintyResult model."""

    def test_instantiation(self):
        """Test creating uncertainty result with all fields."""
        result = UncertaintyResult(
            prompt="2 + 3 = ",
            score=1.5,
            prediction=ConfidenceLevel.CONFIDENT,
            dist_to_compute=0.5,
            dist_to_refusal=2.0,
        )
        assert result.prompt == "2 + 3 = "
        assert result.score == 1.5
        assert result.prediction == ConfidenceLevel.CONFIDENT
        assert result.dist_to_compute == 0.5
        assert result.dist_to_refusal == 2.0

    def test_negative_score(self):
        """Test creating result with negative score (uncertain)."""
        result = UncertaintyResult(
            prompt="impossible problem",
            score=-1.2,
            prediction=ConfidenceLevel.UNCERTAIN,
            dist_to_compute=2.5,
            dist_to_refusal=1.3,
        )
        assert result.score == -1.2
        assert result.prediction == ConfidenceLevel.UNCERTAIN

    def test_all_confidence_levels(self):
        """Test creating result with all confidence levels."""
        for level in ConfidenceLevel:
            result = UncertaintyResult(
                prompt="test",
                score=0.0,
                prediction=level,
                dist_to_compute=1.0,
                dist_to_refusal=1.0,
            )
            assert result.prediction == level


class TestCalibrationResult:
    """Tests for CalibrationResult model."""

    def test_instantiation_minimal(self):
        """Test creating calibration result with minimal fields."""
        compute_center = np.random.randn(768)
        refusal_center = np.random.randn(768)
        separation = float(np.linalg.norm(compute_center - refusal_center))

        calibration = CalibrationResult(
            model_id="test-model",
            detection_layer=5,
            compute_center=compute_center,
            refusal_center=refusal_center,
            separation=separation,
        )
        assert calibration.model_id == "test-model"
        assert calibration.detection_layer == 5
        assert np.array_equal(calibration.compute_center, compute_center)
        assert np.array_equal(calibration.refusal_center, refusal_center)
        assert calibration.separation == separation
        assert calibration.working_prompts == []
        assert calibration.broken_prompts == []

    def test_instantiation_with_all_fields(self):
        """Test creating calibration result with all fields."""
        compute_center = np.random.randn(768)
        refusal_center = np.random.randn(768)
        working_prompts = ["2 + 3 = ", "4 * 5 = ", "10 / 2 = "]
        broken_prompts = ["divide by zero", "impossible", "error"]

        calibration = CalibrationResult(
            model_id="test-model",
            detection_layer=5,
            compute_center=compute_center,
            refusal_center=refusal_center,
            separation=3.5,
            working_prompts=working_prompts,
            broken_prompts=broken_prompts,
        )
        assert len(calibration.working_prompts) == 3
        assert len(calibration.broken_prompts) == 3
        assert calibration.separation == 3.5

    def test_numpy_arrays_allowed(self):
        """Test that numpy arrays are allowed via ConfigDict."""
        compute_center = np.random.randn(768)
        refusal_center = np.random.randn(768)

        calibration = CalibrationResult(
            model_id="test",
            detection_layer=0,
            compute_center=compute_center,
            refusal_center=refusal_center,
            separation=1.0,
        )
        assert isinstance(calibration.compute_center, np.ndarray)
        assert isinstance(calibration.refusal_center, np.ndarray)
        assert calibration.compute_center.shape == (768,)
        assert calibration.refusal_center.shape == (768,)


class TestUncertaintyAnalysis:
    """Tests for UncertaintyAnalysis model."""

    def test_instantiation_minimal(self):
        """Test creating uncertainty analysis with minimal fields."""
        analysis = UncertaintyAnalysis(
            model_id="test-model",
            detection_layer=5,
            separation=2.5,
        )
        assert analysis.model_id == "test-model"
        assert analysis.detection_layer == 5
        assert analysis.separation == 2.5
        assert analysis.results == []

    def test_instantiation_with_results(self):
        """Test creating uncertainty analysis with results."""
        results = [
            UncertaintyResult(
                prompt="2 + 3 = ",
                score=1.5,
                prediction=ConfidenceLevel.CONFIDENT,
                dist_to_compute=0.5,
                dist_to_refusal=2.0,
            ),
            UncertaintyResult(
                prompt="impossible",
                score=-1.2,
                prediction=ConfidenceLevel.UNCERTAIN,
                dist_to_compute=2.5,
                dist_to_refusal=1.3,
            ),
        ]
        analysis = UncertaintyAnalysis(
            model_id="test-model",
            detection_layer=5,
            separation=3.0,
            results=results,
        )
        assert len(analysis.results) == 2

    def test_confident_count_property(self):
        """Test confident_count computation."""
        results = [
            UncertaintyResult(
                prompt="p1",
                score=1.0,
                prediction=ConfidenceLevel.CONFIDENT,
                dist_to_compute=0.5,
                dist_to_refusal=1.5,
            ),
            UncertaintyResult(
                prompt="p2",
                score=1.2,
                prediction=ConfidenceLevel.CONFIDENT,
                dist_to_compute=0.4,
                dist_to_refusal=1.6,
            ),
            UncertaintyResult(
                prompt="p3",
                score=-0.5,
                prediction=ConfidenceLevel.UNCERTAIN,
                dist_to_compute=1.5,
                dist_to_refusal=1.0,
            ),
        ]
        analysis = UncertaintyAnalysis(
            model_id="test",
            detection_layer=5,
            separation=2.0,
            results=results,
        )
        assert analysis.confident_count == 2

    def test_uncertain_count_property(self):
        """Test uncertain_count computation."""
        results = [
            UncertaintyResult(
                prompt="p1",
                score=1.0,
                prediction=ConfidenceLevel.CONFIDENT,
                dist_to_compute=0.5,
                dist_to_refusal=1.5,
            ),
            UncertaintyResult(
                prompt="p2",
                score=-0.5,
                prediction=ConfidenceLevel.UNCERTAIN,
                dist_to_compute=1.5,
                dist_to_refusal=1.0,
            ),
            UncertaintyResult(
                prompt="p3",
                score=-0.8,
                prediction=ConfidenceLevel.UNCERTAIN,
                dist_to_compute=1.8,
                dist_to_refusal=1.0,
            ),
        ]
        analysis = UncertaintyAnalysis(
            model_id="test",
            detection_layer=5,
            separation=2.0,
            results=results,
        )
        assert analysis.uncertain_count == 2

    def test_count_properties_with_unknown(self):
        """Test count properties handle UNKNOWN confidence level."""
        results = [
            UncertaintyResult(
                prompt="p1",
                score=0.0,
                prediction=ConfidenceLevel.CONFIDENT,
                dist_to_compute=1.0,
                dist_to_refusal=1.0,
            ),
            UncertaintyResult(
                prompt="p2",
                score=0.0,
                prediction=ConfidenceLevel.UNCERTAIN,
                dist_to_compute=1.0,
                dist_to_refusal=1.0,
            ),
            UncertaintyResult(
                prompt="p3",
                score=0.0,
                prediction=ConfidenceLevel.UNKNOWN,
                dist_to_compute=1.0,
                dist_to_refusal=1.0,
            ),
        ]
        analysis = UncertaintyAnalysis(
            model_id="test",
            detection_layer=5,
            separation=2.0,
            results=results,
        )
        assert analysis.confident_count == 1
        assert analysis.uncertain_count == 1
        # UNKNOWN is not counted in either

    def test_empty_results(self):
        """Test analysis with no results."""
        analysis = UncertaintyAnalysis(
            model_id="test",
            detection_layer=5,
            separation=2.0,
        )
        assert analysis.confident_count == 0
        assert analysis.uncertain_count == 0
