"""Tests for analyzer models module."""

import pytest

from chuk_lazarus.introspection.analyzer.models import (
    AnalysisResult,
    LayerPredictionResult,
    LayerTransition,
    ModelInfo,
    ResidualContribution,
    TokenEvolutionResult,
    TokenPrediction,
)


class TestTokenPrediction:
    """Tests for TokenPrediction."""

    def test_creation(self):
        """Test creating a token prediction."""
        pred = TokenPrediction(
            token="hello",
            token_id=123,
            probability=0.8,
            rank=1,
        )
        assert pred.token == "hello"
        assert pred.token_id == 123
        assert pred.probability == 0.8
        assert pred.rank == 1

    def test_frozen(self):
        """Test that model is frozen."""
        from pydantic import ValidationError

        pred = TokenPrediction(token="x", token_id=1, probability=0.5, rank=1)
        with pytest.raises(ValidationError):  # Frozen models raise ValidationError
            pred.token = "y"

    def test_probability_bounds(self):
        """Test probability validation."""
        with pytest.raises(ValueError):
            TokenPrediction(token="x", token_id=1, probability=1.5, rank=1)
        with pytest.raises(ValueError):
            TokenPrediction(token="x", token_id=1, probability=-0.1, rank=1)

    def test_rank_minimum(self):
        """Test rank minimum validation."""
        with pytest.raises(ValueError):
            TokenPrediction(token="x", token_id=1, probability=0.5, rank=0)


class TestLayerPredictionResult:
    """Tests for LayerPredictionResult."""

    def test_creation(self):
        """Test creating layer prediction result."""
        preds = [
            TokenPrediction(token="a", token_id=1, probability=0.5, rank=1),
            TokenPrediction(token="b", token_id=2, probability=0.3, rank=2),
        ]
        result = LayerPredictionResult(
            layer_idx=5,
            predictions=preds,
            entropy=1.5,
            entropy_normalized=0.2,
        )
        assert result.layer_idx == 5
        assert len(result.predictions) == 2
        assert result.entropy == 1.5
        assert result.entropy_normalized == 0.2

    def test_top_token(self):
        """Test top_token property."""
        preds = [TokenPrediction(token="top", token_id=1, probability=0.8, rank=1)]
        result = LayerPredictionResult(layer_idx=0, predictions=preds)
        assert result.top_token == "top"

    def test_top_token_empty(self):
        """Test top_token with no predictions."""
        result = LayerPredictionResult(layer_idx=0, predictions=[])
        assert result.top_token == ""

    def test_top_probability(self):
        """Test top_probability property."""
        preds = [TokenPrediction(token="x", token_id=1, probability=0.7, rank=1)]
        result = LayerPredictionResult(layer_idx=0, predictions=preds)
        assert result.top_probability == 0.7

    def test_is_confident(self):
        """Test is_confident property."""
        confident = LayerPredictionResult(layer_idx=0, predictions=[], entropy_normalized=0.1)
        not_confident = LayerPredictionResult(layer_idx=0, predictions=[], entropy_normalized=0.5)
        assert confident.is_confident is True
        assert not_confident.is_confident is False


class TestLayerTransition:
    """Tests for LayerTransition."""

    def test_creation(self):
        """Test creating layer transition."""
        trans = LayerTransition(
            from_layer=0,
            to_layer=1,
            kl_divergence=0.5,
            js_divergence=0.2,
            top_token_changed=True,
            entropy_delta=-0.1,
        )
        assert trans.from_layer == 0
        assert trans.to_layer == 1
        assert trans.kl_divergence == 0.5
        assert trans.js_divergence == 0.2
        assert trans.top_token_changed is True
        assert trans.entropy_delta == -0.1

    def test_is_significant(self):
        """Test is_significant property."""
        significant = LayerTransition(
            from_layer=0,
            to_layer=1,
            kl_divergence=0.5,
            js_divergence=0.15,
            top_token_changed=False,
            entropy_delta=0.0,
        )
        not_significant = LayerTransition(
            from_layer=0,
            to_layer=1,
            kl_divergence=0.1,
            js_divergence=0.05,
            top_token_changed=False,
            entropy_delta=0.0,
        )
        assert significant.is_significant is True
        assert not_significant.is_significant is False


class TestResidualContribution:
    """Tests for ResidualContribution."""

    def test_creation(self):
        """Test creating residual contribution."""
        contrib = ResidualContribution(
            layer_idx=5,
            attention_norm=1.5,
            ffn_norm=2.5,
            total_norm=4.0,
            attention_fraction=0.375,
            ffn_fraction=0.625,
        )
        assert contrib.layer_idx == 5
        assert contrib.attention_norm == 1.5
        assert contrib.ffn_norm == 2.5

    def test_dominant_component_ffn(self):
        """Test dominant_component when FFN dominates."""
        contrib = ResidualContribution(
            layer_idx=0,
            attention_norm=1.0,
            ffn_norm=2.0,
            total_norm=3.0,
            attention_fraction=0.33,
            ffn_fraction=0.67,
        )
        assert contrib.dominant_component == "ffn"

    def test_dominant_component_attention(self):
        """Test dominant_component when attention dominates."""
        contrib = ResidualContribution(
            layer_idx=0,
            attention_norm=2.0,
            ffn_norm=1.0,
            total_norm=3.0,
            attention_fraction=0.67,
            ffn_fraction=0.33,
        )
        assert contrib.dominant_component == "attention"


class TestTokenEvolutionResult:
    """Tests for TokenEvolutionResult."""

    def test_creation(self):
        """Test creating token evolution result."""
        result = TokenEvolutionResult(
            token="Paris",
            token_id=123,
            layer_probabilities={0: 0.1, 5: 0.5, 10: 0.9},
            layer_ranks={0: 50, 5: 5, 10: 1},
            emergence_layer=10,
        )
        assert result.token == "Paris"
        assert result.token_id == 123
        assert result.layer_probabilities[5] == 0.5
        assert result.layer_ranks[10] == 1
        assert result.emergence_layer == 10


class TestModelInfo:
    """Tests for ModelInfo."""

    def test_creation(self):
        """Test creating model info."""
        info = ModelInfo(
            model_id="test-model",
            num_layers=12,
            hidden_size=768,
            vocab_size=32000,
            has_tied_embeddings=True,
        )
        assert info.model_id == "test-model"
        assert info.num_layers == 12
        assert info.hidden_size == 768
        assert info.vocab_size == 32000
        assert info.has_tied_embeddings is True


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_creation(self):
        """Test creating analysis result."""
        final_pred = [TokenPrediction(token="x", token_id=1, probability=0.9, rank=1)]
        layer_preds = [
            LayerPredictionResult(
                layer_idx=0,
                predictions=[TokenPrediction(token="y", token_id=2, probability=0.5, rank=1)],
                entropy=1.0,
                entropy_normalized=0.5,
            ),
            LayerPredictionResult(
                layer_idx=11,
                predictions=[TokenPrediction(token="x", token_id=1, probability=0.9, rank=1)],
                entropy=0.5,
                entropy_normalized=0.2,
            ),
        ]
        result = AnalysisResult(
            prompt="test prompt",
            tokens=["test", " prompt"],
            num_layers=12,
            captured_layers=[0, 11],
            final_prediction=final_pred,
            layer_predictions=layer_preds,
        )
        assert result.prompt == "test prompt"
        assert result.num_layers == 12
        assert result.predicted_token == "x"
        assert result.predicted_probability == 0.9

    def test_decision_layer(self):
        """Test decision_layer property."""
        final_pred = [TokenPrediction(token="x", token_id=1, probability=0.9, rank=1)]
        layer_preds = [
            LayerPredictionResult(
                layer_idx=0,
                predictions=[TokenPrediction(token="y", token_id=2, probability=0.5, rank=1)],
                entropy_normalized=0.5,
            ),
            LayerPredictionResult(
                layer_idx=5,
                predictions=[TokenPrediction(token="x", token_id=1, probability=0.7, rank=1)],
                entropy_normalized=0.1,  # Confident
            ),
        ]
        result = AnalysisResult(
            prompt="test",
            tokens=["test"],
            num_layers=12,
            captured_layers=[0, 5],
            final_prediction=final_pred,
            layer_predictions=layer_preds,
        )
        assert result.decision_layer == 5

    def test_max_kl_transition(self):
        """Test max_kl_transition property."""
        transitions = [
            LayerTransition(
                from_layer=0,
                to_layer=1,
                kl_divergence=0.5,
                js_divergence=0.3,
                top_token_changed=False,
                entropy_delta=0.0,
            ),
            LayerTransition(
                from_layer=1,
                to_layer=2,
                kl_divergence=1.5,
                js_divergence=0.8,
                top_token_changed=True,
                entropy_delta=-0.1,
            ),
        ]
        result = AnalysisResult(
            prompt="test",
            tokens=["test"],
            num_layers=3,
            captured_layers=[0, 1, 2],
            final_prediction=[],
            layer_predictions=[],
            layer_transitions=transitions,
        )
        assert result.max_kl_transition.kl_divergence == 1.5
