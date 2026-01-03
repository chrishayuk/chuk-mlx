"""Tests for patching Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.enums import CommutativityLevel, PatchEffect
from chuk_lazarus.introspection.models.patching import (
    CommutativityPair,
    CommutativityResult,
    PatchingLayerResult,
    PatchingResult,
)


class TestCommutativityPair:
    """Tests for CommutativityPair model."""

    def test_instantiation(self):
        """Test creating commutativity pair with all fields."""
        pair = CommutativityPair(
            prompt_a="2*3=",
            prompt_b="3*2=",
            similarity=0.995,
        )
        assert pair.prompt_a == "2*3="
        assert pair.prompt_b == "3*2="
        assert pair.similarity == 0.995

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            CommutativityPair(prompt_a="2*3=", prompt_b="3*2=")  # Missing similarity


class TestCommutativityResult:
    """Tests for CommutativityResult model."""

    def test_instantiation_minimal(self):
        """Test creating commutativity result with minimal fields."""
        result = CommutativityResult(
            model_id="test-model",
            layer=5,
            num_pairs=10,
            mean_similarity=0.95,
            std_similarity=0.02,
            min_similarity=0.90,
            max_similarity=0.99,
        )
        assert result.model_id == "test-model"
        assert result.layer == 5
        assert result.num_pairs == 10
        assert result.mean_similarity == 0.95
        assert result.std_similarity == 0.02
        assert result.min_similarity == 0.90
        assert result.max_similarity == 0.99
        assert result.pairs == []

    def test_instantiation_with_pairs(self):
        """Test creating commutativity result with pairs."""
        pairs = [
            CommutativityPair(prompt_a="2*3=", prompt_b="3*2=", similarity=0.998),
            CommutativityPair(prompt_a="4*5=", prompt_b="5*4=", similarity=0.992),
        ]
        result = CommutativityResult(
            model_id="test-model",
            layer=5,
            num_pairs=2,
            mean_similarity=0.995,
            std_similarity=0.003,
            min_similarity=0.992,
            max_similarity=0.998,
            pairs=pairs,
        )
        assert len(result.pairs) == 2
        assert result.pairs[0].similarity == 0.998

    def test_level_property_perfect(self):
        """Test level property returns PERFECT for >0.999 similarity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.9995,
            std_similarity=0.0001,
            min_similarity=0.999,
            max_similarity=1.0,
        )
        assert result.level == CommutativityLevel.PERFECT

    def test_level_property_high(self):
        """Test level property returns HIGH for >0.99 similarity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.995,
            std_similarity=0.002,
            min_similarity=0.99,
            max_similarity=0.999,
        )
        assert result.level == CommutativityLevel.HIGH

    def test_level_property_moderate(self):
        """Test level property returns MODERATE for >0.9 similarity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.95,
            std_similarity=0.02,
            min_similarity=0.90,
            max_similarity=0.98,
        )
        assert result.level == CommutativityLevel.MODERATE

    def test_level_property_low(self):
        """Test level property returns LOW for <0.9 similarity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.85,
            std_similarity=0.05,
            min_similarity=0.75,
            max_similarity=0.89,
        )
        assert result.level == CommutativityLevel.LOW

    def test_interpretation_property_perfect(self):
        """Test interpretation property for perfect commutativity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.9995,
            std_similarity=0.0001,
            min_similarity=0.999,
            max_similarity=1.0,
        )
        assert "Perfect commutativity" in result.interpretation
        assert "lookup table" in result.interpretation
        assert "memorization" in result.interpretation

    def test_interpretation_property_high(self):
        """Test interpretation property for high commutativity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.995,
            std_similarity=0.002,
            min_similarity=0.99,
            max_similarity=0.999,
        )
        assert "High commutativity" in result.interpretation
        assert "lookup table" in result.interpretation

    def test_interpretation_property_moderate(self):
        """Test interpretation property for moderate commutativity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.95,
            std_similarity=0.02,
            min_similarity=0.90,
            max_similarity=0.98,
        )
        assert "Moderate commutativity" in result.interpretation
        assert "Partial lookup table" in result.interpretation

    def test_interpretation_property_low(self):
        """Test interpretation property for low commutativity."""
        result = CommutativityResult(
            model_id="test",
            layer=0,
            num_pairs=10,
            mean_similarity=0.85,
            std_similarity=0.05,
            min_similarity=0.75,
            max_similarity=0.89,
        )
        assert "Low commutativity" in result.interpretation
        assert "different algorithms" in result.interpretation


class TestPatchingLayerResult:
    """Tests for PatchingLayerResult model."""

    def test_instantiation_minimal(self):
        """Test creating layer result with minimal fields."""
        result = PatchingLayerResult(
            layer=5,
            top_token="6",
            top_prob=0.95,
            baseline_token="7",
            baseline_prob=0.85,
            effect=PatchEffect.TRANSFERRED,
        )
        assert result.layer == 5
        assert result.top_token == "6"
        assert result.top_prob == 0.95
        assert result.baseline_token == "7"
        assert result.baseline_prob == 0.85
        assert result.effect == PatchEffect.TRANSFERRED
        assert result.notes == ""

    def test_instantiation_with_notes(self):
        """Test creating layer result with notes."""
        result = PatchingLayerResult(
            layer=5,
            top_token="6",
            top_prob=0.95,
            baseline_token="7",
            baseline_prob=0.85,
            effect=PatchEffect.TRANSFERRED,
            notes="Strong transfer",
        )
        assert result.notes == "Strong transfer"

    def test_changed_property_true(self):
        """Test changed property returns True when prediction changed."""
        result = PatchingLayerResult(
            layer=5,
            top_token="6",
            top_prob=0.95,
            baseline_token="7",
            baseline_prob=0.85,
            effect=PatchEffect.TRANSFERRED,
        )
        assert result.changed is True

    def test_changed_property_false(self):
        """Test changed property returns False when prediction unchanged."""
        result = PatchingLayerResult(
            layer=5,
            top_token="6",
            top_prob=0.95,
            baseline_token="6",
            baseline_prob=0.85,
            effect=PatchEffect.NO_CHANGE,
        )
        assert result.changed is False

    def test_all_patch_effects(self):
        """Test creating layer result with all patch effects."""
        for effect in PatchEffect:
            result = PatchingLayerResult(
                layer=0,
                top_token="a",
                top_prob=0.5,
                baseline_token="b",
                baseline_prob=0.5,
                effect=effect,
            )
            assert result.effect == effect


class TestPatchingResult:
    """Tests for PatchingResult model."""

    def test_instantiation_minimal(self):
        """Test creating patching result with minimal fields."""
        result = PatchingResult(
            model_id="test-model",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
        )
        assert result.model_id == "test-model"
        assert result.source_prompt == "2*3="
        assert result.target_prompt == "4*5="
        assert result.baseline_token == "20"
        assert result.baseline_prob == 0.9
        assert result.source_answer is None
        assert result.target_answer is None
        assert result.blend == 1.0
        assert result.layers == []
        assert result.layer_results == []

    def test_instantiation_with_all_fields(self):
        """Test creating patching result with all fields."""
        layer_results = [
            PatchingLayerResult(
                layer=5,
                top_token="6",
                top_prob=0.8,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
            PatchingLayerResult(
                layer=6,
                top_token="6",
                top_prob=0.85,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
        ]
        result = PatchingResult(
            model_id="test-model",
            source_prompt="2*3=",
            target_prompt="4*5=",
            source_answer="6",
            target_answer="20",
            blend=0.5,
            layers=[5, 6, 7],
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert result.source_answer == "6"
        assert result.target_answer == "20"
        assert result.blend == 0.5
        assert result.layers == [5, 6, 7]
        assert len(result.layer_results) == 2

    def test_transferred_layers_property(self):
        """Test transferred_layers property returns correct layers."""
        layer_results = [
            PatchingLayerResult(
                layer=3,
                top_token="6",
                top_prob=0.5,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
            PatchingLayerResult(
                layer=4,
                top_token="15",
                top_prob=0.4,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.CHANGED,
            ),
            PatchingLayerResult(
                layer=5,
                top_token="6",
                top_prob=0.8,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
        ]
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert result.transferred_layers == [3, 5]

    def test_transferred_layers_empty(self):
        """Test transferred_layers property returns empty list when no transfers."""
        layer_results = [
            PatchingLayerResult(
                layer=5,
                top_token="20",
                top_prob=0.9,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.NO_CHANGE,
            ),
        ]
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert result.transferred_layers == []

    def test_any_transfer_property_true(self):
        """Test any_transfer property returns True when transfers exist."""
        layer_results = [
            PatchingLayerResult(
                layer=5,
                top_token="6",
                top_prob=0.8,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
        ]
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert result.any_transfer is True

    def test_any_transfer_property_false(self):
        """Test any_transfer property returns False when no transfers."""
        layer_results = [
            PatchingLayerResult(
                layer=5,
                top_token="20",
                top_prob=0.9,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.NO_CHANGE,
            ),
        ]
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert result.any_transfer is False

    def test_default_blend_value(self):
        """Test default blend value is 1.0."""
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
        )
        assert result.blend == 1.0

    def test_custom_blend_value(self):
        """Test custom blend value is preserved."""
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            blend=0.7,
        )
        assert result.blend == 0.7

    def test_multiple_layer_results_with_different_effects(self):
        """Test multiple layer results with different patch effects."""
        layer_results = [
            PatchingLayerResult(
                layer=1,
                top_token="20",
                top_prob=0.9,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.NO_CHANGE,
            ),
            PatchingLayerResult(
                layer=2,
                top_token="6",
                top_prob=0.7,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.TRANSFERRED,
            ),
            PatchingLayerResult(
                layer=3,
                top_token="20",
                top_prob=0.85,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.STILL_TARGET,
            ),
            PatchingLayerResult(
                layer=4,
                top_token="15",
                top_prob=0.5,
                baseline_token="20",
                baseline_prob=0.9,
                effect=PatchEffect.CHANGED,
            ),
        ]
        result = PatchingResult(
            model_id="test",
            source_prompt="2*3=",
            target_prompt="4*5=",
            baseline_token="20",
            baseline_prob=0.9,
            layer_results=layer_results,
        )
        assert len(result.layer_results) == 4
        assert result.transferred_layers == [2]
        assert result.any_transfer is True
