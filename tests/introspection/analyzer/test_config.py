"""Tests for analyzer config module."""

import pytest

from chuk_lazarus.introspection.analyzer.config import (
    AnalysisConfig,
    LayerStrategy,
    TrackStrategy,
)
from chuk_lazarus.introspection.hooks import PositionSelection


class TestLayerStrategy:
    """Tests for LayerStrategy enum."""

    def test_all_values_exist(self):
        """Verify all expected enum values."""
        assert LayerStrategy.ALL == "all"
        assert LayerStrategy.EVENLY_SPACED == "evenly_spaced"
        assert LayerStrategy.FIRST_LAST == "first_last"
        assert LayerStrategy.CUSTOM == "custom"

    def test_string_conversion(self):
        """Test string conversion."""
        assert str(LayerStrategy.ALL) == "LayerStrategy.ALL"
        assert LayerStrategy.ALL.value == "all"


class TestTrackStrategy:
    """Tests for TrackStrategy enum."""

    def test_all_values_exist(self):
        """Verify all expected enum values."""
        assert TrackStrategy.MANUAL == "manual"
        assert TrackStrategy.TOP_K_FINAL == "top_k_final"
        assert TrackStrategy.EMERGENT == "emergent"
        assert TrackStrategy.TOOL_TOKENS == "tool_tokens"


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AnalysisConfig()
        assert config.layer_strategy == LayerStrategy.EVENLY_SPACED
        assert config.layer_step == 4
        assert config.custom_layers is None
        assert config.position_strategy == PositionSelection.LAST
        assert config.top_k == 5
        assert config.track_tokens == []
        assert config.track_strategy == TrackStrategy.MANUAL
        assert config.compute_entropy is True
        assert config.compute_transitions is True
        assert config.compute_residual_decomposition is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.ALL,
            layer_step=2,
            custom_layers=[0, 5, 10],
            top_k=10,
            track_tokens=["hello", "world"],
            track_strategy=TrackStrategy.TOP_K_FINAL,
            compute_entropy=False,
        )
        assert config.layer_strategy == LayerStrategy.ALL
        assert config.layer_step == 2
        assert config.custom_layers == [0, 5, 10]
        assert config.top_k == 10
        assert config.track_tokens == ["hello", "world"]
        assert config.track_strategy == TrackStrategy.TOP_K_FINAL
        assert config.compute_entropy is False

    def test_validation_top_k_minimum(self):
        """Test top_k validation."""
        with pytest.raises(ValueError):
            AnalysisConfig(top_k=0)

    def test_validation_top_k_maximum(self):
        """Test top_k maximum validation."""
        with pytest.raises(ValueError):
            AnalysisConfig(top_k=101)

    def test_validation_layer_step_minimum(self):
        """Test layer_step validation."""
        with pytest.raises(ValueError):
            AnalysisConfig(layer_step=0)

    def test_config_immutability(self):
        """Test that config allows modification (not frozen)."""
        config = AnalysisConfig()
        # This should work since it's not frozen
        config.top_k = 10
        assert config.top_k == 10
