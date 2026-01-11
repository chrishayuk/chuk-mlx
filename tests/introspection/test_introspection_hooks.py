"""Tests for introspection hooks."""

from chuk_lazarus.introspection.hooks import (
    CaptureConfig,
    CapturedState,
    LayerSelection,
    PositionSelection,
)


class TestLayerSelection:
    """Tests for LayerSelection enum."""

    def test_all(self):
        """Test ALL layer selection."""
        assert LayerSelection.ALL.value == "all"


class TestPositionSelection:
    """Tests for PositionSelection enum."""

    def test_all(self):
        """Test ALL position selection."""
        assert PositionSelection.ALL.value == "all"

    def test_last(self):
        """Test LAST position selection."""
        assert PositionSelection.LAST.value == "last"


class TestCaptureConfig:
    """Tests for CaptureConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CaptureConfig()
        assert config.layers == LayerSelection.ALL
        assert config.capture_hidden_states is True
        assert config.capture_attention_weights is False
        assert config.capture_attention_output is False
        assert config.capture_ffn_output is False
        assert config.capture_pre_norm is False
        assert config.positions == PositionSelection.LAST
        assert config.detach is True

    def test_specific_layers(self):
        """Test config with specific layers."""
        config = CaptureConfig(layers=[0, 4, 8, 12])
        assert config.layers == [0, 4, 8, 12]

    def test_capture_attention(self):
        """Test config with attention capture."""
        config = CaptureConfig(
            capture_attention_weights=True,
            capture_attention_output=True,
        )
        assert config.capture_attention_weights is True
        assert config.capture_attention_output is True

    def test_capture_ffn(self):
        """Test config with FFN capture."""
        config = CaptureConfig(capture_ffn_output=True)
        assert config.capture_ffn_output is True

    def test_capture_pre_norm(self):
        """Test config with pre-norm capture."""
        config = CaptureConfig(capture_pre_norm=True)
        assert config.capture_pre_norm is True

    def test_all_positions(self):
        """Test config with all positions."""
        config = CaptureConfig(positions=PositionSelection.ALL)
        assert config.positions == PositionSelection.ALL

    def test_specific_positions(self):
        """Test config with specific positions."""
        config = CaptureConfig(positions=[0, -1])
        assert config.positions == [0, -1]

    def test_no_detach(self):
        """Test config without detaching."""
        config = CaptureConfig(detach=False)
        assert config.detach is False


class TestCapturedState:
    """Tests for CapturedState."""

    def test_default_state(self):
        """Test default captured state."""
        state = CapturedState()
        assert state.hidden_states == {}
        assert state.attention_weights == {}
        assert state.attention_outputs == {}
        assert state.ffn_outputs == {}
        assert state.pre_norm_states == {}
