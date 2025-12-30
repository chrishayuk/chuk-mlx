"""Tests for steering config module."""

from chuk_lazarus.introspection.steering.config import (
    LegacySteeringConfig,
    SteeringConfig,
    SteeringMode,
)


class TestSteeringMode:
    """Tests for SteeringMode enum."""

    def test_all_values_exist(self):
        """Verify all expected enum values."""
        assert SteeringMode.NORMAL.value == "normal"
        assert SteeringMode.FORCE_TOOL.value == "force_tool"
        assert SteeringMode.PREVENT_TOOL.value == "prevent_tool"
        assert SteeringMode.BOOST_TOOL.value == "boost_tool"
        assert SteeringMode.SUPPRESS_TOOL.value == "suppress_tool"

    def test_enum_count(self):
        """Verify expected number of modes."""
        assert len(SteeringMode) == 5


class TestSteeringConfig:
    """Tests for SteeringConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = SteeringConfig()
        assert config.layers == [24]
        assert config.coefficient == 1.0
        assert config.position is None
        assert config.normalize_direction is True
        assert config.scale_by_activation_norm is False
        assert config.max_new_tokens == 50
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = SteeringConfig(
            layers=[0, 5, 10],
            coefficient=2.5,
            position=5,
            normalize_direction=False,
            scale_by_activation_norm=True,
            max_new_tokens=100,
            temperature=0.7,
        )
        assert config.layers == [0, 5, 10]
        assert config.coefficient == 2.5
        assert config.position == 5
        assert config.normalize_direction is False
        assert config.scale_by_activation_norm is True
        assert config.max_new_tokens == 100
        assert config.temperature == 0.7


class TestLegacySteeringConfig:
    """Tests for LegacySteeringConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = LegacySteeringConfig()
        assert config.mode == SteeringMode.NORMAL
        assert config.steering_scale == 1.0
        assert config.neuron_boost_scale == 5000.0
        assert config.use_kill_switch is False
        assert config.kill_switch_boost == 0.0
        assert config.tool_promoters == [803, 2036, 831]
        assert config.tool_suppressors == [1237, 821, 1347]

    def test_custom_values(self):
        """Test custom configuration."""
        config = LegacySteeringConfig(
            mode=SteeringMode.FORCE_TOOL,
            steering_scale=2.0,
            tool_promoters=[100, 200],
            tool_suppressors=[300],
        )
        assert config.mode == SteeringMode.FORCE_TOOL
        assert config.steering_scale == 2.0
        assert config.tool_promoters == [100, 200]
        assert config.tool_suppressors == [300]

    def test_post_init_default_promoters(self):
        """Test that post_init sets default promoters when None."""
        config = LegacySteeringConfig(tool_promoters=None)
        assert config.tool_promoters == [803, 2036, 831]

    def test_post_init_default_suppressors(self):
        """Test that post_init sets default suppressors when None."""
        config = LegacySteeringConfig(tool_suppressors=None)
        assert config.tool_suppressors == [1237, 821, 1347]
