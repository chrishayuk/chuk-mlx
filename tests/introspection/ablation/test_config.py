"""Tests for ablation config module."""

from chuk_lazarus.introspection.ablation.config import (
    AblationConfig,
    AblationType,
    ComponentType,
)


class TestAblationType:
    """Tests for AblationType enum."""

    def test_all_values_exist(self):
        """Verify all expected enum values."""
        assert AblationType.ZERO.value == "zero"
        assert AblationType.MEAN.value == "mean"
        assert AblationType.NOISE.value == "noise"

    def test_enum_count(self):
        """Verify expected number of ablation types."""
        assert len(AblationType) == 3


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_main_components(self):
        """Verify main component types."""
        assert ComponentType.MLP.value == "mlp"
        assert ComponentType.ATTENTION.value == "attention"
        assert ComponentType.BOTH.value == "both"

    def test_mlp_subcomponents(self):
        """Verify MLP subcomponent types."""
        assert ComponentType.MLP_GATE.value == "mlp_gate"
        assert ComponentType.MLP_UP.value == "mlp_up"
        assert ComponentType.MLP_DOWN.value == "mlp_down"

    def test_attention_subcomponents(self):
        """Verify attention subcomponent types."""
        assert ComponentType.ATTN_Q.value == "attn_q"
        assert ComponentType.ATTN_K.value == "attn_k"
        assert ComponentType.ATTN_V.value == "attn_v"
        assert ComponentType.ATTN_O.value == "attn_o"

    def test_enum_count(self):
        """Verify expected number of component types."""
        assert len(ComponentType) == 10


class TestAblationConfig:
    """Tests for AblationConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AblationConfig()
        assert config.ablation_type == AblationType.ZERO
        assert config.component == ComponentType.MLP
        assert config.max_new_tokens == 60
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = AblationConfig(
            ablation_type=AblationType.NOISE,
            component=ComponentType.ATTENTION,
            max_new_tokens=100,
            temperature=0.5,
        )
        assert config.ablation_type == AblationType.NOISE
        assert config.component == ComponentType.ATTENTION
        assert config.max_new_tokens == 100
        assert config.temperature == 0.5

    def test_all_component_types(self):
        """Test that all component types are valid."""
        for comp in ComponentType:
            config = AblationConfig(component=comp)
            assert config.component == comp

    def test_all_ablation_types(self):
        """Test that all ablation types are valid."""
        for abl in AblationType:
            config = AblationConfig(ablation_type=abl)
            assert config.ablation_type == abl
