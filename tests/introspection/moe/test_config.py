"""Tests for MoE configuration."""

import pytest

from chuk_lazarus.introspection.moe.config import MoEAblationConfig, MoECaptureConfig


class TestMoECaptureConfig:
    """Tests for MoECaptureConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MoECaptureConfig()
        assert config.layers is None
        assert config.capture_router_logits is True
        assert config.capture_router_weights is True
        assert config.capture_selected_experts is True
        assert config.capture_expert_outputs is False
        assert config.compute_entropy is False
        assert config.compute_utilization is False

    def test_custom_layers(self):
        """Test custom layer selection."""
        config = MoECaptureConfig(layers=[0, 2, 4])
        assert config.layers == [0, 2, 4]

    def test_disable_router_logits(self):
        """Test disabling router logits capture."""
        config = MoECaptureConfig(capture_router_logits=False)
        assert config.capture_router_logits is False

    def test_disable_router_weights(self):
        """Test disabling router weights capture."""
        config = MoECaptureConfig(capture_router_weights=False)
        assert config.capture_router_weights is False

    def test_disable_selected_experts(self):
        """Test disabling selected experts capture."""
        config = MoECaptureConfig(capture_selected_experts=False)
        assert config.capture_selected_experts is False

    def test_enable_expert_outputs(self):
        """Test enabling expert outputs capture."""
        config = MoECaptureConfig(capture_expert_outputs=True)
        assert config.capture_expert_outputs is True

    def test_enable_entropy(self):
        """Test enabling entropy computation."""
        config = MoECaptureConfig(compute_entropy=True)
        assert config.compute_entropy is True

    def test_enable_utilization(self):
        """Test enabling utilization computation."""
        config = MoECaptureConfig(compute_utilization=True)
        assert config.compute_utilization is True

    def test_config_is_frozen(self):
        """Test config is frozen (immutable)."""
        config = MoECaptureConfig()
        with pytest.raises(Exception):  # ValidationError for frozen model
            config.capture_router_logits = False


class TestMoEAblationConfig:
    """Tests for MoEAblationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MoEAblationConfig()
        assert config.target_layers is None
        assert config.ablation_method == "zero"
        assert config.preserve_scale is True

    def test_custom_target_layers(self):
        """Test custom target layer selection."""
        config = MoEAblationConfig(target_layers=[1, 3, 5])
        assert config.target_layers == [1, 3, 5]

    def test_mean_ablation_method(self):
        """Test mean ablation method."""
        config = MoEAblationConfig(ablation_method="mean")
        assert config.ablation_method == "mean"

    def test_random_ablation_method(self):
        """Test random ablation method."""
        config = MoEAblationConfig(ablation_method="random")
        assert config.ablation_method == "random"

    def test_disable_preserve_scale(self):
        """Test disabling preserve scale."""
        config = MoEAblationConfig(preserve_scale=False)
        assert config.preserve_scale is False

    def test_config_is_frozen(self):
        """Test config is frozen (immutable)."""
        config = MoEAblationConfig()
        with pytest.raises(Exception):
            config.ablation_method = "mean"
