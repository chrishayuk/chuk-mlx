"""Tests for counterfactual intervention API."""

import pytest

from chuk_lazarus.introspection.interventions import (
    CausalTraceResult,
    ComponentTarget,
    FullCausalTrace,
    InterventionConfig,
    InterventionHook,
    InterventionResult,
    InterventionType,
    PatchingResult,
)

# =============================================================================
# Tests for Configuration Models
# =============================================================================


class TestInterventionType:
    """Tests for InterventionType enum."""

    def test_values(self):
        """Test enum values."""
        assert InterventionType.ZERO == "zero"
        assert InterventionType.PATCH == "patch"
        assert InterventionType.NOISE == "noise"
        assert InterventionType.STEER == "steer"
        assert InterventionType.SCALE == "scale"


class TestComponentTarget:
    """Tests for ComponentTarget enum."""

    def test_values(self):
        """Test enum values."""
        assert ComponentTarget.HIDDEN == "hidden"
        assert ComponentTarget.ATTENTION == "attention"
        assert ComponentTarget.MLP == "mlp"
        assert ComponentTarget.ATTENTION_HEAD == "attn_head"


class TestInterventionConfig:
    """Tests for InterventionConfig model."""

    def test_creation(self):
        """Test config creation."""
        config = InterventionConfig(
            intervention_type=InterventionType.PATCH,
            target=ComponentTarget.HIDDEN,
            layers=(5, 10, 15),
            positions=(-1, -2),
        )

        assert config.intervention_type == InterventionType.PATCH
        assert config.target == ComponentTarget.HIDDEN
        assert config.layers == (5, 10, 15)
        assert config.positions == (-1, -2)

    def test_default_values(self):
        """Test default values."""
        config = InterventionConfig()

        assert config.intervention_type == InterventionType.PATCH
        assert config.target == ComponentTarget.HIDDEN
        assert config.layers == ()
        assert config.positions == (-1,)
        assert config.noise_scale == 0.1
        assert config.scale_factor == 0.0

    def test_frozen(self):
        """Test config is frozen."""
        config = InterventionConfig()
        with pytest.raises(Exception):  # Pydantic validation error
            config.layers = (1, 2, 3)


# =============================================================================
# Tests for Result Models
# =============================================================================


class TestInterventionResult:
    """Tests for InterventionResult model."""

    def test_creation(self):
        """Test result creation."""
        result = InterventionResult(
            clean_output="The capital of France is Paris",
            intervened_output="The capital of France is Berlin",
            effect_size=0.5,
            kl_divergence=0.3,
        )

        assert result.clean_output == "The capital of France is Paris"
        assert result.intervened_output == "The capital of France is Berlin"
        assert result.effect_size == 0.5
        assert result.kl_divergence == 0.3

    def test_default_values(self):
        """Test default values."""
        result = InterventionResult(
            clean_output="test",
            intervened_output="test2",
        )

        assert result.clean_logits is None
        assert result.intervened_logits is None
        assert result.effect_size == 0.0
        assert result.kl_divergence is None
        assert result.intervention_config is None


class TestPatchingResult:
    """Tests for PatchingResult model."""

    def test_creation(self):
        """Test result creation."""
        result = PatchingResult(
            clean_prompt="The capital of France is",
            corrupt_prompt="The capital of Germany is",
            clean_output="Paris",
            corrupt_output="Berlin",
            patched_output="Paris",
            recovery_rate=0.9,
            effect_size=0.4,
            patched_layers=(10, 11, 12),
            patched_positions=(-1,),
        )

        assert result.clean_prompt == "The capital of France is"
        assert result.recovery_rate == 0.9
        assert result.patched_layers == (10, 11, 12)

    def test_default_values(self):
        """Test default values."""
        result = PatchingResult(
            clean_prompt="a",
            corrupt_prompt="b",
            clean_output="x",
            corrupt_output="y",
            patched_output="z",
        )

        assert result.recovery_rate == 0.0
        assert result.effect_size == 0.0
        assert result.patched_layers == ()
        assert result.patched_positions == ()


class TestCausalTraceResult:
    """Tests for CausalTraceResult model."""

    def test_creation(self):
        """Test result creation."""
        result = CausalTraceResult(
            prompt="The capital of France is",
            target_token="Paris",
            target_token_id=12345,
            layer_effects=(
                (0, 0.01),
                (5, 0.15),
                (10, 0.45),
                (15, 0.25),
            ),
            critical_layers=(10, 15, 5),
            peak_layer=10,
            peak_effect=0.45,
            baseline_prob=0.85,
        )

        assert result.prompt == "The capital of France is"
        assert result.target_token == "Paris"
        assert result.peak_layer == 10
        assert result.peak_effect == 0.45
        assert 10 in result.critical_layers

    def test_default_values(self):
        """Test default values."""
        result = CausalTraceResult(
            prompt="test",
            target_token="token",
            target_token_id=0,
        )

        assert result.layer_effects == ()
        assert result.critical_layers == ()
        assert result.peak_layer == 0
        assert result.peak_effect == 0.0
        assert result.baseline_prob == 0.0


class TestFullCausalTrace:
    """Tests for FullCausalTrace model."""

    def test_creation(self):
        """Test result creation."""
        result = FullCausalTrace(
            prompt="The capital is",
            target_token="Paris",
            tokens=("The", " capital", " is"),
            effects=(
                (0.1, 0.2, 0.1),
                (0.3, 0.8, 0.4),
                (0.2, 0.5, 0.3),
            ),
            critical_positions=(1, 2),
            critical_layers=(1, 2, 0),
        )

        assert result.prompt == "The capital is"
        assert len(result.tokens) == 3
        assert len(result.effects) == 3
        assert result.critical_positions[0] == 1

    def test_default_values(self):
        """Test default values."""
        result = FullCausalTrace(
            prompt="test",
            target_token="tok",
        )

        assert result.tokens == ()
        assert result.effects == ()
        assert result.critical_positions == ()
        assert result.critical_layers == ()


# =============================================================================
# Tests for InterventionHook
# =============================================================================


class TestInterventionHook:
    """Tests for InterventionHook class."""

    def test_zero_intervention(self):
        """Test zero intervention type."""
        import mlx.core as mx

        config = InterventionConfig(
            intervention_type=InterventionType.ZERO,
            layers=(0,),
            positions=(-1,),
        )

        hook = InterventionHook(config)

        # Create test tensor [batch=1, seq=3, hidden=4]
        h = mx.ones((1, 3, 4))
        result = hook(h, layer_idx=0)

        # Last position should be zeroed
        assert mx.allclose(result[:, -1, :], mx.zeros((1, 4)))
        # Other positions should be unchanged
        assert mx.allclose(result[:, 0, :], mx.ones((1, 4)))

    def test_scale_intervention(self):
        """Test scale intervention type."""
        import mlx.core as mx

        config = InterventionConfig(
            intervention_type=InterventionType.SCALE,
            layers=(0,),
            positions=(-1,),
            scale_factor=0.5,
        )

        hook = InterventionHook(config)

        h = mx.ones((1, 3, 4))
        result = hook(h, layer_idx=0)

        # Last position should be scaled by 0.5
        assert mx.allclose(result[:, -1, :], mx.ones((1, 4)) * 0.5)

    def test_layer_filtering(self):
        """Test that hook only applies to specified layers."""
        import mlx.core as mx

        config = InterventionConfig(
            intervention_type=InterventionType.ZERO,
            layers=(5,),  # Only layer 5
            positions=(-1,),
        )

        hook = InterventionHook(config)

        h = mx.ones((1, 3, 4))

        # Layer 0 - should not be modified
        result = hook(h, layer_idx=0)
        assert mx.allclose(result, h)

        # Layer 5 - should be modified
        result = hook(h, layer_idx=5)
        assert mx.allclose(result[:, -1, :], mx.zeros((1, 4)))

    def test_patch_intervention(self):
        """Test patch intervention type."""
        import mlx.core as mx

        # Create patch activations
        patch_acts = mx.full((1, 3, 4), 0.5)

        config = InterventionConfig(
            intervention_type=InterventionType.PATCH,
            layers=(0,),
            positions=(-1,),
        )

        hook = InterventionHook(config, patch_activations=patch_acts)

        h = mx.ones((1, 3, 4))
        result = hook(h, layer_idx=0)

        # Last position should be patched with 0.5
        assert mx.allclose(result[:, -1, :], mx.full((1, 4), 0.5))
        # Other positions unchanged
        assert mx.allclose(result[:, 0, :], mx.ones((1, 4)))

    def test_steer_intervention(self):
        """Test steer intervention type."""
        import mlx.core as mx

        # Create steering direction
        direction = mx.full((4,), 0.1)

        config = InterventionConfig(
            intervention_type=InterventionType.STEER,
            layers=(0,),
            positions=(-1,),
        )

        hook = InterventionHook(config, steering_direction=direction)

        h = mx.ones((1, 3, 4))
        result = hook(h, layer_idx=0)

        # Last position should have direction added
        expected = mx.ones((1, 4)) + 0.1
        assert mx.allclose(result[:, -1, :], expected)


# =============================================================================
# Integration test - would require a model
# =============================================================================


class TestCounterfactualIntervention:
    """Placeholder for integration tests."""

    def test_class_exists(self):
        """Test that the class can be imported."""
        from chuk_lazarus.introspection.interventions import CounterfactualIntervention

        assert CounterfactualIntervention is not None

    def test_convenience_functions_exist(self):
        """Test that convenience functions exist."""
        from chuk_lazarus.introspection.interventions import (
            patch_activations,
            trace_causal_path,
        )

        assert callable(patch_activations)
        assert callable(trace_causal_path)
