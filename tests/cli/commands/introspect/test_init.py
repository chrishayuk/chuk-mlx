"""Tests for introspect CLI __init__.py imports."""

import pytest


class TestIntrospectImports:
    """Tests that all introspect commands can be imported."""

    def test_core_analysis_imports(self):
        """Test core analysis command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_analyze,
            introspect_compare,
            introspect_hooks,
        )

        assert callable(introspect_analyze)
        assert callable(introspect_compare)
        assert callable(introspect_hooks)

    def test_ablation_imports(self):
        """Test ablation command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_ablate,
            introspect_activation_diff,
            introspect_weight_diff,
        )

        assert callable(introspect_ablate)
        assert callable(introspect_weight_diff)
        assert callable(introspect_activation_diff)

    def test_steering_imports(self):
        """Test steering command imports."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        assert callable(introspect_steer)

    def test_neuron_imports(self):
        """Test neuron/direction command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_directions,
            introspect_neurons,
            introspect_operand_directions,
        )

        assert callable(introspect_neurons)
        assert callable(introspect_directions)
        assert callable(introspect_operand_directions)

    def test_arithmetic_imports(self):
        """Test arithmetic command imports."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        assert callable(introspect_arithmetic)

    def test_circuit_imports(self):
        """Test circuit command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_circuit_capture,
            introspect_circuit_compare,
            introspect_circuit_decode,
            introspect_circuit_invoke,
            introspect_circuit_test,
            introspect_circuit_view,
        )

        assert callable(introspect_circuit_capture)
        assert callable(introspect_circuit_invoke)
        assert callable(introspect_circuit_test)
        assert callable(introspect_circuit_view)
        assert callable(introspect_circuit_compare)
        assert callable(introspect_circuit_decode)

    def test_patching_imports(self):
        """Test patching command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_commutativity,
            introspect_patch,
        )

        assert callable(introspect_commutativity)
        assert callable(introspect_patch)

    def test_layer_imports(self):
        """Test layer command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_activation_cluster,
            introspect_early_layers,
            introspect_embedding,
            introspect_format_sensitivity,
            introspect_layer,
        )

        assert callable(introspect_layer)
        assert callable(introspect_format_sensitivity)
        assert callable(introspect_embedding)
        assert callable(introspect_early_layers)
        assert callable(introspect_activation_cluster)

    def test_memory_imports(self):
        """Test memory command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_memory,
            introspect_memory_inject,
        )

        assert callable(introspect_memory)
        assert callable(introspect_memory_inject)

    def test_generation_imports(self):
        """Test generation command imports."""
        from chuk_lazarus.cli.commands.introspect import introspect_generate

        assert callable(introspect_generate)

    def test_probing_imports(self):
        """Test probing command imports."""
        from chuk_lazarus.cli.commands.introspect import (
            introspect_metacognitive,
            introspect_probe,
            introspect_uncertainty,
        )

        assert callable(introspect_metacognitive)
        assert callable(introspect_probe)
        assert callable(introspect_uncertainty)

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from chuk_lazarus.cli.commands.introspect import __all__

        expected = [
            "introspect_analyze",
            "introspect_compare",
            "introspect_hooks",
            "introspect_ablate",
            "introspect_weight_diff",
            "introspect_activation_diff",
            "introspect_steer",
            "introspect_neurons",
            "introspect_directions",
            "introspect_operand_directions",
            "introspect_arithmetic",
            "introspect_commutativity",
            "introspect_patch",
            "introspect_circuit_capture",
            "introspect_circuit_invoke",
            "introspect_circuit_test",
            "introspect_circuit_view",
            "introspect_circuit_compare",
            "introspect_circuit_decode",
            "introspect_layer",
            "introspect_format_sensitivity",
            "introspect_embedding",
            "introspect_early_layers",
            "introspect_activation_cluster",
            "introspect_memory",
            "introspect_memory_inject",
            "introspect_generate",
            "introspect_metacognitive",
            "introspect_probe",
            "introspect_uncertainty",
        ]

        for name in expected:
            assert name in __all__, f"{name} not in __all__"

    def test_module_docstring(self):
        """Test module has docstring."""
        import chuk_lazarus.cli.commands.introspect as introspect_module

        assert introspect_module.__doc__ is not None
        assert "Introspection" in introspect_module.__doc__
