"""Tests for virtual_experts/registry.py to improve coverage."""

import mlx.core as mx
import pytest

from chuk_lazarus.inference.virtual_experts.base import VirtualExpertPlugin
from chuk_lazarus.inference.virtual_experts.registry import (
    VirtualExpertRegistry,
    get_default_registry,
    reset_default_registry,
)


class TestPluginForTests(VirtualExpertPlugin):
    """Test plugin for registry tests."""

    name = "test"
    description = "Test plugin"

    def can_handle(self, prompt: str) -> bool:
        return "test" in prompt.lower()

    def execute(self, prompt: str) -> str:
        return "test_result"

    def get_calibration_prompts(self):
        return ["test 1", "test 2"], ["hello", "world"]


class TestVirtualExpertRegistry:
    """Tests for VirtualExpertRegistry class."""

    def test_init_empty(self):
        """Test empty registry initialization."""
        registry = VirtualExpertRegistry()
        assert len(registry) == 0
        assert registry.plugin_names == []

    def test_register_plugin(self):
        """Test plugin registration."""
        registry = VirtualExpertRegistry()
        plugin = TestPluginForTests()

        registry.register(plugin)

        assert len(registry) == 1
        assert "test" in registry
        assert registry.get("test") is plugin

    def test_register_duplicate_raises(self):
        """Test registering duplicate plugin raises error."""
        registry = VirtualExpertRegistry()
        plugin = TestPluginForTests()

        registry.register(plugin)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(plugin)

    def test_unregister_plugin(self):
        """Test plugin unregistration."""
        registry = VirtualExpertRegistry()
        plugin = TestPluginForTests()

        registry.register(plugin)
        assert "test" in registry

        registry.unregister("test")
        assert "test" not in registry

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent plugin (line 63)."""
        registry = VirtualExpertRegistry()

        # Should not raise
        registry.unregister("nonexistent")

    def test_unregister_with_calibration_data(self):
        """Test unregister also removes calibration data (line 63)."""
        registry = VirtualExpertRegistry()
        plugin = TestPluginForTests()

        registry.register(plugin)
        registry.set_calibration_data(
            "test",
            [mx.zeros((64,))],
            [mx.zeros((64,))],
        )

        registry.unregister("test")

        assert "test" not in registry
        assert registry.get_calibration_data("test") is None

    def test_get_nonexistent(self):
        """Test getting nonexistent plugin."""
        registry = VirtualExpertRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all_sorted_by_priority(self):
        """Test get_all returns plugins sorted by priority."""
        registry = VirtualExpertRegistry()

        class LowPriorityPlugin(VirtualExpertPlugin):
            name = "low"
            description = "Low priority"
            priority = 1

            def can_handle(self, prompt):
                return False

            def execute(self, prompt):
                return ""

            def get_calibration_prompts(self):
                return [], []

        class HighPriorityPlugin(VirtualExpertPlugin):
            name = "high"
            description = "High priority"
            priority = 10

            def can_handle(self, prompt):
                return False

            def execute(self, prompt):
                return ""

            def get_calibration_prompts(self):
                return [], []

        registry.register(LowPriorityPlugin())
        registry.register(HighPriorityPlugin())

        plugins = registry.get_all()
        assert plugins[0].name == "high"
        assert plugins[1].name == "low"

    def test_find_handler_found(self):
        """Test finding handler that exists."""
        registry = VirtualExpertRegistry()

        class MathPlugin(VirtualExpertPlugin):
            name = "math"
            description = "Math"

            def can_handle(self, prompt):
                return "+" in prompt

            def execute(self, prompt):
                return "result"

            def get_calibration_prompts(self):
                return [], []

        registry.register(MathPlugin())

        handler = registry.find_handler("2 + 2")
        assert handler is not None
        assert handler.name == "math"

    def test_find_handler_not_found(self):
        """Test finding handler when none matches."""
        registry = VirtualExpertRegistry()

        class MathPlugin(VirtualExpertPlugin):
            name = "math"
            description = "Math"

            def can_handle(self, prompt):
                return "+" in prompt

            def execute(self, prompt):
                return "result"

            def get_calibration_prompts(self):
                return [], []

        registry.register(MathPlugin())

        handler = registry.find_handler("hello world")
        assert handler is None

    def test_set_calibration_data(self):
        """Test setting calibration data (line 105)."""
        registry = VirtualExpertRegistry()

        positive = [mx.random.normal((64,)) for _ in range(3)]
        negative = [mx.random.normal((64,)) for _ in range(3)]

        registry.set_calibration_data("test", positive, negative)

        data = registry.get_calibration_data("test")
        assert data is not None
        pos, neg = data
        assert len(pos) == 3
        assert len(neg) == 3

    def test_get_calibration_data_nonexistent(self):
        """Test getting calibration data for nonexistent plugin (line 112)."""
        registry = VirtualExpertRegistry()
        assert registry.get_calibration_data("nonexistent") is None

    def test_plugin_names(self):
        """Test plugin_names property."""
        registry = VirtualExpertRegistry()

        class Plugin1(VirtualExpertPlugin):
            name = "p1"
            description = "P1"

            def can_handle(self, prompt):
                return False

            def execute(self, prompt):
                return ""

            def get_calibration_prompts(self):
                return [], []

        class Plugin2(VirtualExpertPlugin):
            name = "p2"
            description = "P2"

            def can_handle(self, prompt):
                return False

            def execute(self, prompt):
                return ""

            def get_calibration_prompts(self):
                return [], []

        registry.register(Plugin1())
        registry.register(Plugin2())

        names = registry.plugin_names
        assert "p1" in names
        assert "p2" in names

    def test_len(self):
        """Test __len__."""
        registry = VirtualExpertRegistry()
        assert len(registry) == 0

        registry.register(TestPluginForTests())
        assert len(registry) == 1

    def test_contains(self):
        """Test __contains__."""
        registry = VirtualExpertRegistry()
        registry.register(TestPluginForTests())

        assert "test" in registry
        assert "nonexistent" not in registry

    def test_repr(self):
        """Test __repr__ (lines 126-127)."""
        registry = VirtualExpertRegistry()

        # Empty registry
        repr_str = repr(registry)
        assert "VirtualExpertRegistry" in repr_str
        assert "[]" in repr_str

        # With plugins
        registry.register(TestPluginForTests())
        repr_str = repr(registry)
        assert "test" in repr_str


class TestGetDefaultRegistry:
    """Tests for get_default_registry function."""

    def setup_method(self):
        reset_default_registry()

    def test_returns_registry(self):
        """Test returns registry with math plugin."""
        registry = get_default_registry()

        assert isinstance(registry, VirtualExpertRegistry)
        assert "math" in registry

    def test_singleton(self):
        """Test returns same instance."""
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2


class TestResetDefaultRegistry:
    """Tests for reset_default_registry function."""

    def test_reset(self):
        """Test resetting registry."""
        r1 = get_default_registry()
        reset_default_registry()
        r2 = get_default_registry()

        # Should be different instances
        assert r1 is not r2
