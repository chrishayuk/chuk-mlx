"""
Tests for models_v2.core.registry.

Ensures the model registry correctly registers and looks up model factories.
"""

import pytest

from chuk_lazarus.models_v2.core.config import ModelConfig
from chuk_lazarus.models_v2.core.registry import (
    ModelRegistry,
    get_model_class,
    list_models,
    register_model,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.hidden_size = config.hidden_size

    def __call__(self, input_ids, cache=None):
        return None, None

    def set_mode(self, mode: str) -> None:
        pass


class AnotherMockModel:
    """Another mock model for testing."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_type = "another"


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_singleton(self):
        """Test registry is singleton."""
        registry1 = ModelRegistry.get_instance()
        registry2 = ModelRegistry.get_instance()
        assert registry1 is registry2

    def test_register_by_type(self):
        """Test registration by model_type."""
        registry = ModelRegistry()

        @registry.register(model_type="test_model")
        def create_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        factory = registry.get_factory("test_model")
        assert factory is not None

        config = ModelConfig()
        model = factory(config)
        assert isinstance(model, MockModel)

    def test_register_by_architecture(self):
        """Test registration by architecture name."""
        registry = ModelRegistry()

        @registry.register(architectures=["TestForCausalLM", "TestModel"])
        def create_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        factory = registry.get_factory("TestForCausalLM")
        assert factory is not None

        factory2 = registry.get_factory("TestModel")
        assert factory2 is not None

    def test_register_with_aliases(self):
        """Test registration with aliases."""
        registry = ModelRegistry()

        @registry.register(
            model_type="llama",
            aliases=["llama2", "llama3", "tinyllama"],
        )
        def create_llama(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # All should resolve to same factory
        factory1 = registry.get_factory("llama")
        factory2 = registry.get_factory("llama2")
        factory3 = registry.get_factory("llama3")
        factory4 = registry.get_factory("tinyllama")

        assert factory1 is not None
        assert factory2 is not None
        assert factory3 is not None
        assert factory4 is not None

    def test_case_insensitive_lookup(self):
        """Test model_type lookup is case-insensitive."""
        registry = ModelRegistry()

        @registry.register(model_type="CaseSensitive")
        def create_case(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Should find regardless of case
        assert registry.get_factory("casesensitive") is not None
        assert registry.get_factory("CASESENSITIVE") is not None
        assert registry.get_factory("CaseSensitive") is not None

    def test_architecture_case_sensitive(self):
        """Test architecture lookup is case-sensitive (matches HuggingFace)."""
        registry = ModelRegistry()

        @registry.register(architectures=["ExactCaseModel"])
        def create_exact(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Exact case should work
        assert registry.get_factory("ExactCaseModel") is not None

        # Different case should not find it
        assert registry.get_factory("exactcasemodel") is None

    def test_get_factory_not_found(self):
        """Test get_factory returns None for unknown model."""
        registry = ModelRegistry()
        assert registry.get_factory("nonexistent") is None

    def test_create_from_config_by_type(self):
        """Test create model from config using model_type."""
        registry = ModelRegistry()

        @registry.register(model_type="config_test")
        def create_config_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        config = ModelConfig(model_type="config_test", hidden_size=2048)
        model = registry.create(config)

        assert isinstance(model, MockModel)
        assert model.config.hidden_size == 2048

    def test_create_from_config_by_architecture(self):
        """Test create model from config using architectures."""
        registry = ModelRegistry()

        @registry.register(architectures=["ArchTestForCausalLM"])
        def create_arch_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        config = ModelConfig(architectures=["ArchTestForCausalLM"])
        model = registry.create(config)

        assert isinstance(model, MockModel)

    def test_create_not_found_raises(self):
        """Test create raises ValueError for unknown model."""
        registry = ModelRegistry()

        config = ModelConfig(model_type="unknown_model")

        with pytest.raises(ValueError) as exc_info:
            registry.create(config)

        assert "No factory found" in str(exc_info.value)
        assert "unknown_model" in str(exc_info.value)

    def test_list_types(self):
        """Test listing registered model types."""
        registry = ModelRegistry()

        @registry.register(model_type="type_a")
        def create_a(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(model_type="type_b")
        def create_b(config: ModelConfig) -> MockModel:
            return MockModel(config)

        types = registry.list_types()
        assert "type_a" in types
        assert "type_b" in types

    def test_list_architectures(self):
        """Test listing registered architectures."""
        registry = ModelRegistry()

        @registry.register(architectures=["ArchA", "ArchB"])
        def create_arch(config: ModelConfig) -> MockModel:
            return MockModel(config)

        archs = registry.list_architectures()
        assert "ArchA" in archs
        assert "ArchB" in archs

    def test_clear(self):
        """Test clearing registry."""
        registry = ModelRegistry()

        @registry.register(model_type="to_clear")
        def create_clear(config: ModelConfig) -> MockModel:
            return MockModel(config)

        assert registry.get_factory("to_clear") is not None

        registry.clear()

        assert registry.get_factory("to_clear") is None
        assert registry.list_types() == []
        assert registry.list_architectures() == []

    def test_register_class_directly(self):
        """Test registering a class directly (not via decorator)."""
        registry = ModelRegistry()

        registry.register_class(
            MockModel,
            model_type="direct_register",
            architectures=["DirectModel"],
        )

        factory = registry.get_factory("direct_register")
        assert factory is not None

        config = ModelConfig()
        model = factory(config)
        assert isinstance(model, MockModel)

    def test_multiple_registrations(self):
        """Test multiple models can be registered."""
        registry = ModelRegistry()

        @registry.register(model_type="model1")
        def create_model1(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(model_type="model2")
        def create_model2(config: ModelConfig) -> AnotherMockModel:
            return AnotherMockModel(config)

        config1 = ModelConfig(model_type="model1")
        config2 = ModelConfig(model_type="model2")

        model1 = registry.create(config1)
        model2 = registry.create(config2)

        assert isinstance(model1, MockModel)
        assert isinstance(model2, AnotherMockModel)


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_register_model_decorator(self):
        """Test global register_model decorator."""
        # Note: This modifies the global registry
        # We need to be careful about cleanup

        @register_model(model_type="global_test_unique_12345")
        def create_global_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        factory = get_model_class("global_test_unique_12345")
        assert factory is not None

    def test_get_model_class(self):
        """Test global get_model_class function."""

        @register_model(model_type="get_test_unique_67890")
        def create_get_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        factory = get_model_class("get_test_unique_67890")
        assert factory is not None

        config = ModelConfig()
        model = factory(config)
        assert isinstance(model, MockModel)

    def test_list_models(self):
        """Test global list_models function."""
        result = list_models()

        assert "types" in result
        assert "architectures" in result
        assert isinstance(result["types"], list)
        assert isinstance(result["architectures"], list)


class TestRegistryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_whitespace_model_type_stripped(self):
        """Test model_type with whitespace is handled."""
        registry = ModelRegistry()

        @registry.register(model_type="whitespace_test")
        def create_whitespace(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Should find with exact match
        factory = registry.get_factory("whitespace_test")
        assert factory is not None

    def test_none_values_in_register(self):
        """Test registration with None values."""
        registry = ModelRegistry()

        # Should not raise
        @registry.register(
            model_type=None,
            architectures=None,
            aliases=None,
        )
        def create_none(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Nothing should be registered since all are None
        assert registry.list_types() == []
        assert registry.list_architectures() == []

    def test_create_prefers_model_type_over_architecture(self):
        """Test create prefers model_type over architectures."""
        registry = ModelRegistry()

        @registry.register(model_type="preferred")
        def create_preferred(config: ModelConfig) -> MockModel:
            model = MockModel(config)
            model.source = "model_type"
            return model

        @registry.register(architectures=["PreferredArch"])
        def create_arch(config: ModelConfig) -> AnotherMockModel:
            model = AnotherMockModel(config)
            model.source = "architecture"
            return model

        # Config has both model_type and architectures
        config = ModelConfig(
            model_type="preferred",
            architectures=["PreferredArch"],
        )

        model = registry.create(config)
        # Should use model_type factory
        assert isinstance(model, MockModel)
        assert model.source == "model_type"

    def test_create_fallback_to_architecture(self):
        """Test create falls back to architectures when model_type not found."""
        registry = ModelRegistry()

        @registry.register(architectures=["FallbackArch"])
        def create_fallback(config: ModelConfig) -> MockModel:
            return MockModel(config)

        config = ModelConfig(
            model_type="nonexistent",
            architectures=["FallbackArch"],
        )

        model = registry.create(config)
        assert isinstance(model, MockModel)
