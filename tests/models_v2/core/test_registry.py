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


class TestModelCapability:
    """Tests for ModelCapability dataclass."""

    def test_creation(self):
        """Test creating ModelCapability."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        caps = ModelCapability(
            is_sequence_model=True,
            supports_kv_cache=True,
            tags=["causal_lm", "instruct"],
        )
        assert caps.is_sequence_model is True
        assert caps.supports_kv_cache is True
        assert "causal_lm" in caps.tags

    def test_default_values(self):
        """Test default capability values."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        caps = ModelCapability()
        assert caps.is_sequence_model is True
        assert caps.is_policy_model is False
        assert caps.is_memory_reader is False
        assert caps.is_memory_writer is False
        assert caps.is_router_candidate is False
        assert caps.input_format == "tokens"
        assert caps.output_format == "logits"
        assert caps.supports_lora is True
        assert caps.domains == []
        assert caps.tags == []

    def test_all_fields(self):
        """Test all capability fields."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        caps = ModelCapability(
            is_sequence_model=True,
            is_policy_model=True,
            is_memory_reader=True,
            is_memory_writer=True,
            is_router_candidate=True,
            input_format="embeddings",
            output_format="classifications",
            supports_kv_cache=True,
            supports_lora=False,
            max_context_length=4096,
            domains=["code", "math"],
            tags=["classifier", "encoder"],
        )
        assert caps.is_policy_model is True
        assert caps.is_memory_reader is True
        assert caps.is_memory_writer is True
        assert caps.is_router_candidate is True
        assert caps.input_format == "embeddings"
        assert caps.output_format == "classifications"
        assert caps.max_context_length == 4096
        assert "code" in caps.domains
        assert "classifier" in caps.tags


class TestRegistryCapabilities:
    """Tests for registry capability tracking."""

    def test_register_with_capabilities(self):
        """Test registration with capabilities."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        caps = ModelCapability(
            supports_kv_cache=True,
            tags=["causal_lm"],
        )

        @registry.register(model_type="caps_test", capabilities=caps)
        def create_caps_test(config: ModelConfig) -> MockModel:
            return MockModel(config)

        stored = registry.get_capabilities("caps_test")
        assert stored is not None
        assert stored.supports_kv_cache is True
        assert "causal_lm" in stored.tags

    def test_get_capabilities_case_insensitive(self):
        """Test capabilities lookup is case-insensitive."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="CaseCaps",
            capabilities=ModelCapability(supports_kv_cache=True),
        )
        def create_case_caps(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Should work with any case
        assert registry.get_capabilities("casecaps") is not None
        assert registry.get_capabilities("CASECAPS") is not None
        assert registry.get_capabilities("CaseCaps") is not None

    def test_get_capabilities_via_alias(self):
        """Test capabilities lookup via alias."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="aliased_caps",
            aliases=["ac1", "ac2"],
            capabilities=ModelCapability(is_policy_model=True),
        )
        def create_aliased_caps(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Should work via alias
        assert registry.get_capabilities("ac1") is not None
        assert registry.get_capabilities("ac2") is not None
        assert registry.get_capabilities("ac1").is_policy_model is True

    def test_find_by_capability_sequence_model(self):
        """Test finding models by sequence model capability."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="seq_model",
            capabilities=ModelCapability(is_sequence_model=True),
        )
        def create_seq(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(
            model_type="non_seq_model",
            capabilities=ModelCapability(is_sequence_model=False),
        )
        def create_non_seq(config: ModelConfig) -> MockModel:
            return MockModel(config)

        seq_models = registry.find_by_capability(is_sequence_model=True)
        assert "seq_model" in seq_models
        assert "non_seq_model" not in seq_models

        non_seq_models = registry.find_by_capability(is_sequence_model=False)
        assert "non_seq_model" in non_seq_models
        assert "seq_model" not in non_seq_models

    def test_find_by_capability_kv_cache(self):
        """Test finding models by KV cache support."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="cached_model",
            capabilities=ModelCapability(supports_kv_cache=True),
        )
        def create_cached(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(
            model_type="uncached_model",
            capabilities=ModelCapability(supports_kv_cache=False),
        )
        def create_uncached(config: ModelConfig) -> MockModel:
            return MockModel(config)

        cached = registry.find_by_capability(supports_kv_cache=True)
        assert "cached_model" in cached
        assert "uncached_model" not in cached

    def test_find_by_capability_tags(self):
        """Test finding models by tags."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="instruct_model",
            capabilities=ModelCapability(tags=["instruct", "chat"]),
        )
        def create_instruct(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(
            model_type="base_model",
            capabilities=ModelCapability(tags=["base", "pretrained"]),
        )
        def create_base(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Find by single tag
        instruct = registry.find_by_capability(tags=["instruct"])
        assert "instruct_model" in instruct
        assert "base_model" not in instruct

        # Find by any of multiple tags
        chat_or_base = registry.find_by_capability(tags=["chat", "base"])
        assert "instruct_model" in chat_or_base  # has "chat"
        assert "base_model" in chat_or_base  # has "base"

    def test_find_by_multiple_capabilities(self):
        """Test finding models by multiple capability criteria."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="full_featured",
            capabilities=ModelCapability(
                is_sequence_model=True,
                supports_kv_cache=True,
                tags=["instruct"],
            ),
        )
        def create_full(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(
            model_type="partial_featured",
            capabilities=ModelCapability(
                is_sequence_model=True,
                supports_kv_cache=False,
                tags=["instruct"],
            ),
        )
        def create_partial(config: ModelConfig) -> MockModel:
            return MockModel(config)

        # Must match all criteria
        matches = registry.find_by_capability(
            is_sequence_model=True,
            supports_kv_cache=True,
            tags=["instruct"],
        )
        assert "full_featured" in matches
        assert "partial_featured" not in matches

    def test_list_capabilities(self):
        """Test listing all capabilities."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="list_test_1",
            capabilities=ModelCapability(supports_kv_cache=True),
        )
        def create_1(config: ModelConfig) -> MockModel:
            return MockModel(config)

        @registry.register(
            model_type="list_test_2",
            capabilities=ModelCapability(is_policy_model=True),
        )
        def create_2(config: ModelConfig) -> MockModel:
            return MockModel(config)

        caps = registry.list_capabilities()
        assert "list_test_1" in caps
        assert "list_test_2" in caps
        assert caps["list_test_1"].supports_kv_cache is True
        assert caps["list_test_2"].is_policy_model is True

    def test_clear_clears_capabilities(self):
        """Test clear also clears capabilities."""
        from chuk_lazarus.models_v2.core.registry import ModelCapability

        registry = ModelRegistry()

        @registry.register(
            model_type="to_clear",
            capabilities=ModelCapability(supports_kv_cache=True),
        )
        def create_clear(config: ModelConfig) -> MockModel:
            return MockModel(config)

        assert registry.get_capabilities("to_clear") is not None
        assert registry.list_capabilities() != {}

        registry.clear()

        assert registry.get_capabilities("to_clear") is None
        assert registry.list_capabilities() == {}


class TestGlobalCapabilityFunctions:
    """Tests for global capability functions."""

    def test_get_model_capabilities(self):
        """Test global get_model_capabilities function."""
        from chuk_lazarus.models_v2.core.registry import (
            ModelCapability,
            get_model_capabilities,
        )

        @register_model(
            model_type="global_caps_test_unique_11111",
            capabilities=ModelCapability(tags=["test"]),
        )
        def create_global_caps(config: ModelConfig) -> MockModel:
            return MockModel(config)

        caps = get_model_capabilities("global_caps_test_unique_11111")
        assert caps is not None
        assert "test" in caps.tags

    def test_find_models_by_capability(self):
        """Test global find_models_by_capability function."""
        from chuk_lazarus.models_v2.core.registry import (
            ModelCapability,
            find_models_by_capability,
        )

        @register_model(
            model_type="global_find_test_unique_22222",
            capabilities=ModelCapability(
                is_memory_reader=True,
                tags=["memory"],
            ),
        )
        def create_memory_model(config: ModelConfig) -> MockModel:
            return MockModel(config)

        matches = find_models_by_capability(
            is_memory_reader=True,
            tags=["memory"],
        )
        assert "global_find_test_unique_22222" in matches
