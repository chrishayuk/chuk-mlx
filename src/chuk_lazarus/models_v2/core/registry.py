"""
Model registry for architecture discovery and instantiation.

Provides:
- Decorator-based registration
- Lookup by model_type or architecture name
- Factory functions for creating models from config
- Capability tracking for MoE routing and gym-driven selection

This is Phase 1 of the registry roadmap:
- Phase 1: Model registry + capabilities (current)
- Phase 2: Expert-ready model boundaries
- Phase 3: Deterministic routing
- Phase 4: Gym-driven architecture pressure
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .config import ModelConfig
    from .protocols import Model

logger = logging.getLogger(__name__)

# Type for model classes
M = TypeVar("M", bound="Model")
ModelClass = type[M]
ModelFactory = Callable[["ModelConfig"], M]


@dataclass
class ModelCapability:
    """
    Capability descriptor for model registry.

    Capabilities answer:
    - "What models exist?" → list_models()
    - "What capabilities do they have?" → get_capabilities()
    - "What inputs do they accept?" → input_format

    Future: Used for MoE routing, gym-driven selection, and expert specialization.
    """

    # Core capability flags
    is_sequence_model: bool = True
    is_policy_model: bool = False
    is_memory_reader: bool = False
    is_memory_writer: bool = False
    is_router_candidate: bool = False

    # Input/output format
    input_format: str = "tokens"  # tokens, embeddings, images, etc.
    output_format: str = "logits"  # logits, embeddings, classifications, etc.

    # Model characteristics
    supports_kv_cache: bool = False
    supports_lora: bool = True
    max_context_length: int | None = None

    # Domain specializations (from fine-tuning)
    domains: list[str] = field(default_factory=list)

    # Tags for filtering
    tags: list[str] = field(default_factory=list)


class ModelRegistry:
    """
    Registry for model architectures.

    Supports lookup by:
    - model_type (e.g., "llama", "qwen2", "mamba")
    - architecture class name (e.g., "LlamaForCausalLM")
    - capabilities (for MoE routing)

    Phase 1 of the registry roadmap - provides:
    - "What models exist?" → list_models()
    - "What capabilities do they have?" → get_capabilities()
    - "What inputs do they accept?" → get_capabilities().input_format
    """

    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._by_type: dict[str, ModelFactory] = {}
        self._by_arch: dict[str, ModelFactory] = {}
        self._aliases: dict[str, str] = {}
        self._capabilities: dict[str, ModelCapability] = {}

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        model_type: str | None = None,
        architectures: list[str] | None = None,
        aliases: list[str] | None = None,
        capabilities: ModelCapability | None = None,
    ) -> Callable[[ModelFactory], ModelFactory]:
        """
        Decorator to register a model factory.

        Args:
            model_type: Primary model type identifier (e.g., "llama")
            architectures: HuggingFace architecture class names
            aliases: Alternative names for lookup
            capabilities: Model capabilities for routing and selection

        Returns:
            Decorator function

        Example:
            @registry.register(
                model_type="llama",
                architectures=["LlamaForCausalLM", "LlamaModel"],
                aliases=["llama2", "llama3"],
                capabilities=ModelCapability(
                    is_sequence_model=True,
                    supports_kv_cache=True,
                    tags=["causal_lm", "instruct"],
                ),
            )
            def create_llama(config: ModelConfig) -> LlamaModel:
                return LlamaModel(config)
        """

        def decorator(factory: ModelFactory) -> ModelFactory:
            # Register by model_type
            if model_type:
                self._by_type[model_type.lower()] = factory
                logger.debug(f"Registered model type: {model_type}")

                # Store capabilities if provided
                if capabilities:
                    self._capabilities[model_type.lower()] = capabilities

            # Register by architecture names
            if architectures:
                for arch in architectures:
                    self._by_arch[arch] = factory
                    logger.debug(f"Registered architecture: {arch}")

            # Register aliases
            if aliases:
                for alias in aliases:
                    if model_type:
                        self._aliases[alias.lower()] = model_type.lower()
                    logger.debug(f"Registered alias: {alias} -> {model_type}")

            return factory

        return decorator

    def register_class(
        self,
        model_class: ModelClass,
        model_type: str | None = None,
        architectures: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """
        Register a model class directly.

        Args:
            model_class: The model class to register
            model_type: Primary model type identifier
            architectures: HuggingFace architecture class names
            aliases: Alternative names for lookup
        """

        def factory(config: ModelConfig) -> Any:
            return model_class(config)

        if model_type:
            self._by_type[model_type.lower()] = factory

        if architectures:
            for arch in architectures:
                self._by_arch[arch] = factory

        if aliases:
            for alias in aliases:
                if model_type:
                    self._aliases[alias.lower()] = model_type.lower()

    def get_factory(self, identifier: str) -> ModelFactory | None:
        """
        Get model factory by identifier.

        Looks up in order:
        1. Exact model_type match
        2. Alias match
        3. Architecture class name match

        Args:
            identifier: Model type, alias, or architecture name

        Returns:
            Model factory function or None
        """
        # Try model_type
        factory = self._by_type.get(identifier.lower())
        if factory:
            return factory

        # Try alias
        canonical = self._aliases.get(identifier.lower())
        if canonical:
            factory = self._by_type.get(canonical)
            if factory:
                return factory

        # Try architecture name
        factory = self._by_arch.get(identifier)
        if factory:
            return factory

        return None

    def create(self, config: ModelConfig) -> Model:
        """
        Create model from config.

        Uses config.model_type or config.architectures to find factory.

        Args:
            config: Model configuration

        Returns:
            Instantiated model

        Raises:
            ValueError: If no matching factory found
        """
        factory = None

        # Try model_type first
        if config.model_type:
            factory = self.get_factory(config.model_type)

        # Fall back to architectures
        if factory is None and config.architectures:
            for arch in config.architectures:
                factory = self.get_factory(arch)
                if factory:
                    break

        if factory is None:
            available = sorted(set(self._by_type.keys()) | set(self._by_arch.keys()))
            raise ValueError(
                f"No factory found for model_type={config.model_type}, "
                f"architectures={config.architectures}. "
                f"Available: {available}"
            )

        return factory(config)

    def list_types(self) -> list[str]:
        """List all registered model types."""
        return sorted(self._by_type.keys())

    def list_architectures(self) -> list[str]:
        """List all registered architecture names."""
        return sorted(self._by_arch.keys())

    def get_capabilities(self, model_type: str) -> ModelCapability | None:
        """
        Get capabilities for a model type.

        Args:
            model_type: Model type identifier

        Returns:
            ModelCapability or None if not registered
        """
        # Resolve alias if needed
        key = model_type.lower()
        if key in self._aliases:
            key = self._aliases[key]
        return self._capabilities.get(key)

    def find_by_capability(
        self,
        is_sequence_model: bool | None = None,
        is_policy_model: bool | None = None,
        is_memory_reader: bool | None = None,
        is_memory_writer: bool | None = None,
        supports_kv_cache: bool | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        Find models matching capability criteria.

        Used for MoE routing and expert selection.

        Args:
            is_sequence_model: Filter by sequence model capability
            is_policy_model: Filter by policy model capability
            is_memory_reader: Filter by memory reader capability
            is_memory_writer: Filter by memory writer capability
            supports_kv_cache: Filter by KV cache support
            tags: Filter by tags (any match)

        Returns:
            List of model type identifiers matching the criteria
        """
        matches = []

        for model_type, caps in self._capabilities.items():
            # Check each criterion if specified
            if is_sequence_model is not None and caps.is_sequence_model != is_sequence_model:
                continue
            if is_policy_model is not None and caps.is_policy_model != is_policy_model:
                continue
            if is_memory_reader is not None and caps.is_memory_reader != is_memory_reader:
                continue
            if is_memory_writer is not None and caps.is_memory_writer != is_memory_writer:
                continue
            if supports_kv_cache is not None and caps.supports_kv_cache != supports_kv_cache:
                continue
            if tags is not None:
                # Check if any tag matches
                if not any(tag in caps.tags for tag in tags):
                    continue

            matches.append(model_type)

        return sorted(matches)

    def list_capabilities(self) -> dict[str, ModelCapability]:
        """
        List all registered capabilities.

        Returns:
            Dict mapping model_type to ModelCapability
        """
        return dict(self._capabilities)

    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._by_type.clear()
        self._by_arch.clear()
        self._aliases.clear()
        self._capabilities.clear()


# Global registry instance
_registry = ModelRegistry.get_instance()


def register_model(
    model_type: str | None = None,
    architectures: list[str] | None = None,
    aliases: list[str] | None = None,
    capabilities: ModelCapability | None = None,
) -> Callable[[ModelFactory], ModelFactory]:
    """
    Decorator to register a model factory with the global registry.

    Example:
        @register_model(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            capabilities=ModelCapability(
                supports_kv_cache=True,
                tags=["causal_lm"],
            ),
        )
        def create_llama(config: ModelConfig) -> LlamaModel:
            return LlamaModel(config)
    """
    return _registry.register(
        model_type=model_type,
        architectures=architectures,
        aliases=aliases,
        capabilities=capabilities,
    )


def get_model_class(identifier: str) -> ModelFactory | None:
    """
    Get model factory by identifier from global registry.

    Args:
        identifier: Model type, alias, or architecture name

    Returns:
        Model factory function or None
    """
    return _registry.get_factory(identifier)


# Alias for convenience
get_factory = get_model_class


def create_model(config: ModelConfig) -> Model:
    """
    Create model from config using global registry.

    Args:
        config: Model configuration

    Returns:
        Instantiated model
    """
    return _registry.create(config)


def list_models() -> dict[str, list[str]]:
    """
    List all registered models.

    Returns:
        Dict with 'types' and 'architectures' keys
    """
    return {
        "types": _registry.list_types(),
        "architectures": _registry.list_architectures(),
    }


def get_model_capabilities(model_type: str) -> ModelCapability | None:
    """
    Get capabilities for a model type.

    Args:
        model_type: Model type identifier

    Returns:
        ModelCapability or None if not registered
    """
    return _registry.get_capabilities(model_type)


def find_models_by_capability(
    is_sequence_model: bool | None = None,
    is_policy_model: bool | None = None,
    is_memory_reader: bool | None = None,
    is_memory_writer: bool | None = None,
    supports_kv_cache: bool | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    """
    Find models matching capability criteria.

    Used for MoE routing and expert selection.

    Args:
        is_sequence_model: Filter by sequence model capability
        is_policy_model: Filter by policy model capability
        is_memory_reader: Filter by memory reader capability
        is_memory_writer: Filter by memory writer capability
        supports_kv_cache: Filter by KV cache support
        tags: Filter by tags (any match)

    Returns:
        List of model type identifiers matching the criteria
    """
    return _registry.find_by_capability(
        is_sequence_model=is_sequence_model,
        is_policy_model=is_policy_model,
        is_memory_reader=is_memory_reader,
        is_memory_writer=is_memory_writer,
        supports_kv_cache=supports_kv_cache,
        tags=tags,
    )
