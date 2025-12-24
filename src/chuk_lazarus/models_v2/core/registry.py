"""
Model registry for architecture discovery and instantiation.

Provides:
- Decorator-based registration
- Lookup by model_type or architecture name
- Factory functions for creating models from config
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .config import ModelConfig
    from .protocols import Model

logger = logging.getLogger(__name__)

# Type for model classes
M = TypeVar("M", bound="Model")
ModelClass = type[M]
ModelFactory = Callable[["ModelConfig"], M]


class ModelRegistry:
    """
    Registry for model architectures.

    Supports lookup by:
    - model_type (e.g., "llama", "qwen2", "mamba")
    - architecture class name (e.g., "LlamaForCausalLM")
    """

    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._by_type: dict[str, ModelFactory] = {}
        self._by_arch: dict[str, ModelFactory] = {}
        self._aliases: dict[str, str] = {}

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
    ) -> Callable[[ModelFactory], ModelFactory]:
        """
        Decorator to register a model factory.

        Args:
            model_type: Primary model type identifier (e.g., "llama")
            architectures: HuggingFace architecture class names
            aliases: Alternative names for lookup

        Returns:
            Decorator function

        Example:
            @registry.register(
                model_type="llama",
                architectures=["LlamaForCausalLM", "LlamaModel"],
                aliases=["llama2", "llama3"],
            )
            def create_llama(config: ModelConfig) -> LlamaModel:
                return LlamaModel(config)
        """

        def decorator(factory: ModelFactory) -> ModelFactory:
            # Register by model_type
            if model_type:
                self._by_type[model_type.lower()] = factory
                logger.debug(f"Registered model type: {model_type}")

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

    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._by_type.clear()
        self._by_arch.clear()
        self._aliases.clear()


# Global registry instance
_registry = ModelRegistry.get_instance()


def register_model(
    model_type: str | None = None,
    architectures: list[str] | None = None,
    aliases: list[str] | None = None,
) -> Callable[[ModelFactory], ModelFactory]:
    """
    Decorator to register a model factory with the global registry.

    Example:
        @register_model(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
        )
        def create_llama(config: ModelConfig) -> LlamaModel:
            return LlamaModel(config)
    """
    return _registry.register(
        model_type=model_type,
        architectures=architectures,
        aliases=aliases,
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
