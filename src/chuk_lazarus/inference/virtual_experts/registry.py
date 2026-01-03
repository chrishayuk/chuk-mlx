"""
Plugin registry for virtual experts.

The registry manages plugin registration, lookup, and calibration data.
A global default registry is provided with the math plugin pre-registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from .base import VirtualExpertPlugin


class VirtualExpertRegistry:
    """
    Registry for virtual expert plugins.

    Manages plugin registration, lookup, and calibration data storage.

    Example:
        >>> registry = VirtualExpertRegistry()
        >>> registry.register(MathExpertPlugin())
        >>> registry.register(TranslationExpert())
        >>>
        >>> # Find handler for a prompt
        >>> plugin = registry.find_handler("127 * 89 = ")
        >>> if plugin:
        ...     result = plugin.execute("127 * 89 = ")
    """

    def __init__(self):
        self._plugins: dict[str, VirtualExpertPlugin] = {}
        self._calibration_data: dict[str, tuple[list[mx.array], list[mx.array]]] = {}

    def register(self, plugin: VirtualExpertPlugin) -> None:
        """
        Register a virtual expert plugin.

        Args:
            plugin: The plugin to register

        Raises:
            ValueError: If a plugin with the same name is already registered
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        """
        Unregister a plugin by name.

        Args:
            name: Name of the plugin to remove
        """
        if name in self._plugins:
            del self._plugins[name]
        if name in self._calibration_data:
            del self._calibration_data[name]

    def get(self, name: str) -> VirtualExpertPlugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_all(self) -> list[VirtualExpertPlugin]:
        """Get all registered plugins, sorted by priority (highest first)."""
        return sorted(self._plugins.values(), key=lambda p: -p.priority)

    def find_handler(self, prompt: str) -> VirtualExpertPlugin | None:
        """
        Find the first plugin that can handle a prompt.

        Checks plugins in priority order and returns the first one
        where can_handle() returns True.

        Args:
            prompt: The prompt to find a handler for

        Returns:
            The first matching plugin, or None if no handler found
        """
        for plugin in self.get_all():
            if plugin.can_handle(prompt):
                return plugin
        return None

    def set_calibration_data(
        self,
        name: str,
        positive: list[mx.array],
        negative: list[mx.array],
    ) -> None:
        """
        Store calibration data for a plugin.

        Args:
            name: Plugin name
            positive: Activations for positive examples
            negative: Activations for negative examples
        """
        self._calibration_data[name] = (positive, negative)

    def get_calibration_data(
        self,
        name: str,
    ) -> tuple[list[mx.array], list[mx.array]] | None:
        """Get stored calibration data for a plugin."""
        return self._calibration_data.get(name)

    @property
    def plugin_names(self) -> list[str]:
        """Get names of all registered plugins."""
        return list(self._plugins.keys())

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __repr__(self) -> str:
        plugins = ", ".join(self.plugin_names)
        return f"VirtualExpertRegistry([{plugins}])"


# Global default registry
_default_registry: VirtualExpertRegistry | None = None


def get_default_registry() -> VirtualExpertRegistry:
    """
    Get the default plugin registry.

    The default registry comes with the MathExpertPlugin pre-registered.
    """
    global _default_registry
    if _default_registry is None:
        from .plugins.math import MathExpertPlugin

        _default_registry = VirtualExpertRegistry()
        _default_registry.register(MathExpertPlugin())
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (mainly for testing)."""
    global _default_registry
    _default_registry = None
