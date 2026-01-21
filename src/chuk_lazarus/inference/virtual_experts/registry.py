"""
Plugin registry for virtual experts.

The registry manages plugin registration, lookup, and calibration data.
A global default registry is provided with the math plugin pre-registered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

# Import VirtualExpert (aliased as VirtualExpertPlugin for backwards compat)
from chuk_virtual_expert import VirtualExpert

if TYPE_CHECKING:
    pass


class VirtualExpertRegistry:
    """
    Registry for virtual expert plugins.

    Manages plugin registration, lookup, and calibration data storage.
    Now works with VirtualExpert from chuk-virtual-expert.

    Example:
        >>> from chuk_virtual_expert_time import TimeExpert
        >>> registry = VirtualExpertRegistry()
        >>> registry.register(TimeExpert())
        >>>
        >>> # Find handler for a prompt
        >>> expert = registry.find_handler("What time is it?")
        >>> if expert:
        ...     action = VirtualExpertAction(expert="time", operation="get_time", parameters={})
        ...     result = expert.execute(action)
    """

    def __init__(self):
        self._plugins: dict[str, VirtualExpert] = {}
        self._calibration_data: dict[str, tuple[list[mx.array], list[mx.array]]] = {}

    def register(self, plugin: VirtualExpert) -> None:
        """
        Register a virtual expert.

        Args:
            plugin: The expert to register (VirtualExpert instance or adapter)

        Raises:
            ValueError: If an expert with the same name is already registered
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Expert '{plugin.name}' already registered")
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        """
        Unregister an expert by name.

        Args:
            name: Name of the expert to remove
        """
        if name in self._plugins:
            del self._plugins[name]
        if name in self._calibration_data:
            del self._calibration_data[name]

    def get(self, name: str) -> VirtualExpert | None:
        """Get an expert by name."""
        return self._plugins.get(name)

    def get_all(self) -> list[VirtualExpert]:
        """Get all registered experts, sorted by priority (highest first)."""
        return sorted(self._plugins.values(), key=lambda p: -p.priority)

    def find_handler(self, prompt: str) -> VirtualExpert | None:
        """
        Find the first expert that can handle a prompt.

        Checks experts in priority order using can_handle() method
        (provided by LazarusAdapter for VirtualExpert instances).

        Args:
            prompt: The prompt to find a handler for

        Returns:
            The first matching expert, or None if no handler found
        """
        for plugin in self.get_all():
            # Check if plugin has can_handle method (adapters have it)
            if hasattr(plugin, 'can_handle') and plugin.can_handle(prompt):
                return plugin
        return None

    def set_calibration_data(
        self,
        name: str,
        positive: list[mx.array],
        negative: list[mx.array],
    ) -> None:
        """
        Store calibration data for an expert.

        Args:
            name: Expert name
            positive: Activations for positive examples
            negative: Activations for negative examples
        """
        self._calibration_data[name] = (positive, negative)

    def get_calibration_data(
        self,
        name: str,
    ) -> tuple[list[mx.array], list[mx.array]] | None:
        """Get stored calibration data for an expert."""
        return self._calibration_data.get(name)

    @property
    def plugin_names(self) -> list[str]:
        """Get names of all registered experts."""
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

    The default registry comes with MathExpert pre-registered.
    TimeExpert is also registered if chuk-virtual-expert-time is installed.
    """
    global _default_registry
    if _default_registry is None:
        from .plugins.math import MathExpert

        _default_registry = VirtualExpertRegistry()
        _default_registry.register(MathExpert())

        # Register TimeExpert if available
        try:
            from chuk_virtual_expert_time import TimeExpert
            _default_registry.register(TimeExpert())
        except ImportError:
            pass  # chuk-virtual-expert-time not installed

    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (mainly for testing)."""
    global _default_registry
    _default_registry = None
