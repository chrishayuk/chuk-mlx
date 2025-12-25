"""
Common types for protocols.

Abstract tensor types - actual implementation depends on the backend.
"""

from __future__ import annotations

from typing import Any, TypeVar

# Abstract tensor type - concrete type depends on backend
Tensor = TypeVar("Tensor")

# Type aliases for cache types (backend-agnostic)
AttentionCache = tuple[Any, Any]  # (key_cache, value_cache)
BlockCache = Any  # Block-specific cache type
LayerCache = list[Any]  # List of per-layer caches
