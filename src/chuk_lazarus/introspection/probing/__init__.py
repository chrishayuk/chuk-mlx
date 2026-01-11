"""Probing services for introspection.

This module provides services for probing model activations:
- MetacognitiveService: Detect strategy switches
- UncertaintyService: Analyze model uncertainty
- ProbeService: Train linear probes
"""

from __future__ import annotations

from .service import (
    MetacognitiveConfig,
    MetacognitiveService,
    ProbeConfig,
    ProbeService,
    UncertaintyConfig,
    UncertaintyService,
)

__all__ = [
    "MetacognitiveConfig",
    "MetacognitiveService",
    "ProbeConfig",
    "ProbeService",
    "UncertaintyConfig",
    "UncertaintyService",
]
