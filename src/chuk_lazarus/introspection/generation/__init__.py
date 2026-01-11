"""Generation services for introspection.

This module provides generation services with logit lens analysis:
- GenerationService: Generate with layer analysis
- LogitEvolutionService: Analyze logit evolution across layers
"""

from __future__ import annotations

from .service import (
    GenerationConfig,
    GenerationResult,
    GenerationService,
    LogitEvolutionConfig,
    LogitEvolutionResult,
    LogitEvolutionService,
)

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "GenerationService",
    "LogitEvolutionConfig",
    "LogitEvolutionResult",
    "LogitEvolutionService",
]
