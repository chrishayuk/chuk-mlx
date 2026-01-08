"""
Ablation subpackage for causal circuit discovery.

This package provides the AblationStudy class and related utilities
for running ablation experiments to identify causal circuits.
"""

from .adapter import ModelAdapter
from .config import AblationConfig, AblationType, ComponentType
from .loader import load_model_for_ablation
from .models import AblationResult, LayerSweepResult
from .service import (
    AblationCriterionFunctions,
    AblationService,
    AblationServiceConfig,
    AblationSweepResult,
    MultiPromptAblationResult,
    SingleAblationResult,
)
from .study import AblationStudy

__all__ = [
    # Config
    "AblationConfig",
    "AblationType",
    "ComponentType",
    # Models
    "AblationResult",
    "LayerSweepResult",
    # Adapter
    "ModelAdapter",
    # Study
    "AblationStudy",
    # Service
    "AblationService",
    "AblationServiceConfig",
    "AblationCriterionFunctions",
    "SingleAblationResult",
    "MultiPromptAblationResult",
    "AblationSweepResult",
    # Loader
    "load_model_for_ablation",
]
