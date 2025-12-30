"""
Analyzer subpackage for model introspection.

This package provides the ModelAnalyzer class for async-native
model introspection using logit lens and token tracking.
"""

from .config import AnalysisConfig, LayerStrategy, TrackStrategy
from .core import ModelAnalyzer, analyze_prompt
from .loader import (
    _is_quantized_model,
    _load_model_sync,
    get_model_hidden_size,
    get_model_num_layers,
    get_model_vocab_size,
)
from .models import (
    AnalysisResult,
    LayerPredictionResult,
    LayerTransition,
    ModelInfo,
    ResidualContribution,
    TokenEvolutionResult,
    TokenPrediction,
)
from .utils import (
    compute_entropy,
    compute_js_divergence,
    compute_kl_divergence,
    get_layers_to_capture,
)

__all__ = [
    # Config
    "AnalysisConfig",
    "LayerStrategy",
    "TrackStrategy",
    # Models
    "AnalysisResult",
    "LayerPredictionResult",
    "LayerTransition",
    "ModelInfo",
    "ResidualContribution",
    "TokenEvolutionResult",
    "TokenPrediction",
    # Core
    "ModelAnalyzer",
    "analyze_prompt",
    # Loader
    "_is_quantized_model",
    "_load_model_sync",
    "get_model_hidden_size",
    "get_model_num_layers",
    "get_model_vocab_size",
    # Utils
    "compute_entropy",
    "compute_js_divergence",
    "compute_kl_divergence",
    "get_layers_to_capture",
]
