"""
Introspection tools for model analysis.

This module provides tools for understanding model behavior:
- Async-native analyzer with pydantic models
- Hooks for capturing intermediate activations
- Attention visualization
- Logit lens for layer-by-layer prediction analysis

Example - Async Analyzer (Recommended):
    >>> from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig, LayerStrategy
    >>>
    >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    ...     result = await analyzer.analyze("The capital of France is")
    ...     print(result.predicted_token)  # " Paris"
    ...     for layer in result.layer_predictions:
    ...         print(f"Layer {layer.layer_idx}: {layer.top_token}")

Example - Track Token Evolution:
    >>> from chuk_lazarus.introspection import AnalysisConfig, PositionSelection
    >>>
    >>> config = AnalysisConfig(track_tokens=["Paris", " Paris"])
    >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    ...     result = await analyzer.analyze("The capital of France is", config)
    ...     for evolution in result.token_evolutions:
    ...         print(f"{evolution.token} emerges at layer {evolution.emergence_layer}")

Example - Low-level Hook Usage:
    >>> from chuk_lazarus.introspection import ModelHooks, CaptureConfig, LayerSelection
    >>>
    >>> hooks = ModelHooks(model)
    >>> hooks.configure(CaptureConfig(
    ...     layers=[0, 4, 8, 12],
    ...     capture_attention_weights=True,
    ... ))
    >>> logits = hooks.forward(input_ids)
    >>> hidden = hooks.state.hidden_states[4]

Example - Logit Lens:
    >>> from chuk_lazarus.introspection import LogitLens
    >>>
    >>> lens = LogitLens(hooks, tokenizer)
    >>> lens.print_evolution()  # See predictions at each layer
"""

# Async analyzer (recommended API)
from .analyzer import (
    AnalysisConfig,
    AnalysisResult,
    LayerPredictionResult,
    LayerStrategy,
    LayerTransition,
    ModelAnalyzer,
    ModelInfo,
    ResidualContribution,
    TokenEvolutionResult,
    TokenPrediction,
    TrackStrategy,
    analyze_prompt,
)

# Attention analysis
from .attention import (
    AggregationStrategy,
    AttentionAnalyzer,
    AttentionFocus,
    AttentionPattern,
    extract_attention_weights,
)

# Low-level hooks with enums
from .hooks import (
    CaptureConfig,
    CapturedState,
    LayerSelection,
    ModelHooks,
    PositionSelection,
)

# Logit lens
from .logit_lens import (
    LayerPrediction,
    LogitLens,
    TokenEvolution,
    run_logit_lens,
)

__all__ = [
    # Async analyzer (recommended)
    "ModelAnalyzer",
    "AnalysisConfig",
    "AnalysisResult",
    "LayerPredictionResult",
    "LayerTransition",
    "ResidualContribution",
    "TokenPrediction",
    "TokenEvolutionResult",
    "ModelInfo",
    "LayerStrategy",
    "TrackStrategy",
    "analyze_prompt",
    # Core hooks and enums
    "ModelHooks",
    "CaptureConfig",
    "CapturedState",
    "LayerSelection",
    "PositionSelection",
    # Attention analysis
    "AttentionAnalyzer",
    "AttentionPattern",
    "AttentionFocus",
    "AggregationStrategy",
    "extract_attention_weights",
    # Logit lens
    "LogitLens",
    "LayerPrediction",
    "TokenEvolution",
    "run_logit_lens",
    # Ablation studies
    "AblationStudy",
    "AblationConfig",
    "AblationResult",
    "AblationType",
    "ComponentType",
    "LayerSweepResult",
    # Layer analysis
    "LayerAnalyzer",
    "LayerAnalysisResult",
    "RepresentationResult",
    "AttentionResult",
    "ClusterResult",
    "analyze_format_sensitivity",
    # Activation steering
    "ActivationSteering",
    "SteeringConfig",
]

# Ablation study (lazy import to avoid circular deps)
from .ablation import (
    AblationConfig,
    AblationResult,
    AblationStudy,
    AblationType,
    ComponentType,
    LayerSweepResult,
)

# Layer analysis
from .layer_analysis import (
    AttentionResult,
    ClusterResult,
    LayerAnalysisResult,
    LayerAnalyzer,
    RepresentationResult,
    analyze_format_sensitivity,
)

# Activation steering
from .steering import (
    ActivationSteering,
    SteeringConfig,
)
