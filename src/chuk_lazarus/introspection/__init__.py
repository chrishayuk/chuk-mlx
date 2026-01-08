"""
Introspection tools for model analysis.

This module provides tools for understanding model behavior:
- Async-native analyzer with pydantic models
- Hooks for capturing intermediate activations
- Attention visualization
- Logit lens for layer-by-layer prediction analysis
- Ablation studies for causal circuit discovery
- Activation steering for behavior modification
- MoE introspection for router analysis and expert utilization

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

Example - MoE Analysis (GPT-OSS, Mixtral, Llama4):
    >>> from chuk_lazarus.introspection import MoEHooks, MoECaptureConfig
    >>>
    >>> hooks = MoEHooks(model)
    >>> hooks.configure(MoECaptureConfig(
    ...     capture_router_logits=True,
    ...     capture_selected_experts=True,
    ... ))
    >>> logits = hooks.forward(input_ids)
    >>>
    >>> # Analyze routing decisions
    >>> utilization = hooks.get_expert_utilization(layer_idx=4)
    >>> print(f"Load balance: {utilization.load_balance_score:.2%}")
    >>>
    >>> # Check router entropy (confidence)
    >>> entropy = hooks.get_router_entropy(layer_idx=4)
    >>> print(f"Routing confidence: {1 - entropy.normalized_entropy:.2%}")

Example - Logit Lens:
    >>> from chuk_lazarus.introspection import LogitLens
    >>>
    >>> lens = LogitLens(hooks, tokenizer)
    >>> lens.print_evolution()  # See predictions at each layer
"""

# Async analyzer (recommended API) - from subpackage
# Ablation study - from subpackage
from .ablation import (
    AblationConfig,
    AblationResult,
    AblationStudy,
    AblationType,
    ComponentType,
    LayerSweepResult,
    ModelAdapter,
)

# Model accessor for unified model component access
from .accessor import AsyncModelAccessor, ModelAccessor
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

# Enums for type-safe values
from .enums import (
    ArithmeticOperator,
    CommutativityLevel,
    ComputeStrategy,
    ConfidenceLevel,
    CriterionType,
    Difficulty,
    DirectionMethod,
    FactType,
    FormatDiagnosis,
    InvocationMethod,
    MemorizationLevel,
    NeuronRole,
    OverrideMode,
    PatchEffect,
    Region,
    TestStatus,
)

# Low-level hooks with enums
from .hooks import (
    CaptureConfig,
    CapturedState,
    LayerSelection,
    ModelHooks,
    PositionSelection,
)

# Counterfactual interventions for causal analysis
from .interventions import (
    CausalTraceResult,
    ComponentTarget,
    CounterfactualIntervention,
    FullCausalTrace,
    InterventionConfig,
    InterventionResult,
    InterventionType,
    patch_activations,
    trace_causal_path,
)
from .interventions import (
    PatchingResult as CounterfactualPatchingResult,
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

# Logit lens
from .logit_lens import (
    LayerPrediction,
    LogitLens,
    TokenEvolution,
    run_logit_lens,
)

# Pydantic models for structured results
from .models import (
    # Arithmetic
    ArithmeticStats,
    ArithmeticTestCase,
    ArithmeticTestResult,
    ArithmeticTestSuite,
    # Memory
    AttractorNode,
    # Uncertainty
    CalibrationResult,
    # Facts
    CapitalFact,
    # Circuit
    CapturedCircuit,
    CircuitComparisonResult,
    CircuitDirection,
    CircuitEntry,
    CircuitInvocationResult,
    CircuitTestResult,
    # Patching
    CommutativityPair,
    CommutativityResult,
    ElementFact,
    Fact,
    FactNeighborhood,
    FactSet,
    MathFact,
    MemoryAnalysisResult,
    MemoryStats,
    MetacognitiveResult,
    ParsedArithmeticPrompt,
    PatchingLayerResult,
    PatchingResult,
    # Probing
    ProbeLayerResult,
    ProbeResult,
    ProbeTopNeuron,
    RetrievalResult,
    UncertaintyResult,
)

# MoE introspection - from modular subpackage
from .moe import (
    # Identification
    CategoryActivation,
    # Datasets
    CategoryPrompts,
    # Models
    CoactivationAnalysis,
    # Compression
    CompressionAnalysis,
    CompressionPlan,
    ExpertAblationResult,
    # Enums
    ExpertCategory,
    ExpertIdentity,
    # Logit Lens
    ExpertLogitContribution,
    ExpertPair,
    ExpertProfile,
    ExpertRole,
    ExpertSimilarity,
    ExpertUtilization,
    LayerRoutingSnapshot,
    # Config
    MoEAblationConfig,
    MoEArchitecture,
    MoECaptureConfig,
    # Hooks
    MoECapturedState,
    MoEHooks,
    MoELayerInfo,
    MoELogitLens,
    PromptCategory,
    PromptCategoryGroup,
    RouterEntropy,
    # Ablation
    ablate_expert,
    ablate_expert_batch,
    # Router analysis
    analyze_coactivation,
    analyze_compression_opportunities,
    analyze_expert_vocabulary,
    cluster_experts_by_specialization,
    compare_routing,
    compute_expert_similarity,
    compute_routing_diversity,
    create_compression_plan,
    # Detection
    detect_moe_architecture,
    find_causal_experts,
    find_generalists,
    find_merge_candidates,
    find_prune_candidates,
    find_specialists,
    get_all_prompts,
    get_category_prompts,
    get_dominant_experts,
    get_grouped_prompts,
    get_moe_layer_info,
    get_moe_layers,
    get_prompts_by_group,
    get_prompts_flat,
    get_rare_experts,
    identify_all_experts,
    identify_expert,
    is_moe_model,
    print_compression_summary,
    print_expert_summary,
    sweep_layer_experts,
)
from .moe import (
    compute_similarity_matrix as moe_compute_similarity_matrix,
)

# Activation patching for causal interventions
from .patcher import ActivationPatcher, CommutativityAnalyzer

# Activation steering - from subpackage
from .steering import (
    ActivationSteering,
    LegacySteeringConfig,
    SteeredGemmaMLP,
    SteeringConfig,
    SteeringHook,
    SteeringMode,
    ToolCallingSteering,
    compare_steering_effects,
    format_functiongemma_prompt,
    steer_model,
)

# Utilities for CLI and programmatic use
from .utils import (
    analyze_orthogonality,
    apply_chat_template,
    compute_similarity_matrix,
    cosine_similarity,
    extract_expected_answer,
    find_answer_onset,
    find_discriminative_neurons,
    generate_arithmetic_prompts,
    load_external_chat_template,
    normalize_number_string,
    parse_layers_arg,
    parse_prompts_from_arg,
)

# Virtual expert system (re-exported from inference, with demo functions)
from .virtual_expert import (
    ExpertHijacker,
    HybridEmbeddingInjector,
    MathExpertPlugin,
    SafeMathEvaluator,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertConfig,
    VirtualExpertPlugin,
    VirtualExpertRegistry,
    VirtualExpertResult,
    VirtualExpertService,
    VirtualExpertServiceResult,
    VirtualExpertSlot,
    VirtualMoEWrapper,
    VirtualRouter,
    create_virtual_expert,
    create_virtual_expert_wrapper,
    demo_all_approaches,
    demo_virtual_expert,
    get_default_registry,
)

# Service layer for CLI commands
from .analyzer import AnalyzerService, AnalyzerServiceConfig, ComparisonResult
from .memory import MemoryAnalysisConfig, MemoryAnalysisResult, MemoryAnalysisService
from .probing import (
    MetacognitiveConfig,
    MetacognitiveService,
    ProbeConfig,
    ProbeService,
    UncertaintyConfig,
    UncertaintyService,
)
from .clustering import ClusteringConfig, ClusteringResult, ClusteringService
from .classifier import ClassifierConfig, ClassifierResult, ClassifierService
from .generation import (
    GenerationConfig,
    GenerationResult,
    GenerationService,
    LogitEvolutionConfig,
    LogitEvolutionResult,
    LogitEvolutionService,
)
from .circuit import (
    CircuitCaptureConfig,
    CircuitCaptureResult,
    CircuitCompareConfig,
    CircuitCompareResult,
    CircuitDecodeConfig,
    CircuitDecodeResult,
    CircuitExportConfig,
    CircuitExportResult,
    CircuitInvokeConfig,
    CircuitInvokeResult,
    CircuitService,
    CircuitTestConfig,
    CircuitTestResult,
    CircuitViewConfig,
    CircuitViewResult,
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
    "ModelAdapter",
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
    "SteeringHook",
    "SteeringMode",
    "LegacySteeringConfig",
    "SteeredGemmaMLP",
    "ToolCallingSteering",
    "steer_model",
    "compare_steering_effects",
    "format_functiongemma_prompt",
    # MoE introspection - Enums
    "MoEArchitecture",
    "ExpertCategory",
    "ExpertRole",
    # MoE introspection - Config
    "MoECaptureConfig",
    "MoEAblationConfig",
    # MoE introspection - Models
    "MoELayerInfo",
    "RouterEntropy",
    "ExpertUtilization",
    "ExpertIdentity",
    "ExpertPair",
    "CoactivationAnalysis",
    "ExpertAblationResult",
    "CompressionPlan",
    # MoE introspection - Detection
    "detect_moe_architecture",
    "get_moe_layer_info",
    "get_moe_layers",
    "is_moe_model",
    # MoE introspection - Hooks
    "MoEHooks",
    "MoECapturedState",
    # MoE introspection - Router analysis
    "analyze_coactivation",
    "compute_routing_diversity",
    "get_dominant_experts",
    "get_rare_experts",
    "compare_routing",
    # MoE introspection - Datasets
    "PromptCategory",
    "PromptCategoryGroup",
    "CategoryPrompts",
    "get_category_prompts",
    "get_all_prompts",
    "get_grouped_prompts",
    "get_prompts_by_group",
    "get_prompts_flat",
    # MoE introspection - Ablation
    "ablate_expert",
    "ablate_expert_batch",
    "find_causal_experts",
    "sweep_layer_experts",
    # MoE introspection - Logit Lens
    "ExpertLogitContribution",
    "LayerRoutingSnapshot",
    "MoELogitLens",
    "analyze_expert_vocabulary",
    # MoE introspection - Identification
    "CategoryActivation",
    "ExpertProfile",
    "identify_expert",
    "identify_all_experts",
    "find_specialists",
    "find_generalists",
    "cluster_experts_by_specialization",
    "print_expert_summary",
    # MoE introspection - Compression
    "ExpertSimilarity",
    "CompressionAnalysis",
    "compute_expert_similarity",
    "moe_compute_similarity_matrix",
    "find_merge_candidates",
    "find_prune_candidates",
    "create_compression_plan",
    "analyze_compression_opportunities",
    "print_compression_summary",
    # Utilities
    "generate_arithmetic_prompts",
    "cosine_similarity",
    "compute_similarity_matrix",
    "analyze_orthogonality",
    "find_discriminative_neurons",
    "normalize_number_string",
    "parse_prompts_from_arg",
    "parse_layers_arg",
    "apply_chat_template",
    "load_external_chat_template",
    "extract_expected_answer",
    "find_answer_onset",
    # Model accessor
    "ModelAccessor",
    "AsyncModelAccessor",
    # Activation patching
    "ActivationPatcher",
    "CommutativityAnalyzer",
    # Enums
    "ArithmeticOperator",
    "CommutativityLevel",
    "ComputeStrategy",
    "ConfidenceLevel",
    "CriterionType",
    "Difficulty",
    "DirectionMethod",
    "FactType",
    "FormatDiagnosis",
    "InvocationMethod",
    "MemorizationLevel",
    "NeuronRole",
    "OverrideMode",
    "PatchEffect",
    "Region",
    "TestStatus",
    # Pydantic models - Arithmetic
    "ParsedArithmeticPrompt",
    "ArithmeticTestCase",
    "ArithmeticTestResult",
    "ArithmeticStats",
    "ArithmeticTestSuite",
    # Pydantic models - Circuit
    "CircuitEntry",
    "CircuitDirection",
    "CapturedCircuit",
    "CircuitInvocationResult",
    "CircuitTestResult",
    "CircuitComparisonResult",
    # Pydantic models - Facts
    "Fact",
    "MathFact",
    "CapitalFact",
    "ElementFact",
    "FactSet",
    "FactNeighborhood",
    # Pydantic models - Memory
    "RetrievalResult",
    "AttractorNode",
    "MemoryStats",
    "MemoryAnalysisResult",
    # Pydantic models - Patching
    "CommutativityPair",
    "CommutativityResult",
    "PatchingLayerResult",
    "PatchingResult",
    # Pydantic models - Probing
    "ProbeLayerResult",
    "ProbeTopNeuron",
    "ProbeResult",
    # Pydantic models - Uncertainty
    "MetacognitiveResult",
    "UncertaintyResult",
    "CalibrationResult",
    # Virtual expert system
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "create_virtual_expert",
    "create_virtual_expert_wrapper",
    "get_default_registry",
    "demo_virtual_expert",
    "demo_all_approaches",
    # Legacy aliases
    "ExpertHijacker",
    "VirtualExpertSlot",
    "HybridEmbeddingInjector",
    # Counterfactual interventions
    "CounterfactualIntervention",
    "InterventionConfig",
    "InterventionResult",
    "InterventionType",
    "ComponentTarget",
    "CounterfactualPatchingResult",
    "CausalTraceResult",
    "FullCausalTrace",
    "patch_activations",
    "trace_causal_path",
    # Service layer for CLI
    "AnalyzerService",
    "AnalyzerServiceConfig",
    "ComparisonResult",
    "MemoryAnalysisConfig",
    "MemoryAnalysisResult",
    "MemoryAnalysisService",
    "MetacognitiveConfig",
    "MetacognitiveService",
    "ProbeConfig",
    "ProbeService",
    "UncertaintyConfig",
    "UncertaintyService",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusteringService",
    "ClassifierConfig",
    "ClassifierResult",
    "ClassifierService",
    "GenerationConfig",
    "GenerationResult",
    "GenerationService",
    "LogitEvolutionConfig",
    "LogitEvolutionResult",
    "LogitEvolutionService",
    "CircuitCaptureConfig",
    "CircuitCaptureResult",
    "CircuitCompareConfig",
    "CircuitCompareResult",
    "CircuitDecodeConfig",
    "CircuitDecodeResult",
    "CircuitExportConfig",
    "CircuitExportResult",
    "CircuitInvokeConfig",
    "CircuitInvokeResult",
    "CircuitService",
    "CircuitTestConfig",
    "CircuitTestResult",
    "CircuitViewConfig",
    "CircuitViewResult",
    "VirtualExpertConfig",
    "VirtualExpertService",
    "VirtualExpertServiceResult",
]
