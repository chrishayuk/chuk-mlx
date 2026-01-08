"""
Mixture of Experts (MoE) introspection subpackage.

Provides tools for understanding MoE routing decisions, expert specialization,
and per-expert contributions to model predictions.

Example:
    >>> from chuk_lazarus.introspection.moe import MoEHooks, MoECaptureConfig
    >>> from chuk_lazarus.introspection.moe.datasets import PromptCategory
    >>>
    >>> hooks = MoEHooks(model)
    >>> hooks.configure(MoECaptureConfig(capture_router_logits=True))
    >>> output = hooks.forward(input_ids)
    >>>
    >>> # Analyze routing
    >>> utilization = hooks.get_expert_utilization(layer_idx=4)
    >>> print(f"Load balance: {utilization.load_balance_score:.2%}")
"""

# Enums
# Ablation
from .ablation import (
    ablate_expert,
    ablate_expert_batch,
    find_causal_experts,
    sweep_layer_experts,
)

# Compression
from .compression import (
    ActivationOverlapResult,
    CompressionAnalysis,
    ExpertActivationStats,
    ExpertSimilarity,
    analyze_compression_opportunities,
    collect_expert_activations,
    compute_activation_overlap,
    compute_expert_similarity,
    compute_expert_similarity_with_activations,
    compute_similarity_matrix,
    compute_similarity_matrix_with_activations,
    create_compression_plan,
    find_merge_candidates,
    find_merge_candidates_with_activations,
    find_prune_candidates,
    print_activation_overlap_matrix,
    print_compression_summary,
)

# Config
from .config import MoEAblationConfig, MoECaptureConfig

# Datasets
from .datasets import (
    CATEGORY_GROUPS,
    CategoryPrompts,
    PromptCategory,
    PromptCategoryGroup,
    get_all_prompts,
    get_category_prompts,
    get_grouped_prompts,
    get_prompts_by_group,
    get_prompts_flat,
)

# Detection
from .detector import (
    detect_moe_architecture,
    get_moe_layer_info,
    get_moe_layers,
    is_moe_model,
)
from .enums import ExpertCategory, ExpertRole, MoEAction, MoEArchitecture

# Expert Router
from .expert_router import ExpertRouter

# Hooks
from .hooks import MoECapturedState, MoEHooks

# Identification
from .identification import (
    CategoryActivation,
    ExpertProfile,
    cluster_experts_by_specialization,
    find_generalists,
    find_specialists,
    identify_all_experts,
    identify_expert,
    print_expert_summary,
)

# Logit Lens
from .logit_lens import (
    ExpertLogitContribution,
    ExpertVocabContribution,
    LayerRoutingSnapshot,
    LayerVocabAnalysis,
    MoELogitLens,
    TokenExpertPreference,
    VocabExpertMapping,
    analyze_expert_vocabulary,
    compute_expert_vocab_contribution,
    compute_token_expert_mapping,
    find_expert_specialists,
    print_expert_vocab_summary,
    print_token_expert_preferences,
)

# Models
from .models import (
    CoactivationAnalysis,
    CompressionPlan,
    ExpertAblationResult,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertIdentity,
    ExpertPair,
    ExpertPattern,
    ExpertTaxonomy,
    ExpertUtilization,
    GenerationStats,
    LayerDivergenceResult,
    LayerRouterWeights,
    LayerRoutingAnalysis,
    MoELayerInfo,
    MoEModelInfo,
    RouterEntropy,
    RouterWeightCapture,
    TokenExpertMapping,
    TopKVariationResult,
    VocabExpertAnalysis,
)

# Router analysis
from .router import (
    analyze_coactivation,
    compare_routing,
    compute_routing_diversity,
    get_dominant_experts,
    get_rare_experts,
)

# Cross-layer tracking
from .tracking import (
    CrossLayerAnalysis,
    ExpertPipeline,
    ExpertPipelineNode,
    LayerAlignmentResult,
    analyze_cross_layer_routing,
    compute_expert_activation_profile,
    compute_layer_alignment,
    identify_functional_pipelines,
    print_alignment_matrix,
    print_pipeline_summary,
    track_expert_across_layers,
)

# Analysis Service
from .analysis_service import (
    AttentionCaptureResult,
    ExpertWeightInfo,
    LayerRoutingInfo,
    MoEAnalysisService,
    MoEAnalysisServiceConfig,
    PositionRoutingInfo,
    TaxonomyExpertMapping,
    classify_token,
    get_layer_phase,
    get_trigram,
)

# Test Data
from .test_data import (
    ATTENTION_ROUTING_CONTEXTS,
    CONTEXT_WINDOW_TESTS,
    DEFAULT_CONTEXTS,
    DOMAIN_PROMPTS,
    TAXONOMY_TEST_PROMPTS,
    TOKEN_CONTEXTS,
)

# Visualization
from .visualization import (
    multi_layer_routing_matrix,
    plot_expert_utilization,
    plot_multi_layer_heatmap,
    plot_routing_flow,
    plot_routing_heatmap,
    routing_heatmap_ascii,
    routing_weights_to_matrix,
    save_routing_heatmap,
    save_utilization_chart,
    utilization_bar_ascii,
)

__all__ = [
    # Enums
    "MoEArchitecture",
    "MoEAction",
    "ExpertCategory",
    "ExpertRole",
    # Config
    "MoECaptureConfig",
    "MoEAblationConfig",
    # Models - Core
    "MoELayerInfo",
    "MoEModelInfo",
    "RouterEntropy",
    "ExpertUtilization",
    "ExpertIdentity",
    "ExpertPair",
    "CoactivationAnalysis",
    "ExpertAblationResult",
    "CompressionPlan",
    # Models - Generation
    "GenerationStats",
    "ExpertChatResult",
    "ExpertComparisonResult",
    "TopKVariationResult",
    # Models - Router Weights
    "RouterWeightCapture",
    "LayerRouterWeights",
    # Models - Layer Analysis
    "LayerRoutingAnalysis",
    "LayerDivergenceResult",
    # Models - Pattern Discovery
    "ExpertPattern",
    "ExpertTaxonomy",
    # Models - Tokenizer Analysis
    "TokenExpertMapping",
    "VocabExpertAnalysis",
    # Detection
    "detect_moe_architecture",
    "get_moe_layer_info",
    "get_moe_layers",
    "is_moe_model",
    # Hooks
    "MoEHooks",
    "MoECapturedState",
    # Router analysis
    "analyze_coactivation",
    "compute_routing_diversity",
    "get_dominant_experts",
    "get_rare_experts",
    "compare_routing",
    # Datasets
    "PromptCategory",
    "PromptCategoryGroup",
    "CategoryPrompts",
    "CATEGORY_GROUPS",
    "get_category_prompts",
    "get_all_prompts",
    "get_grouped_prompts",
    "get_prompts_by_group",
    "get_prompts_flat",
    # Ablation
    "ablate_expert",
    "ablate_expert_batch",
    "find_causal_experts",
    "sweep_layer_experts",
    # Logit Lens
    "ExpertLogitContribution",
    "ExpertVocabContribution",
    "LayerRoutingSnapshot",
    "LayerVocabAnalysis",
    "MoELogitLens",
    "TokenExpertPreference",
    "VocabExpertMapping",
    "analyze_expert_vocabulary",
    "compute_expert_vocab_contribution",
    "compute_token_expert_mapping",
    "find_expert_specialists",
    "print_expert_vocab_summary",
    "print_token_expert_preferences",
    # Identification
    "CategoryActivation",
    "ExpertProfile",
    "identify_expert",
    "identify_all_experts",
    "find_specialists",
    "find_generalists",
    "cluster_experts_by_specialization",
    "print_expert_summary",
    # Compression
    "ExpertSimilarity",
    "ExpertActivationStats",
    "ActivationOverlapResult",
    "CompressionAnalysis",
    "compute_expert_similarity",
    "compute_expert_similarity_with_activations",
    "compute_similarity_matrix",
    "compute_similarity_matrix_with_activations",
    "collect_expert_activations",
    "compute_activation_overlap",
    "find_merge_candidates",
    "find_merge_candidates_with_activations",
    "find_prune_candidates",
    "create_compression_plan",
    "analyze_compression_opportunities",
    "print_compression_summary",
    "print_activation_overlap_matrix",
    # Expert Router
    "ExpertRouter",
    # Visualization
    "routing_weights_to_matrix",
    "multi_layer_routing_matrix",
    "plot_routing_heatmap",
    "plot_multi_layer_heatmap",
    "plot_expert_utilization",
    "plot_routing_flow",
    "routing_heatmap_ascii",
    "utilization_bar_ascii",
    "save_routing_heatmap",
    "save_utilization_chart",
    # Cross-layer tracking
    "ExpertPipelineNode",
    "ExpertPipeline",
    "LayerAlignmentResult",
    "CrossLayerAnalysis",
    "compute_expert_activation_profile",
    "compute_layer_alignment",
    "track_expert_across_layers",
    "identify_functional_pipelines",
    "analyze_cross_layer_routing",
    "print_pipeline_summary",
    "print_alignment_matrix",
]
