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
    CompressionAnalysis,
    ExpertSimilarity,
    analyze_compression_opportunities,
    compute_expert_similarity,
    compute_similarity_matrix,
    create_compression_plan,
    find_merge_candidates,
    find_prune_candidates,
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
    LayerRoutingSnapshot,
    MoELogitLens,
    analyze_expert_vocabulary,
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
    "LayerRoutingSnapshot",
    "MoELogitLens",
    "analyze_expert_vocabulary",
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
    "CompressionAnalysis",
    "compute_expert_similarity",
    "compute_similarity_matrix",
    "find_merge_candidates",
    "find_prune_candidates",
    "create_compression_plan",
    "analyze_compression_opportunities",
    "print_compression_summary",
    # Expert Router
    "ExpertRouter",
]
