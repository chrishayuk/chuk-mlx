"""MoE-specific enums."""

from enum import Enum


class MoEArchitecture(str, Enum):
    """Supported MoE architecture types."""

    GPT_OSS = "gpt_oss"
    """GPT-OSS: 32 experts, 4 active, MXFP4 quantized."""

    LLAMA4 = "llama4"
    """Llama 4: Shared expert (always active) + routed experts."""

    GRANITE_HYBRID = "granite_hybrid"
    """Granite Hybrid: MoE with Mamba-2/Attention hybrid."""

    MIXTRAL = "mixtral"
    """Mixtral: 8 experts, 2 active, standard routing."""

    GENERIC = "generic"
    """Generic MoE: Uses standard MoE component."""


class MoEImplementationType(str, Enum):
    """Internal MoE implementation type for expert routing."""

    NONE = "none"
    """No MoE layers detected."""

    GPT_OSS_BATCHED = "gpt_oss_batched"
    """GPT-OSS style with batched experts (gate_up_proj on experts)."""

    STANDARD = "standard"
    """Standard MoE implementation."""


class ExpertCategory(str, Enum):
    """Categories of expert specialization."""

    CODE = "code"
    MATH = "math"
    LANGUAGE = "language"
    PUNCTUATION = "punctuation"
    PROPER_NOUNS = "proper_nouns"
    FUNCTION_WORDS = "function_words"
    NUMBERS = "numbers"
    POSITION_FIRST = "position_first"
    POSITION_LAST = "position_last"
    GENERALIST = "generalist"
    UNKNOWN = "unknown"


class ExpertRole(str, Enum):
    """Roles that experts can play."""

    SPECIALIST = "specialist"
    """Expert specializes in specific token/domain type."""

    GENERALIST = "generalist"
    """Expert activates across many domains."""

    POSITIONAL = "positional"
    """Expert specializes by position (first/last tokens)."""

    RARE = "rare"
    """Expert rarely activates."""


class MoEType(str, Enum):
    """Classification of MoE training origin.

    Determines whether a model was converted from dense (pseudo-MoE)
    or trained natively as MoE. This affects compression strategies.
    """

    UPCYCLED = "upcycled"
    """Dense→MoE conversion (upcycling). Experts share a base with low-rank deltas. Compressible via SVD."""

    PRETRAINED_MOE = "pretrained_moe"
    """Trained natively as MoE from scratch. Orthogonal experts. Not compressible via SVD overlay."""

    UNKNOWN = "unknown"
    """Inconclusive metrics. Requires manual inspection."""

    # Legacy aliases for backwards compatibility
    PSEUDO = "upcycled"
    """Alias for UPCYCLED (deprecated)."""

    NATIVE = "pretrained_moe"
    """Alias for PRETRAINED_MOE (deprecated)."""


class MoEAction(str, Enum):
    """Available MoE expert CLI actions."""

    # Core analysis
    ANALYZE = "analyze"
    """Analyze expert routing patterns across prompts."""

    CHAT = "chat"
    """Chat with a specific expert (force all routing to one expert)."""

    COMPARE = "compare"
    """Compare multiple experts on the same prompt."""

    ABLATE = "ablate"
    """Ablate (remove) an expert from routing."""

    # Routing visualization
    WEIGHTS = "weights"
    """Show router weights for a prompt."""

    TRACE = "trace"
    """Trace token-level expert assignments across layers."""

    HEATMAP = "heatmap"
    """Generate routing heatmap visualization."""

    # Semantic trigram methodology
    FULL_TAXONOMY = "full-taxonomy"
    """Semantic trigram pattern analysis across categories."""

    DOMAIN_TEST = "domain-test"
    """Demonstrate that domain experts don't exist."""

    TOKEN_ROUTING = "token-routing"
    """Demonstrate that single token routing is context-dependent."""

    CONTEXT_TEST = "context-test"
    """Test context independence of routing."""

    CONTEXT_WINDOW = "context-window"
    """Test how much context the router actually uses (trigram vs full attention)."""

    ATTENTION_ROUTING = "attention-routing"
    """Analyze how attention patterns drive expert routing decisions."""

    ATTENTION_PATTERN = "attention-pattern"
    """Show attention weights for a specific position."""

    # Interactive
    EXPLORE = "explore"
    """Interactive expert explorer for real-time analysis."""

    # MoE Type Detection
    MOE_TYPE_ANALYZE = "moe-type-analyze"
    """Analyze MoE type (pseudo vs native) via SVD analysis."""

    MOE_TYPE_COMPARE = "moe-type-compare"
    """Compare MoE types between two models."""

    # MoE Compression
    MOE_OVERLAY_COMPUTE = "moe-overlay-compute"
    """Compute overlay representation (base + low-rank deltas)."""

    MOE_OVERLAY_VERIFY = "moe-overlay-verify"
    """Verify reconstruction accuracy of overlay representation."""

    MOE_OVERLAY_ESTIMATE = "moe-overlay-estimate"
    """Estimate storage savings from overlay compression."""

    MOE_OVERLAY_COMPRESS = "moe-overlay-compress"
    """Compress model to overlay format (base + low-rank deltas)."""

    # Expert Dynamics Analysis
    COLD_EXPERTS = "cold-experts"
    """Analyze cold (rarely-activated) experts and their triggers."""

    GENERATION_DYNAMICS = "generation-dynamics"
    """Analyze expert routing during autoregressive generation."""

    EXPERT_CIRCUITS = "expert-circuits"
    """Identify cross-layer expert circuits."""

    EXPERT_INTERFERENCE = "expert-interference"
    """Analyze multi-expert interference patterns."""

    EXPERT_MERGING = "expert-merging"
    """Analyze expert merging opportunities."""

    ATTENTION_PREDICTION = "attention-prediction"
    """Predict routing from attention patterns (routing without router)."""

    TASK_PREDICTION = "task-prediction"
    """Predict needed experts from early-layer activations (task-aware loading)."""

    ROUTING_MANIPULATION = "routing-manipulation"
    """Analyze and craft inputs for specific expert routing patterns."""

    CONTEXT_ATTENTION_ROUTING = "context-attention-routing"
    """Analyze attention-routing correlation with context awareness."""

    @property
    def handler_name(self) -> str:
        """Get the handler function/module name for this action."""
        return self.value.replace("-", "_")
