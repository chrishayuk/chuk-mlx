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

    @property
    def handler_name(self) -> str:
        """Get the handler function/module name for this action."""
        return self.value.replace("-", "_")
