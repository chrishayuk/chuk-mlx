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

    ANALYZE = "analyze"
    """Analyze expert routing patterns across prompts."""

    CHAT = "chat"
    """Chat with a specific expert (force all routing to one expert)."""

    COMPARE = "compare"
    """Compare multiple experts on the same prompt."""

    ABLATE = "ablate"
    """Ablate (remove) an expert from routing."""

    TOPK = "topk"
    """Vary top-k expert selection."""

    COLLABORATION = "collab"
    """Analyze expert co-activation patterns."""

    PAIRS = "pairs"
    """Test specific expert pairs/groups."""

    INTERACTIVE = "interactive"
    """Interactive expert explorer mode."""

    WEIGHTS = "weights"
    """Show router weights for a prompt."""

    TOKENIZER = "tokenizer"
    """Analyze tokenizer-expert relationships."""

    CONTROL_TOKENS = "control-tokens"
    """Analyze control token expert assignments."""

    TRACE = "trace"
    """Trace token-level expert assignments."""

    ENTROPY = "entropy"
    """Analyze routing entropy across layers."""

    DIVERGENCE = "divergence"
    """Analyze layer divergence in routing."""

    ROLE = "role"
    """Analyze layer-specific roles."""

    CONTEXT_TEST = "context-test"
    """Test context independence of routing."""

    VOCAB_MAP = "vocab-map"
    """Map vocabulary to expert assignments."""

    ROUTER_PROBE = "router-probe"
    """Probe router inputs and outputs."""

    PATTERN_DISCOVERY = "pattern-discovery"
    """Discover expert activation patterns."""

    FULL_TAXONOMY = "full-taxonomy"
    """Generate full expert taxonomy."""

    LAYER_SWEEP = "layer-sweep"
    """Sweep analysis across all layers."""

    @property
    def handler_name(self) -> str:
        """Get the handler function/module name for this action."""
        return self.value.replace("-", "_")
