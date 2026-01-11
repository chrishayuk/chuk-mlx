"""Constants and enums for CLI commands.

This module centralizes all magic numbers, default values, and string constants
used across CLI commands. CLI commands should be thin wrappers and should not
contain hardcoded values.
"""

from __future__ import annotations

from enum import Enum

# Import shared constants from introspection module to avoid circular imports
# These are re-exported here for backwards compatibility
from chuk_lazarus.introspection._shared_constants import (
    Domain,
    LayerPhase,
    LayerPhaseDefaults,
    PatternCategory,
    TokenType,
)

# =============================================================================
# Layer and Position Enums
# =============================================================================


class LayerDepthRatio(float, Enum):
    """Common layer depth ratios used for analysis."""

    EARLY = 0.25
    MIDDLE = 0.5
    DECISION = 0.55
    LATE = 0.7
    DEEP = 0.8
    FINAL = 0.9


# =============================================================================
# Display and Formatting Enums
# =============================================================================


class DisplayDefaults(int, Enum):
    """Default values for display formatting."""

    SEPARATOR_WIDTH = 60
    PROBABILITY_BAR_WIDTH = 50
    ASCII_GRID_WIDTH = 60
    ASCII_GRID_HEIGHT = 20
    TABLE_COLUMN_WIDTH = 12
    LAYER_COLUMN_WIDTH = 8
    TOKEN_PREVIEW_LENGTH = 10
    PROGRESS_BAR_MAX = 50
    FORMATTER_WIDTH = 70


class HeatmapChars:
    """Characters for ASCII heatmap visualizations."""

    FILLED = "█"
    EMPTY = "░"
    HASH = "#"
    DASH = "-"
    GRADIENT = " .-+*#"


# =============================================================================
# Analysis Defaults
# =============================================================================


class AnalysisDefaults(int, Enum):
    """Default values for analysis operations."""

    TOP_K = 10
    TOP_K_LAYER = 5
    MAX_TOKENS = 20
    GEN_TOKENS = 30
    CROSS_VAL_FOLDS = 5
    MAX_ITERATIONS = 1000
    RANDOM_SEED = 42


class TrainingDefaults:
    """Default values for training operations."""

    # Common
    BATCH_SIZE: int = 4
    MAX_LENGTH: int = 512
    LOG_INTERVAL: int = 10
    LORA_RANK: int = 8

    # SFT
    SFT_EPOCHS: int = 3
    SFT_LEARNING_RATE: float = 1e-5

    # DPO
    DPO_EPOCHS: int = 3
    DPO_LEARNING_RATE: float = 1e-6
    DPO_BETA: float = 0.1

    # GRPO
    GRPO_ITERATIONS: int = 1000
    GRPO_PROMPTS_PER_ITERATION: int = 16
    GRPO_GROUP_SIZE: int = 4
    GRPO_LEARNING_RATE: float = 1e-6
    GRPO_KL_COEF: float = 0.1
    GRPO_MAX_RESPONSE_LENGTH: int = 256
    GRPO_TEMPERATURE: float = 1.0


class InferenceDefaults:
    """Default values for inference operations."""

    MAX_TOKENS: int = 100
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9


class MemoryDefaults:
    """Default values for memory analysis."""

    # Memorization thresholds
    MEMORIZED_PROB_THRESHOLD: float = 0.1
    PARTIAL_PROB_THRESHOLD: float = 0.01
    WEAK_PROB_THRESHOLD: float = 0.001
    MEMORIZED_RANK: int = 1
    PARTIAL_RANK: int = 5
    WEAK_RANK: int = 15

    # External memory
    BLEND: float = 1.0
    SIMILARITY_THRESHOLD: float = 0.7
    DEFAULT_QUERY_LAYER: int = 22
    DEFAULT_INJECT_LAYER: int = 21


class GymDefaults:
    """Default values for gym operations."""

    HOST: str = "localhost"
    PORT: int = 8023
    TRANSPORT: str = "telnet"
    OUTPUT_MODE: str = "json"
    DIFFICULTY: float = 0.5
    SUCCESS_RATE: float = 0.7
    BUFFER_SIZE: int = 10000
    TIMEOUT: float = 10.0
    TOKEN_BUDGET: int = 8192
    MAX_LENGTH: int = 2048


class CircuitDefaults:
    """Default values for circuit analysis."""

    THRESHOLD: float = 0.1
    DIRECTION: str = "TB"


class ProbeDefaults:
    """Default values for probing analysis."""

    RIDGE_ALPHA: float = 1.0
    LOGISTIC_MAX_ITER: int = 1000


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    DOT = "dot"
    MERMAID = "mermaid"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


class InvocationMethod(str, Enum):
    """Circuit invocation methods."""

    STEER = "steer"
    LINEAR = "linear"
    INTERPOLATE = "interpolate"
    EXTRAPOLATE = "extrapolate"


class DirectionMethod(str, Enum):
    """Direction extraction methods for probing."""

    LOGISTIC = "logistic"
    MEAN_DIFFERENCE = "mean_diff"


class OverrideMode(str, Enum):
    """Compute override modes for analysis."""

    NONE = "none"
    ARITHMETIC = "arithmetic"
    CUSTOM = "custom"


class LayerStrategy(str, Enum):
    """Strategies for selecting layers to analyze."""

    ALL = "all"
    EVENLY_SPACED = "evenly_spaced"
    SPECIFIC = "specific"
    CUSTOM = "custom"
    KEY_LAYERS = "key_layers"


class InputMode(str, Enum):
    """Input modes for commands."""

    SINGLE = "single"
    FILE = "file"
    INTERACTIVE = "interactive"
    BATCH = "batch"


class TrainMode(str, Enum):
    """Training modes."""

    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"


class DataGenType(str, Enum):
    """Data generation types."""

    MATH = "math"
    TOOL_CALL = "tool_call"
    PREFERENCE = "preference"


# =============================================================================
# Ablation Enums and Constants
# =============================================================================


class AblationCriterion(str, Enum):
    """Criterion types for ablation studies."""

    FUNCTION_CALL = "function_call"
    SORRY = "sorry"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    REFUSAL = "refusal"
    CONTAINS = "contains"  # Generic substring check


class AblationCriterionPatterns:
    """Patterns for ablation criterion matching."""

    FUNCTION_CALL_MARKERS: tuple[str, ...] = (
        "<start_function_call>",
        "<function_call>",
        "get_weather(",
        '{"name":',
    )
    SORRY_MARKERS: tuple[str, ...] = ("sorry", "apologize")
    POSITIVE_MARKERS: tuple[str, ...] = (
        "great",
        "good",
        "excellent",
        "wonderful",
        "love",
    )
    NEGATIVE_MARKERS: tuple[str, ...] = ("bad", "terrible", "awful", "hate", "poor")
    REFUSAL_MARKERS: tuple[str, ...] = ("cannot", "can't", "won't", "unable", "decline")


# =============================================================================
# Steering Enums and Constants
# =============================================================================


class SteeringDirectionFormat(str, Enum):
    """File formats for steering direction files."""

    NPZ = ".npz"
    JSON = ".json"


class SteeringDefaults:
    """Default values for steering operations."""

    DEFAULT_COEFFICIENT: float = 1.0
    DEFAULT_POSITIVE_LABEL: str = "positive"
    DEFAULT_NEGATIVE_LABEL: str = "negative"
    DEFAULT_NAME: str = "custom"


# =============================================================================
# MoE Analysis Enums and Constants
# =============================================================================
# TokenType, PatternCategory, Domain, LayerPhase, LayerPhaseDefaults
# are imported from introspection._shared_constants at the top of this file
# to avoid circular imports.


class ContextVerdict(str, Enum):
    """Context window analysis verdicts."""

    TRIGRAM_SUFFICIENT = "TRIGRAM SUFFICIENT"
    EXTENDED_CONTEXT_MATTERS = "EXTENDED CONTEXT MATTERS"
    MIXED = "MIXED"


class ContextType(str, Enum):
    """Context types for expert routing analysis."""

    NUMERIC = "numeric"
    AFTER_WORD = "after_word"
    AFTER_ARTICLE = "after_article"
    STANDALONE = "standalone"
    AFTER_OPERATOR = "after_operator"


class ExploreCommand(str, Enum):
    """Interactive explore REPL commands."""

    LAYER = "l"
    COMPARE = "c"
    ALL_LAYERS = "a"
    DEEP_DIVE = "d"
    QUIT = "q"


class MoEDefaults:
    """Default values for MoE analysis."""

    DEFAULT_LAYER: int = 11
    DEFAULT_TOKEN: str = "127"
    TOP_EXPERTS: int = 4
    EXPERT_DISPLAY_WIDTH: int = 6


# =============================================================================
# Embedding Analysis Defaults
# =============================================================================


class EmbeddingDefaults:
    """Default values for embedding analysis."""

    DEFAULT_LAYERS: tuple[int, ...] = (0, 1, 2)
    DIGIT_RANGE_START: int = 2
    DIGIT_RANGE_END: int = 8


# Common delimiters and separators
class Delimiters:
    """Common delimiter characters used in CLI parsing."""

    PROMPT_SEPARATOR: str = "|"
    LAYER_SEPARATOR: str = ","
    OPERAND_SEPARATOR: str = ","
    KEY_VALUE_SEPARATOR: str = ":"
    FILE_PREFIX: str = "@"


# Common format strings
class FormatStrings:
    """Common format strings for output."""

    LAYER_FORMAT: str = "L{layer}"
    PROBABILITY_FORMAT: str = "{prob:.4f}"
    PERCENTAGE_FORMAT: str = "{value:.1%}"
    ACCURACY_FORMAT: str = "{accuracy:.3f}"
    ERROR_FORMAT: str = "{error:+.1f}"


__all__ = [
    # Ablation
    "AblationCriterion",
    "AblationCriterionPatterns",
    # Analysis
    "AnalysisDefaults",
    # Circuit
    "CircuitDefaults",
    # Context
    "ContextType",
    "ContextVerdict",
    # Data
    "DataGenType",
    "Delimiters",
    # Direction
    "DirectionMethod",
    # Display
    "DisplayDefaults",
    # Domain
    "Domain",
    # Embedding
    "EmbeddingDefaults",
    # Explore
    "ExploreCommand",
    # Format
    "FormatStrings",
    # Gym
    "GymDefaults",
    # Heatmap
    "HeatmapChars",
    # Inference
    "InferenceDefaults",
    # Input
    "InputMode",
    # Invocation
    "InvocationMethod",
    # Layer
    "LayerDepthRatio",
    "LayerPhase",
    "LayerPhaseDefaults",
    "LayerStrategy",
    # Memory
    "MemoryDefaults",
    # MoE
    "MoEDefaults",
    # Output
    "OutputFormat",
    # Override
    "OverrideMode",
    # Pattern
    "PatternCategory",
    # Probe
    "ProbeDefaults",
    # Steering
    "SteeringDefaults",
    "SteeringDirectionFormat",
    # Token
    "TokenType",
    # Train
    "TrainMode",
    "TrainingDefaults",
]
