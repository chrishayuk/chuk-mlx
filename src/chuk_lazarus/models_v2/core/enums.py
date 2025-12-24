"""
Enums for type-safe model configuration.

No magic strings - all configuration options are typed enums.
"""

from enum import Enum


class ModelMode(str, Enum):
    """Model operating mode."""

    TRAIN = "train"
    INFERENCE = "inference"
    EVAL = "eval"

    def __str__(self) -> str:
        return self.value


class BlockType(str, Enum):
    """Types of model blocks."""

    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    LSTM = "lstm"
    GRU = "gru"
    MINGRU = "mingru"
    CONV = "conv"
    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value


class BackboneType(str, Enum):
    """Types of model backbones."""

    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    RECURRENT = "recurrent"
    HYBRID = "hybrid"
    ENCODER_DECODER = "encoder_decoder"

    def __str__(self) -> str:
        return self.value


class HeadType(str, Enum):
    """Types of output heads."""

    LM = "lm"  # Language modeling (next token prediction)
    CLASSIFIER = "classifier"  # Classification
    REGRESSION = "regression"  # Regression
    SEQUENCE_LABELING = "sequence_labeling"  # Token-level classification
    CONTRASTIVE = "contrastive"  # Contrastive learning

    def __str__(self) -> str:
        return self.value


class AttentionType(str, Enum):
    """Types of attention mechanisms."""

    MULTI_HEAD = "multi_head"  # Standard MHA
    GROUPED_QUERY = "grouped_query"  # GQA (Llama 2+)
    MULTI_QUERY = "multi_query"  # MQA
    MULTI_LATENT = "multi_latent"  # MLA (DeepSeek)
    SLIDING_WINDOW = "sliding_window"  # Sliding window (Mistral)
    LINEAR = "linear"  # Linear attention
    FLASH = "flash"  # Flash attention (same math, optimized)

    def __str__(self) -> str:
        return self.value


class NormType(str, Enum):
    """Types of normalization layers."""

    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"
    GEMMA_NORM = "gemma_norm"  # RMSNorm with +1 offset
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class ActivationType(str, Enum):
    """Types of activation functions."""

    SILU = "silu"  # SiLU / Swish
    GELU = "gelu"
    GELU_APPROX = "gelu_approx"  # Fast GELU approximation
    RELU = "relu"
    RELU2 = "relu2"  # ReLU squared
    TANH = "tanh"
    SIGMOID = "sigmoid"
    NONE = "none"  # Linear / identity

    def __str__(self) -> str:
        return self.value


class PositionEmbeddingType(str, Enum):
    """Types of position embeddings."""

    ROPE = "rope"  # Rotary position embeddings
    ALIBI = "alibi"  # ALiBi
    LEARNED = "learned"  # Learned absolute positions
    SINUSOIDAL = "sinusoidal"  # Sinusoidal (original Transformer)
    RELATIVE = "relative"  # Relative position bias
    NONE = "none"  # No position embeddings

    def __str__(self) -> str:
        return self.value


class PoolingType(str, Enum):
    """Types of sequence pooling for classification."""

    CLS = "cls"  # Use [CLS] token
    MEAN = "mean"  # Mean pooling
    MAX = "max"  # Max pooling
    LAST = "last"  # Use last token
    FIRST = "first"  # Use first token

    def __str__(self) -> str:
        return self.value


class FFNType(str, Enum):
    """Types of feed-forward networks."""

    MLP = "mlp"  # Standard MLP
    SWIGLU = "swiglu"  # SwiGLU (Llama)
    GEGLU = "geglu"  # GEGLU
    GATED = "gated"  # Generic gated MLP
    MOE = "moe"  # Mixture of Experts

    def __str__(self) -> str:
        return self.value


class SSMType(str, Enum):
    """Types of state space models."""

    MAMBA = "mamba"  # Mamba selective scan
    MAMBA2 = "mamba2"  # Mamba2
    S4 = "s4"  # Structured SSM
    S4D = "s4d"  # Diagonal S4
    H3 = "h3"  # H3

    def __str__(self) -> str:
        return self.value


class RecurrentType(str, Enum):
    """Types of recurrent cells."""

    LSTM = "lstm"
    GRU = "gru"
    MINGRU = "mingru"  # Minimal GRU
    RNN = "rnn"  # Vanilla RNN

    def __str__(self) -> str:
        return self.value


class InitType(str, Enum):
    """Types of weight initialization."""

    NORMAL = "normal"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    ORTHOGONAL = "orthogonal"
    ZEROS = "zeros"
    ONES = "ones"

    def __str__(self) -> str:
        return self.value


class HybridMixStrategy(str, Enum):
    """Strategies for mixing attention and SSM layers in hybrid models."""

    ALTERNATING = "alternating"  # Alternate between attention and Mamba layers
    INTERLEAVED = "interleaved"  # Every Nth layer is attention, rest are Mamba
    PARALLEL = "parallel"  # Each layer has both attention and Mamba in parallel

    def __str__(self) -> str:
        return self.value


class HybridCombineMode(str, Enum):
    """Modes for combining outputs in hybrid blocks."""

    CONCAT = "concat"  # Concatenate and project
    GATE = "gate"  # Learnable gating between streams
    ADD = "add"  # Simple addition

    def __str__(self) -> str:
        return self.value


class ClassificationTask(str, Enum):
    """Types of classification tasks."""

    SEQUENCE = "sequence"  # Sequence-level classification
    TOKEN = "token"  # Token-level classification (NER, POS, etc.)

    def __str__(self) -> str:
        return self.value


class DtRank(str, Enum):
    """Delta time rank options for SSM."""

    AUTO = "auto"  # Automatically compute dt_rank

    def __str__(self) -> str:
        return self.value
