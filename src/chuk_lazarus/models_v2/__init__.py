"""
Unified Model Architecture for chuk-mlx.

A composable, async-native, Pydantic-native model framework supporting:
- Transformers (Llama, Mistral, Gemma, and compatible)
- State Space Models (Mamba)
- Recurrent Networks (LSTM, GRU, MinGRU)
- Hybrid Architectures
- Classifiers

Design Principles:
- Async-native: All I/O operations are async
- Pydantic-native: All configs use BaseModel for validation
- No magic strings: Enums and constants for type safety
- No dictionary goop: Structured types throughout
- Backend-agnostic: Works on MLX, PyTorch, JAX

Architecture:
    Components -> Blocks -> Backbones -> Heads -> Models
                                   \\       /
                                    Families

Components: Embeddings, Attention, FFN, Normalization, SSM, Recurrent
Blocks: Transformer, Mamba, Recurrent, Hybrid
Backbones: Stack of blocks with embeddings
Heads: LM, Classifier, Regression
Models: Complete end-to-end models
Families: Architecture-specific implementations (Llama, Mamba, etc.)
"""

# Core abstractions
# Backbones
# Adapters (LoRA, etc.)
from .adapters import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    merge_lora_weights,
)
from .backbones import (
    Backbone,
    BackboneOutput,
    HybridBackbone,
    MambaBackbone,
    RecurrentBackbone,
    TransformerBackbone,
)

# Blocks
from .blocks import (
    Block,
    BlockOutput,
    HybridBlock,
    MambaBlockWrapper,
    RecurrentBlockWrapper,
    TransformerBlock,
)
from .components.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    SlidingWindowAttention,
)

# Components
from .components.embeddings import (
    ALiBi,
    LearnedPositionEmbedding,
    RoPE,
    SinusoidalPositionEmbedding,
    TokenEmbedding,
)
from .components.ffn import GEGLU, MLP, MoE, SwiGLU
from .components.normalization import GemmaNorm, LayerNorm, RMSNorm
from .components.recurrent import GRU, LSTM, MinGRU
from .components.ssm import Mamba, MambaBlock, SelectiveSSM
from .core import (
    # Enums
    ActivationType,
    AttentionConfig,
    AttentionType,
    BackboneType,
    BackendType,
    BlockType,
    FFNConfig,
    FFNType,
    HeadType,
    # Registry
    ModelCapability,
    # Config
    ModelConfig,
    ModelMode,
    NormConfig,
    NormType,
    PositionEmbeddingType,
    SSMConfig,
    find_models_by_capability,
    # Backend
    get_backend,
    get_factory,
    get_model_capabilities,
    list_models,
    register_model,
    set_backend,
)

# Families
from .families import granite, llama, llama4, mamba
from .families.granite import (
    GraniteConfig,
    GraniteForCausalLM,
    GraniteHybridConfig,
    GraniteHybridForCausalLM,
)
from .families.llama import LlamaConfig, LlamaForCausalLM
from .families.llama4 import Llama4Config, Llama4ForCausalLM, Llama4TextConfig
from .families.mamba import MambaConfig, MambaForCausalLM

# Heads
from .heads import (
    ClassifierHead,
    Head,
    HeadOutput,
    LMHead,
    RegressionHead,
)

# Introspection
from .introspection import (
    FLOPsEstimate,
    MemoryEstimate,
    ModelCapabilities,
    ModelInfo,
    ParameterStats,
    count_parameters,
    detect_model_capabilities,
    estimate_flops,
    estimate_memory,
    get_model_info,
    introspect,
    print_introspection,
)

# Loader
from .loader import (
    create_from_preset,
    create_model,
    load_model,
    load_model_async,
)

# Loss functions
from .losses import compute_lm_loss

# Models
from .models import (
    CausalLM,
    Model,
    ModelOutput,
    SequenceClassifier,
    TokenClassifier,
)

__all__ = [
    # === Core ===
    # Enums
    "ModelMode",
    "BlockType",
    "BackboneType",
    "HeadType",
    "AttentionType",
    "FFNType",
    "NormType",
    "ActivationType",
    "PositionEmbeddingType",
    "BackendType",
    # Config
    "ModelConfig",
    "AttentionConfig",
    "FFNConfig",
    "NormConfig",
    "SSMConfig",
    # Registry
    "register_model",
    "get_factory",
    "list_models",
    "ModelCapability",
    "get_model_capabilities",
    "find_models_by_capability",
    # Backend
    "get_backend",
    "set_backend",
    # === Components ===
    # Embeddings
    "TokenEmbedding",
    "RoPE",
    "ALiBi",
    "LearnedPositionEmbedding",
    "SinusoidalPositionEmbedding",
    # Attention
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "SlidingWindowAttention",
    # FFN
    "MLP",
    "SwiGLU",
    "GEGLU",
    "MoE",
    # Normalization
    "RMSNorm",
    "LayerNorm",
    "GemmaNorm",
    # SSM
    "SelectiveSSM",
    "Mamba",
    "MambaBlock",
    # Recurrent
    "LSTM",
    "GRU",
    "MinGRU",
    # === Blocks ===
    "Block",
    "BlockOutput",
    "TransformerBlock",
    "MambaBlockWrapper",
    "RecurrentBlockWrapper",
    "HybridBlock",
    # === Backbones ===
    "Backbone",
    "BackboneOutput",
    "TransformerBackbone",
    "MambaBackbone",
    "RecurrentBackbone",
    "HybridBackbone",
    # === Heads ===
    "Head",
    "HeadOutput",
    "LMHead",
    "ClassifierHead",
    "RegressionHead",
    # === Models ===
    "Model",
    "ModelOutput",
    "CausalLM",
    "SequenceClassifier",
    "TokenClassifier",
    # === Families ===
    "granite",
    "llama",
    "llama4",
    "mamba",
    "GraniteConfig",
    "GraniteForCausalLM",
    "GraniteHybridConfig",
    "GraniteHybridForCausalLM",
    "LlamaConfig",
    "LlamaForCausalLM",
    "Llama4Config",
    "Llama4TextConfig",
    "Llama4ForCausalLM",
    "MambaConfig",
    "MambaForCausalLM",
    # === Loader ===
    "load_model",
    "load_model_async",
    "create_model",
    "create_from_preset",
    # === Adapters ===
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "merge_lora_weights",
    "count_lora_parameters",
    # === Introspection ===
    "ParameterStats",
    "FLOPsEstimate",
    "MemoryEstimate",
    "ModelCapabilities",
    "ModelInfo",
    "count_parameters",
    "estimate_flops",
    "estimate_memory",
    "get_model_capabilities",
    "detect_model_capabilities",
    "get_model_info",
    "introspect",
    "print_introspection",
    # === Losses ===
    "compute_lm_loss",
]
