"""
Inference and text generation utilities.

Provides a high-level API for loading and running inference
with any supported model family.

Example usage:

    from chuk_lazarus.inference import InferencePipeline
    from chuk_lazarus.models_v2 import LlamaForCausalLM, LlamaConfig

    # One-liner setup
    pipeline = InferencePipeline.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        LlamaForCausalLM,
        LlamaConfig,
    )

    # Chat
    response = pipeline.chat("What is the capital of France?")
    print(response.text)
"""

# Core generation
# Chat utilities
from .chat import (
    ASSISTANT_SUFFIX,
    ChatHistory,
    ChatMessage,
    FallbackTemplate,
    Role,
    format_chat_prompt,
    format_history,
)

# Generation utilities
from .generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    generate,
    generate_stream,
    get_stop_tokens,
)
from .generator import generate_response, generate_sequence

# Loader utilities
from .loader import (
    DownloadConfig,
    DownloadResult,
    DType,
    HFLoader,
    LoadedWeights,
    StandardWeightConverter,
    WeightConverter,
)

# High-level pipeline
from .pipeline import (
    InferencePipeline,
    PipelineConfig,
    PipelineState,
)

__all__ = [
    # Legacy
    "generate_response",
    "generate_sequence",
    # Loader
    "DownloadConfig",
    "DownloadResult",
    "DType",
    "HFLoader",
    "LoadedWeights",
    "StandardWeightConverter",
    "WeightConverter",
    # Chat
    "ASSISTANT_SUFFIX",
    "ChatHistory",
    "ChatMessage",
    "FallbackTemplate",
    "Role",
    "format_chat_prompt",
    "format_history",
    # Generation
    "GenerationConfig",
    "GenerationResult",
    "GenerationStats",
    "generate",
    "generate_stream",
    "get_stop_tokens",
    # Pipeline
    "InferencePipeline",
    "PipelineConfig",
    "PipelineState",
]
