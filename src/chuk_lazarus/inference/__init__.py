"""
Inference and text generation utilities.

Provides a high-level API for loading and running inference
with any supported model family.

Recommended: Use UnifiedPipeline which auto-detects model family:

    from chuk_lazarus.inference import UnifiedPipeline

    # One-liner for any supported model - auto-detects family!
    pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Chat
    response = pipeline.chat("What is the capital of France?")
    print(response.text)

    # Generation with custom settings
    response = pipeline.generate(
        "Write a poem about AI",
        max_new_tokens=200,
        temperature=0.9,
    )
"""

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

# Unified pipeline (recommended)
from .unified import (
    IntrospectionResult,
    UnifiedPipeline,
    UnifiedPipelineConfig,
    UnifiedPipelineState,
)

__all__ = [
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
    # Unified Pipeline (recommended)
    "UnifiedPipeline",
    "UnifiedPipelineConfig",
    "UnifiedPipelineState",
    "IntrospectionResult",
]
