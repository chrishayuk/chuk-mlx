"""
Tokenizer Preprocessing Module

Pre-tokenization transforms for robust tokenization:
- Numeric normalization and encoding
- Structure token injection (JSON, UUID, paths)
- Pre/post tokenization hooks
- Tokenizer profiles (training vs inference)
- Byte fallback wrapper
"""

from chuk_lazarus.data.tokenizers.preprocessing.fallback import (
    ByteFallbackConfig,
    ByteFallbackStats,
    ByteFallbackWrapper,
    ensure_byte_safety,
    run_byte_safety_tests,
    wrap_with_fallback,
)
from chuk_lazarus.data.tokenizers.preprocessing.hooks import (
    HookedTokenizer,
    HookPipeline,
    PostDecodeHook,
    PreTokenizeHook,
    create_math_pipeline,
    create_standard_pipeline,
    create_tool_pipeline,
)
from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
    NumericConfig,
    NumericEncoding,
    NumericSpan,
    decode_number,
    detect_numbers,
    encode_number,
    normalize_numbers,
    restore_numbers,
)
from chuk_lazarus.data.tokenizers.preprocessing.profiles import (
    ProfiledTokenizer,
    TokenizerProfile,
    create_inference_profile,
    create_training_profile,
)
from chuk_lazarus.data.tokenizers.preprocessing.structure import (
    StructureConfig,
    StructureSpan,
    StructureType,
    detect_structures,
    inject_structure_tokens,
    restore_structures,
)

__all__ = [
    # Numeric
    "NumericConfig",
    "NumericSpan",
    "NumericEncoding",
    "detect_numbers",
    "encode_number",
    "decode_number",
    "normalize_numbers",
    "restore_numbers",
    # Structure
    "StructureConfig",
    "StructureSpan",
    "StructureType",
    "detect_structures",
    "inject_structure_tokens",
    "restore_structures",
    # Hooks
    "PreTokenizeHook",
    "PostDecodeHook",
    "HookPipeline",
    "HookedTokenizer",
    "create_standard_pipeline",
    "create_math_pipeline",
    "create_tool_pipeline",
    # Profiles
    "TokenizerProfile",
    "ProfiledTokenizer",
    "create_training_profile",
    "create_inference_profile",
    # Fallback
    "ByteFallbackConfig",
    "ByteFallbackStats",
    "ByteFallbackWrapper",
    "ensure_byte_safety",
    "run_byte_safety_tests",
    "wrap_with_fallback",
]
