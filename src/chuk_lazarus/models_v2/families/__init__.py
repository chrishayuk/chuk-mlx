"""
Model family implementations.

Each family provides:
- Configuration class (Pydantic BaseModel)
- Model implementation (CausalLM)
- Weight conversion utilities (HF -> our format)
- Registry integration for auto-detection

Available families:
- gemma: Gemma 3, FunctionGemma
- gpt2: GPT-2 and compatible models
- granite: IBM Granite 3.x/4.x with hybrid Mamba-2/Transformer
- jamba: Jamba hybrid Mamba-Transformer MoE from AI21 Labs
- llama: Llama 1/2/3, Mistral, and compatible models
- llama4: Llama 4 with MoE and multimodal support
- mamba: Mamba SSM models
- qwen3: Qwen 2/3 models
- starcoder2: StarCoder2 code generation models

Usage:
    from chuk_lazarus.models_v2.families import detect_model_family, get_family_info

    # Auto-detect from config.json
    family_type = detect_model_family(hf_config)
    family_info = get_family_info(family_type)

    # Create model
    config = family_info.config_class.from_hf_config(hf_config)
    model = family_info.model_class(config)
"""

from . import gemma, gpt2, granite, jamba, llama, llama4, mamba, qwen3, starcoder2
from .registry import (
    FamilyInfo,
    IntrospectionHooks,
    ModelFamilyRegistry,
    ModelFamilyType,
    WeightConverter,
    detect_model_family,
    get_family_info,
    get_family_registry,
    list_model_families,
)

__all__ = [
    # Families
    "gemma",
    "gpt2",
    "granite",
    "jamba",
    "llama",
    "llama4",
    "mamba",
    "qwen3",
    "starcoder2",
    # Registry
    "ModelFamilyType",
    "ModelFamilyRegistry",
    "FamilyInfo",
    "WeightConverter",
    "IntrospectionHooks",
    "detect_model_family",
    "get_family_info",
    "get_family_registry",
    "list_model_families",
]
