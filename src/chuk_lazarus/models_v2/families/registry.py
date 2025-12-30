"""
Model family registry for unified HuggingFace model loading.

Each model family provides:
- Config class (Pydantic BaseModel)
- Model class (CausalLM, etc.)
- Weight converter (HF -> our format)
- Optional: Introspection hooks, chat template, special tokens

The registry auto-detects the model family from HuggingFace config.json
and provides a unified interface for loading any supported model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .constants import HFModelType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelFamilyType(str, Enum):
    """Supported model family types."""

    LLAMA = "llama"
    LLAMA4 = "llama4"
    GEMMA = "gemma"
    GRANITE = "granite"
    GRANITE_HYBRID = "granitemoehybrid"
    JAMBA = "jamba"
    MAMBA = "mamba"
    STARCODER2 = "starcoder2"
    QWEN3 = "qwen3"
    GPT2 = "gpt2"
    GPT_NEO = "gpt_neo"
    GPT_NEOX = "gpt_neox"
    GPT_OSS = "gpt_oss"


# Model type patterns for auto-detection from config.json
MODEL_TYPE_PATTERNS = {
    # Exact matches first
    "llama4": ModelFamilyType.LLAMA4,
    "granitemoehybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite_hybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite-hybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite": ModelFamilyType.GRANITE,
    "jamba": ModelFamilyType.JAMBA,
    "mamba": ModelFamilyType.MAMBA,
    "starcoder2": ModelFamilyType.STARCODER2,
    "qwen3": ModelFamilyType.QWEN3,
    "qwen2": ModelFamilyType.QWEN3,  # Qwen2 uses same arch
    "gpt2": ModelFamilyType.GPT2,
    "gpt_neo": ModelFamilyType.GPT_NEO,
    "gpt_neox": ModelFamilyType.GPT_NEOX,
    "gpt-neo": ModelFamilyType.GPT_NEO,
    "gpt-neox": ModelFamilyType.GPT_NEOX,
    "gpt_oss": ModelFamilyType.GPT_OSS,
    # Gemma variants
    "gemma": ModelFamilyType.GEMMA,
    "gemma2": ModelFamilyType.GEMMA,
    "gemma3": ModelFamilyType.GEMMA,  # May have nested text_config
    "gemma3_text": ModelFamilyType.GEMMA,
    "gemma-3": ModelFamilyType.GEMMA,
    # Llama-compatible (last, as catch-all for llama-like models)
    "llama": ModelFamilyType.LLAMA,
    "mistral": ModelFamilyType.LLAMA,
    "mixtral": ModelFamilyType.LLAMA,
    "codellama": ModelFamilyType.LLAMA,
}


# Architecture names for auto-detection
ARCHITECTURE_PATTERNS = {
    "Llama4ForCausalLM": ModelFamilyType.LLAMA4,
    "Llama4ForConditionalGeneration": ModelFamilyType.LLAMA4,
    "LlamaForCausalLM": ModelFamilyType.LLAMA,
    "MistralForCausalLM": ModelFamilyType.LLAMA,
    "MixtralForCausalLM": ModelFamilyType.LLAMA,
    "GemmaForCausalLM": ModelFamilyType.GEMMA,
    "Gemma2ForCausalLM": ModelFamilyType.GEMMA,
    "Gemma3ForCausalLM": ModelFamilyType.GEMMA,
    "PaliGemmaForConditionalGeneration": ModelFamilyType.GEMMA,
    "GraniteForCausalLM": ModelFamilyType.GRANITE,
    "GraniteMoeHybridForCausalLM": ModelFamilyType.GRANITE_HYBRID,
    "JambaForCausalLM": ModelFamilyType.JAMBA,
    "MambaForCausalLM": ModelFamilyType.MAMBA,
    "Starcoder2ForCausalLM": ModelFamilyType.STARCODER2,
    "Qwen2ForCausalLM": ModelFamilyType.QWEN3,
    "Qwen3ForCausalLM": ModelFamilyType.QWEN3,
    "GPT2LMHeadModel": ModelFamilyType.GPT2,
    "GPTNeoForCausalLM": ModelFamilyType.GPT_NEO,
    "GPTNeoXForCausalLM": ModelFamilyType.GPT_NEOX,
    "GptOssForCausalLM": ModelFamilyType.GPT_OSS,
}


@runtime_checkable
class WeightConverter(Protocol):
    """Protocol for weight name converters."""

    def convert(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Convert weight names from HF format to our format."""
        ...

    def reverse_convert(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Convert weight names from our format back to HF format."""
        ...


@dataclass
class IntrospectionHooks:
    """Hooks for model introspection during inference.

    These hooks enable:
    - Logit lens analysis
    - Activation patching
    - Attention pattern extraction
    - Layer-wise feature analysis
    """

    # Called after each layer
    layer_output_hook: Any | None = None

    # Called after attention computation
    attention_hook: Any | None = None

    # Called after FFN computation
    ffn_hook: Any | None = None

    # Called after final norm (before LM head)
    pre_head_hook: Any | None = None

    # Called with the final logits
    logits_hook: Any | None = None

    # Track which layers to hook (None = all)
    layer_indices: list[int] | None = None


@dataclass
class FamilyInfo:
    """Information about a model family."""

    family_type: ModelFamilyType
    config_class: type[BaseModel]
    model_class: type
    weight_converter: WeightConverter | None = None

    # Model metadata
    model_types: list[str] = field(default_factory=list)
    architectures: list[str] = field(default_factory=list)

    # Capabilities
    supports_chat: bool = True
    supports_kv_cache: bool = True
    supports_introspection: bool = True

    # Default introspection hooks creator
    create_hooks: Any | None = None


class ModelFamilyRegistry:
    """Registry for model families.

    Provides auto-detection and unified loading for any supported model.
    """

    _instance: ModelFamilyRegistry | None = None
    _families: dict[ModelFamilyType, FamilyInfo]

    def __init__(self) -> None:
        self._families = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> ModelFamilyRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> None:
        """Lazy initialization of family registry."""
        if self._initialized:
            return

        # Import and register all families
        self._register_builtin_families()
        self._initialized = True

    def _register_builtin_families(self) -> None:
        """Register all built-in model families."""
        # Import families lazily to avoid circular imports
        from . import gemma, gpt2, gpt_oss, granite, jamba, llama, llama4, mamba, qwen3, starcoder2

        # Llama family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.LLAMA,
                config_class=llama.LlamaConfig,
                model_class=llama.LlamaForCausalLM,
                model_types=[
                    HFModelType.LLAMA.value,
                    HFModelType.MISTRAL.value,
                    HFModelType.MIXTRAL.value,
                    HFModelType.CODELLAMA.value,
                ],
                architectures=["LlamaForCausalLM", "MistralForCausalLM"],
            )
        )

        # Llama4 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.LLAMA4,
                config_class=llama4.Llama4Config,
                model_class=llama4.Llama4ForCausalLM,
                model_types=[HFModelType.LLAMA4.value],
                architectures=["Llama4ForCausalLM"],
            )
        )

        # Gemma family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GEMMA,
                config_class=gemma.GemmaConfig,
                model_class=gemma.GemmaForCausalLM,
                model_types=[
                    HFModelType.GEMMA.value,
                    HFModelType.GEMMA2.value,
                    HFModelType.GEMMA3.value,
                    HFModelType.GEMMA3_TEXT.value,
                ],
                architectures=[
                    "GemmaForCausalLM",
                    "Gemma2ForCausalLM",
                    "Gemma3ForCausalLM",
                    "Gemma3ForConditionalGeneration",  # Text models with VLM wrapper
                ],
            )
        )

        # Granite family (dense)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GRANITE,
                config_class=granite.GraniteConfig,
                model_class=granite.GraniteForCausalLM,
                model_types=[HFModelType.GRANITE.value],
                architectures=["GraniteForCausalLM"],
            )
        )

        # Granite hybrid family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GRANITE_HYBRID,
                config_class=granite.GraniteHybridConfig,
                model_class=granite.GraniteHybridForCausalLM,
                model_types=[HFModelType.GRANITE_MOE_HYBRID.value],
                architectures=["GraniteMoeHybridForCausalLM"],
            )
        )

        # Jamba family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.JAMBA,
                config_class=jamba.JambaConfig,
                model_class=jamba.JambaForCausalLM,
                model_types=[HFModelType.JAMBA.value],
                architectures=["JambaForCausalLM"],
            )
        )

        # Mamba family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.MAMBA,
                config_class=mamba.MambaConfig,
                model_class=mamba.MambaForCausalLM,
                model_types=[HFModelType.MAMBA.value],
                architectures=["MambaForCausalLM"],
            )
        )

        # StarCoder2 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.STARCODER2,
                config_class=starcoder2.StarCoder2Config,
                model_class=starcoder2.StarCoder2ForCausalLM,
                model_types=[HFModelType.STARCODER2.value],
                architectures=["Starcoder2ForCausalLM"],
            )
        )

        # Qwen3 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.QWEN3,
                config_class=qwen3.Qwen3Config,
                model_class=qwen3.Qwen3ForCausalLM,
                model_types=[HFModelType.QWEN2.value, HFModelType.QWEN3.value],
                architectures=["Qwen2ForCausalLM", "Qwen3ForCausalLM"],
            )
        )

        # GPT-2 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GPT2,
                config_class=gpt2.GPT2Config,
                model_class=gpt2.GPT2ForCausalLM,
                model_types=[HFModelType.GPT2.value],
                architectures=["GPT2LMHeadModel"],
            )
        )

        # GPT-OSS family (OpenAI open-source MoE)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GPT_OSS,
                config_class=gpt_oss.GptOssConfig,
                model_class=gpt_oss.GptOssForCausalLM,
                model_types=[HFModelType.GPT_OSS.value],
                architectures=["GptOssForCausalLM"],
            )
        )

    def register(self, family_info: FamilyInfo) -> None:
        """Register a model family."""
        self._families[family_info.family_type] = family_info
        logger.debug(f"Registered model family: {family_info.family_type.value}")

    def get_family(self, family_type: ModelFamilyType) -> FamilyInfo | None:
        """Get family info by type."""
        self._ensure_initialized()
        return self._families.get(family_type)

    def detect_family(self, config: dict[str, Any]) -> ModelFamilyType | None:
        """Auto-detect model family from HuggingFace config.

        Args:
            config: Dict loaded from config.json

        Returns:
            Detected ModelFamilyType or None
        """
        self._ensure_initialized()

        # Try model_type first
        model_type = config.get("model_type", "").lower()
        if model_type in MODEL_TYPE_PATTERNS:
            return MODEL_TYPE_PATTERNS[model_type]

        # Try architectures
        architectures = config.get("architectures", [])
        for arch in architectures:
            if arch in ARCHITECTURE_PATTERNS:
                return ARCHITECTURE_PATTERNS[arch]

        # Try to infer from config structure
        # Granite hybrid has specific keys
        if "mamba_d_state" in config and "attn_layer_period" in config:
            return ModelFamilyType.GRANITE_HYBRID

        # Mamba has ssm-specific keys
        if "d_state" in config and "d_conv" in config:
            return ModelFamilyType.MAMBA

        # Gemma has sliding_window_pattern
        if "sliding_window_pattern" in config:
            return ModelFamilyType.GEMMA

        # Jamba has hybrid pattern keys
        if "attn_layer_period" in config and "expert_layer_period" in config:
            return ModelFamilyType.JAMBA

        return None

    def detect_family_from_path(self, model_path: Path) -> ModelFamilyType | None:
        """Detect family from model path."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        return self.detect_family(config)

    def list_families(self) -> list[ModelFamilyType]:
        """List all registered families."""
        self._ensure_initialized()
        return list(self._families.keys())


# Global registry instance
_registry = ModelFamilyRegistry.get_instance()


def get_family_registry() -> ModelFamilyRegistry:
    """Get the global family registry."""
    return _registry


def detect_model_family(config: dict[str, Any]) -> ModelFamilyType | None:
    """Detect model family from config dict."""
    return _registry.detect_family(config)


def get_family_info(family_type: ModelFamilyType) -> FamilyInfo | None:
    """Get family info by type."""
    return _registry.get_family(family_type)


def list_model_families() -> list[ModelFamilyType]:
    """List all registered model families."""
    return _registry.list_families()
