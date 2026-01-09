"""
Centralized model loading for Lazarus.

This is THE single entry point for loading models - used by:
- Inference (UnifiedPipeline)
- Training (SFT, GRPO, DPO, DualReward)
- Introspection (ModelAnalyzer, AblationStudy)

Design principles:
- Pydantic-native: All configs use BaseModel for validation
- Async-native: Primary API is async, sync wrappers provided
- No dictionary goop: Structured return types
- No magic strings: Use enums for dtype, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .adapters.lora import LoRAConfig, LoRALinear

logger = logging.getLogger(__name__)


class ModelDType(str, Enum):
    """Supported data types for model weights."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    def to_mlx(self) -> mx.Dtype:
        """Convert to MLX dtype."""
        return {
            ModelDType.FLOAT16: mx.float16,
            ModelDType.FLOAT32: mx.float32,
            ModelDType.BFLOAT16: mx.bfloat16,
        }[self]


class LoadedModel(BaseModel):
    """Result of loading a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(..., description="The loaded model instance")
    tokenizer: Any = Field(..., description="The tokenizer")
    config: Any = Field(..., description="Model configuration")
    model_path: Path = Field(..., description="Path to model files")
    family_type: str = Field(..., description="Detected model family")


class LoadedModelWithLoRA(LoadedModel):
    """Result of loading a model with LoRA adapters."""

    lora_layers: dict[str, Any] = Field(
        default_factory=dict,
        description="Applied LoRA layers by name",
    )
    lora_parameter_count: int = Field(0, description="Total trainable LoRA parameters")


class AdapterConfig(BaseModel):
    """Configuration for loading adapter weights."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter_path: Path = Field(..., description="Path to adapter directory")
    rank: int = Field(8, description="LoRA rank")
    alpha: float = Field(16.0, description="LoRA alpha scaling")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Modules to apply LoRA to",
    )

    @classmethod
    def from_directory(cls, adapter_path: Path | str) -> AdapterConfig:
        """Load adapter config from directory."""
        adapter_path = Path(adapter_path)
        config_path = adapter_path / "adapter_config.json"

        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)

            # Handle different config formats
            lora_params = data.get("lora_parameters", data)
            return cls(
                adapter_path=adapter_path,
                rank=lora_params.get("rank", lora_params.get("lora_rank", 8)),
                alpha=lora_params.get("alpha", lora_params.get("lora_alpha", 16.0)),
                target_modules=lora_params.get(
                    "target_modules",
                    lora_params.get("lora_targets", ["q_proj", "v_proj"]),
                ),
            )

        return cls(adapter_path=adapter_path)


# ============================================================================
# Async API (Primary)
# ============================================================================


async def load_model_async(
    model_id: str,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModel:
    """
    Load a model asynchronously.

    This is the primary entry point for loading models. It:
    1. Downloads from HuggingFace if needed
    2. Detects model family automatically
    3. Loads weights and tokenizer
    4. Optionally applies adapter weights

    Args:
        model_id: HuggingFace model ID or local path
        dtype: Data type for weights
        adapter_path: Optional path to LoRA adapter directory

    Returns:
        LoadedModel with model, tokenizer, and config

    Example:
        >>> result = await load_model_async("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> model, tokenizer, config = result.model, result.tokenizer, result.config
    """
    loop = asyncio.get_event_loop()

    # Run the synchronous load in executor
    result = await loop.run_in_executor(
        None,
        lambda: _load_model_impl(model_id, dtype=dtype, adapter_path=adapter_path),
    )

    return result


async def load_model_with_lora_async(
    model_id: str,
    lora_config: LoRAConfig,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModelWithLoRA:
    """
    Load a model and apply LoRA adapters asynchronously.

    Use this for training - creates fresh LoRA layers for fine-tuning.
    Optionally loads pre-trained adapter weights.

    Args:
        model_id: HuggingFace model ID or local path
        lora_config: LoRA configuration
        dtype: Data type for weights
        adapter_path: Optional path to pre-trained adapter weights

    Returns:
        LoadedModelWithLoRA with model, tokenizer, config, and lora_layers

    Example:
        >>> from chuk_lazarus.models_v2 import LoRAConfig
        >>> lora_cfg = LoRAConfig(rank=16, target_modules=["v_proj", "o_proj"])
        >>> result = await load_model_with_lora_async("TinyLlama/...", lora_cfg)
        >>> lora_layers = result.lora_layers
    """
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None,
        lambda: _load_model_with_lora_impl(
            model_id, lora_config, dtype=dtype, adapter_path=adapter_path
        ),
    )

    return result


# ============================================================================
# Sync API (Convenience wrappers)
# ============================================================================


def load_model(
    model_id: str,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModel:
    """
    Load a model synchronously.

    Convenience wrapper around load_model_async.

    Args:
        model_id: HuggingFace model ID or local path
        dtype: Data type for weights
        adapter_path: Optional path to LoRA adapter directory

    Returns:
        LoadedModel with model, tokenizer, and config
    """
    return _load_model_impl(model_id, dtype=dtype, adapter_path=adapter_path)


def load_model_with_lora(
    model_id: str,
    lora_config: LoRAConfig,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModelWithLoRA:
    """
    Load a model and apply LoRA adapters synchronously.

    Convenience wrapper around load_model_with_lora_async.

    Args:
        model_id: HuggingFace model ID or local path
        lora_config: LoRA configuration
        dtype: Data type for weights
        adapter_path: Optional path to pre-trained adapter weights

    Returns:
        LoadedModelWithLoRA with model, tokenizer, config, and lora_layers
    """
    return _load_model_with_lora_impl(model_id, lora_config, dtype=dtype, adapter_path=adapter_path)


# ============================================================================
# Tuple API (for backwards compatibility with _load_model_sync pattern)
# ============================================================================


def load_model_tuple(
    model_id: str,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> tuple[Any, Any, Any]:
    """
    Load a model and return (model, tokenizer, config) tuple.

    Backwards-compatible API for code expecting tuple returns.

    Args:
        model_id: HuggingFace model ID or local path
        dtype: Data type for weights
        adapter_path: Optional path to LoRA adapter directory

    Returns:
        Tuple of (model, tokenizer, config)
    """
    result = _load_model_impl(model_id, dtype=dtype, adapter_path=adapter_path)
    return result.model, result.tokenizer, result.config


# ============================================================================
# Implementation
# ============================================================================


def _load_model_impl(
    model_id: str,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModel:
    """Internal implementation of model loading."""
    from ..inference.loader import DType, HFLoader
    from .families.registry import detect_model_family, get_family_info

    # Convert dtype enum
    hf_dtype = DType(dtype.value)

    # Download/locate model
    logger.info(f"Loading model: {model_id}")
    result = HFLoader.download(model_id)
    model_path = result.model_path

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Detect model family
    family_type = detect_model_family(config_data)
    if family_type is None:
        model_type = config_data.get("model_type", "unknown")
        raise ValueError(
            f"Unsupported model family: {model_type}. "
            f"Supported: gemma, llama, mistral, qwen3, granite, jamba, mamba, etc."
        )

    # Get family-specific classes
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    # Load weights
    HFLoader.apply_weights_to_model(model, model_path, config, dtype=hf_dtype)

    # Load tokenizer
    tokenizer = HFLoader.load_tokenizer(model_path)

    # Apply adapter if provided
    if adapter_path is not None:
        _apply_adapter_weights(model, Path(adapter_path))

    logger.info(f"Loaded {family_type} model from {model_path}")

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_path=model_path,
        family_type=family_type,
    )


def _load_model_with_lora_impl(
    model_id: str,
    lora_config: LoRAConfig,
    *,
    dtype: ModelDType = ModelDType.BFLOAT16,
    adapter_path: Path | str | None = None,
) -> LoadedModelWithLoRA:
    """Internal implementation of model loading with LoRA."""
    from .adapters.lora import apply_lora, count_lora_parameters

    # Load base model (without adapter - we'll apply fresh LoRA)
    base_result = _load_model_impl(model_id, dtype=dtype, adapter_path=None)

    # Apply LoRA adapters
    lora_layers = apply_lora(base_result.model, lora_config)
    param_count = count_lora_parameters(lora_layers)

    logger.info(f"Applied LoRA to {len(lora_layers)} layers ({param_count:,} params)")

    # Load pre-trained adapter weights if provided
    if adapter_path is not None:
        _load_adapter_weights_into_lora(lora_layers, Path(adapter_path))

    return LoadedModelWithLoRA(
        model=base_result.model,
        tokenizer=base_result.tokenizer,
        config=base_result.config,
        model_path=base_result.model_path,
        family_type=base_result.family_type,
        lora_layers=lora_layers,
        lora_parameter_count=param_count,
    )


def _apply_adapter_weights(model: nn.Module, adapter_path: Path) -> None:
    """Apply adapter weights to a model (for inference)."""
    from .adapters.lora import apply_lora

    # Load adapter config
    adapter_cfg = AdapterConfig.from_directory(adapter_path)

    # Create LoRAConfig from adapter config
    from .adapters.lora import LoRAConfig

    lora_config = LoRAConfig(
        rank=adapter_cfg.rank,
        alpha=adapter_cfg.alpha,
        target_modules=adapter_cfg.target_modules,
    )

    # Apply LoRA structure
    lora_layers = apply_lora(model, lora_config)

    # Load weights
    _load_adapter_weights_into_lora(lora_layers, adapter_path)


def _load_adapter_weights_into_lora(
    lora_layers: dict[str, LoRALinear],
    adapter_path: Path,
) -> None:
    """Load adapter weights into existing LoRA layers."""
    # Find weights file
    weights_path = None
    for name in ["adapters.safetensors", "adapter.safetensors", "lora.safetensors"]:
        candidate = adapter_path / name
        if candidate.exists():
            weights_path = candidate
            break

    if weights_path is None:
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_path}. Expected: adapters.safetensors"
        )

    logger.info(f"Loading adapter weights from {weights_path}")
    weights = mx.load(str(weights_path))

    # Map weights to LoRA layers
    loaded_count = 0
    for name, lora_layer in lora_layers.items():
        # Try different key patterns
        patterns = [
            (f"model.{name}.lora_a", f"model.{name}.lora_b"),
            (f"{name}.lora_a", f"{name}.lora_b"),
            (f"lora.{name}.lora_a", f"lora.{name}.lora_b"),
        ]

        for a_key, b_key in patterns:
            if a_key in weights and b_key in weights:
                lora_layer.lora_A = weights[a_key]
                lora_layer.lora_B = weights[b_key]
                loaded_count += 1
                break

    logger.info(f"Loaded weights for {loaded_count}/{len(lora_layers)} LoRA layers")


def save_adapter(
    lora_layers: dict[str, LoRALinear],
    output_path: Path | str,
    *,
    lora_config: LoRAConfig | None = None,
) -> None:
    """
    Save LoRA adapter weights in standard format.

    Saves:
    - adapters.safetensors: The LoRA weights
    - adapter_config.json: Configuration metadata

    Args:
        lora_layers: LoRA layers from load_model_with_lora or apply_lora
        output_path: Directory to save adapter
        lora_config: Optional config to save (for reproducibility)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect weights
    weights = {}
    for name, lora_layer in lora_layers.items():
        weights[f"model.{name}.lora_a"] = lora_layer.lora_A
        weights[f"model.{name}.lora_b"] = lora_layer.lora_B

    # Save weights
    weights_path = output_path / "adapters.safetensors"
    mx.save_safetensors(str(weights_path), weights)
    logger.info(f"Saved adapter weights to {weights_path}")

    # Save config
    if lora_config is not None:
        config_data = {
            "lora_parameters": {
                "rank": lora_config.rank,
                "alpha": lora_config.alpha,
                "target_modules": lora_config.target_modules,
            }
        }
        config_path = output_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved adapter config to {config_path}")


# ============================================================================
# Legacy compatibility - will be removed
# ============================================================================


def create_model(
    model_type: str,
    config: Any | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create a model from type and config.

    DEPRECATED: Use load_model() instead.
    """
    from .core.config import ModelConfig
    from .core.registry import get_factory

    factory = get_factory(model_type)
    if factory is None:
        raise ValueError(f"Unknown model type: {model_type}")

    if config is None:
        config = ModelConfig(**kwargs)
    elif isinstance(config, dict):
        config = ModelConfig(**{**config, **kwargs})

    return factory(config)


def create_from_preset(preset: str, model_type: str = "llama") -> Any:
    """
    Create model from a preset configuration.

    DEPRECATED: Use load_model() with a HuggingFace model ID instead.
    """
    if preset.startswith("llama") or preset.startswith("mistral"):
        from .families.llama import LlamaConfig, LlamaForCausalLM

        preset_method = getattr(LlamaConfig, preset, None)
        if preset_method:
            config = preset_method()
            return LlamaForCausalLM(config)

    if preset.startswith("mamba"):
        from .families.mamba import MambaConfig, MambaForCausalLM

        preset_method = getattr(MambaConfig, preset, None)
        if preset_method:
            config = preset_method()
            return MambaForCausalLM(config)

    raise ValueError(f"Unknown preset: {preset}")
