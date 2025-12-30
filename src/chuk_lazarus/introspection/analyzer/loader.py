"""
Model loading utilities for the analyzer.

This module handles loading models from HuggingFace and detecting
model properties like quantization and architecture family.
"""

from __future__ import annotations

import json
import math
from typing import Any


def _is_quantized_model(config_data: dict, model_id: str) -> bool:
    """Check if a model is quantized."""
    # Check config for quantization settings
    if "quantization_config" in config_data:
        return True

    # Check model ID patterns
    model_id_lower = model_id.lower()
    quant_patterns = ["-4bit", "-8bit", "4bit", "8bit", "-bnb-", "-awq"]
    return any(pattern in model_id_lower for pattern in quant_patterns)


def _load_model_sync(
    model_id: str,
) -> tuple[Any, Any, Any]:
    """
    Load model synchronously using the models_v2 registry.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        Tuple of (model, tokenizer, config)
    """
    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info

    # Download/locate model
    result = HFLoader.download(model_id)
    model_path = result.model_path

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Detect family and load appropriately
    family_type = detect_model_family(config_data)

    if family_type is None:
        raise ValueError(
            f"Unsupported model family for model_type={config_data.get('model_type')}. "
            f"Supported: gemma, llama, mistral, qwen3, granite, jamba, etc."
        )

    family_info = get_family_info(family_type)
    config_class = family_info.config_class
    model_class = family_info.model_class

    # Create config and model
    config = config_class.from_hf_config(config_data)
    model = model_class(config)

    # Load weights
    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

    # Load tokenizer
    tokenizer = HFLoader.load_tokenizer(model_path)

    # For Gemma models, attach embedding scale for logit lens
    if "gemma" in config_data.get("model_type", "").lower():
        hidden_size = config_data.get("hidden_size", 2048)
        embedding_scale = math.sqrt(hidden_size)
        model._embedding_scale_for_hooks = embedding_scale

    return model, tokenizer, config


def get_model_hidden_size(model: Any, config: Any | None = None) -> int:
    """
    Get hidden size from model or config.

    Args:
        model: The model
        config: Optional config object

    Returns:
        Hidden dimension size
    """
    # Try config first
    if config is not None:
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "d_model"):
            return config.d_model

    # Try model attributes
    if hasattr(model, "model") and hasattr(model.model, "hidden_size"):
        return model.model.hidden_size

    if hasattr(model, "args") and hasattr(model.args, "hidden_size"):
        return model.args.hidden_size

    # Fallback
    return 4096


def get_model_num_layers(model: Any, config: Any | None = None) -> int:
    """
    Get number of layers from model or config.

    Args:
        model: The model
        config: Optional config object

    Returns:
        Number of transformer layers
    """
    # Try config first
    if config is not None:
        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers
        if hasattr(config, "num_layers"):
            return config.num_layers

    # Try model structure
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)

    if hasattr(model, "layers"):
        return len(model.layers)

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)

    # Fallback
    return 32


def get_model_vocab_size(model: Any, tokenizer: Any, config: Any | None = None) -> int:
    """
    Get vocabulary size from model, tokenizer, or config.

    Args:
        model: The model
        tokenizer: The tokenizer
        config: Optional config object

    Returns:
        Vocabulary size
    """
    # Try config first
    if config is not None and hasattr(config, "vocab_size"):
        return config.vocab_size

    # Try tokenizer
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size

    if hasattr(tokenizer, "__len__"):
        return len(tokenizer)

    # Try model
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.shape[0]

    # Fallback
    return 32000
