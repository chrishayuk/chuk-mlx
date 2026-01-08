"""
Model loading utilities for the analyzer.

This module provides backwards-compatible wrappers around the centralized
model loader in models_v2.loader.

All new code should use:
    from chuk_lazarus.models_v2 import load_model, load_model_tuple
"""

from __future__ import annotations

import math
from typing import Any

from ...models_v2.loader import load_model_tuple as _central_load


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
    adapter_path: str | None = None,
) -> tuple[Any, Any, Any]:
    """
    Load model synchronously.

    This is a thin wrapper around the centralized loader in models_v2.

    Args:
        model_id: HuggingFace model ID or local path
        adapter_path: Optional path to LoRA adapter weights

    Returns:
        Tuple of (model, tokenizer, config)

    Note:
        New code should use:
            from chuk_lazarus.models_v2 import load_model_tuple
            model, tokenizer, config = load_model_tuple(model_id, adapter_path=adapter_path)
    """
    from pathlib import Path

    adapter = Path(adapter_path) if adapter_path else None
    model, tokenizer, config = _central_load(model_id, adapter_path=adapter)

    # For Gemma models, attach embedding scale for logit lens
    if config is not None:
        model_type = getattr(config, "model_type", "")
        if "gemma" in str(model_type).lower():
            hidden_size = getattr(config, "hidden_size", 2048)
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
