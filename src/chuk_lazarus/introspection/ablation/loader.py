"""
Model loading utilities for ablation studies.

This module handles loading models and creating ModelAdapter instances.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import ModelAdapter


def load_model_for_ablation(model_id: str) -> ModelAdapter:
    """
    Load a model and wrap it in a ModelAdapter.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        ModelAdapter wrapping the loaded model
    """
    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info
    from .adapter import ModelAdapter

    # Download/locate model
    result = HFLoader.download(model_id)
    model_path = result.model_path

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Detect family
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

    return ModelAdapter(model, tokenizer, config)
