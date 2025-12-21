"""
Model loading and architectures.

This module provides:
- Unified model loading from HuggingFace or local paths
- LoRA adapter support
- Model configuration
- Architecture implementations (Llama, Mistral, Gemma, Starcoder2, Granite, etc.)
- MLP implementations (SwiGLU, GELU-GLU, ReLU)
- Transformer blocks
"""

# Configuration
# Inference (re-exported from chuk_lazarus.inference for convenience)
from chuk_lazarus.inference import generate_response, generate_sequence

# Loss functions
from .chuk_loss_function import chukloss
from .config import LoRAConfig, ModelConfig

# Weights
from .load_weights import (
    load_checkpoint_weights,
    load_model_weights,
    load_safetensors,
)

# Loading
from .loader import load_model, load_tokenizer

# LoRA
from .lora import LoRALinear, apply_lora
from .loss_function_loader import load_loss_function

# MLX adapter
from .mlx_adapter import MLXAdapter

__all__ = [
    # Inference
    "generate_response",
    "generate_sequence",
    # Loss
    "chukloss",
    # Config
    "LoRAConfig",
    "ModelConfig",
    # Weights
    "load_checkpoint_weights",
    "load_model_weights",
    "load_safetensors",
    # Loading
    "load_model",
    "load_tokenizer",
    # LoRA
    "LoRALinear",
    "apply_lora",
    "load_loss_function",
    # MLX
    "MLXAdapter",
]
